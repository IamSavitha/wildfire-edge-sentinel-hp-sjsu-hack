"""Response cache for the cloud-only baseline, so re-runs never re-bill the same request.

A JSONL file (default data/cloud_cache.jsonl, which is git-ignored) keyed by sha256 of everything
that changes the answer: provider host, model, response format, the prompts (system + user + schema
hint), max_tokens, and the image bytes. Each line stores the parsed context (or null for a parse
failure), the provider-reported usage and the latency measured when the call was really made.
Request errors (no response at all) are not cached, so they are retried on the next run.

`CachedVLM` wraps a ContextVLM. Transient provider errors (rate limit, timeout, connection, 5xx) are
retried up to 3 times after sleeping Retry-After or 2^n s; the sleeps are not timed and only the final
attempt's latency is recorded. After each classify() it exposes `last_cached` (True for a cache hit),
`last_latency_ms` (for a hit: the latency measured when the call was originally made, which must be
reported separately from fresh latency), `last_usage` and `last_error`.
"""
import hashlib
import json
import time
from pathlib import Path

from sentinel.schema import ContextResult


def _fingerprint(vlm) -> dict:
    if hasattr(vlm, "fingerprint"):
        return vlm.fingerprint()
    return {"model": vlm.model, "mode": getattr(vlm, "response_format", "json_schema")}


def cache_key(fingerprint: dict, jpeg: bytes) -> str:
    h = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode())
    h.update(b"\0")
    h.update(jpeg)
    return h.hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if not self.path.exists():
            return
        text = self.path.read_text()
        good, damaged = [], not text.endswith("\n") and bool(text)
        for line in text.splitlines():
            try:
                rec = json.loads(line)
            except ValueError:  # a line cut short by an interrupted run
                damaged = True
                continue
            if isinstance(rec, dict) and "key" in rec:
                self.records[rec["key"]] = rec
                good.append(line)
        if damaged:  # repair, so the next append starts on a clean line
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text("".join(g + "\n" for g in good))
            tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, key: str, record: dict) -> None:
        rec = {"key": key, **record}
        self.records[key] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


class FreshCallLimit(SystemExit):
    """Raised before a fresh (billed) call beyond max_fresh_calls; the cache keeps the progress."""


def is_transient(exc) -> bool:
    """Rate limit, timeout, connection error or provider 5xx: worth retrying."""
    if exc is None:
        return False
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                            openai.InternalServerError))


def retry_delay_s(exc, attempt: int) -> float:
    """Retry-After (seconds) when the provider sends it, else 2^attempt."""
    response = getattr(exc, "response", None)
    try:
        value = response.headers.get("retry-after") if response is not None else None
        if value is not None:
            return max(0.0, float(value))
    except (TypeError, ValueError, AttributeError):
        pass
    return float(2 ** attempt)


class CachedVLM:
    def __init__(self, vlm, cache: ResponseCache, clock=time.perf_counter,
                 max_fresh_calls: int | None = None, max_consecutive_errors: int | None = 5,
                 max_retries: int = 3, sleep=time.sleep):
        self.vlm = vlm
        self.cache = cache
        self.clock = clock
        self.sleep = sleep
        self.max_fresh_calls = max_fresh_calls
        self.max_consecutive_errors = max_consecutive_errors
        self.max_retries = max_retries
        self.model = vlm.model
        self.mode = getattr(vlm, "response_format", "json_schema")
        self.fingerprint = _fingerprint(vlm)
        self.n_fresh = self.n_cached = self.n_errors = self.n_retries = 0
        self._consecutive_errors = 0
        self.last_cached = False
        self.last_latency_ms: float | None = None
        self.last_usage: dict = {}
        self.last_error: str | None = None

    def classify(self, jpeg: bytes) -> tuple[ContextResult | None, int]:
        key = cache_key(self.fingerprint, jpeg)
        rec = self.cache.get(key)
        if rec is not None:
            self.n_cached += 1
            self.last_cached, self.last_error = True, None
            self.last_latency_ms = rec.get("latency_ms")
            self.last_usage = dict(rec.get("usage") or {})
            ctx = ContextResult.model_validate(rec["context"]) if rec.get("context") else None
            return ctx, int(rec.get("tokens") or 0)

        if self.max_fresh_calls is not None and self.n_fresh + self.n_errors >= self.max_fresh_calls:
            raise FreshCallLimit(f"stopping before fresh call {self.n_fresh + self.n_errors + 1}: "
                                 f"--max-fresh-calls {self.max_fresh_calls} reached "
                                 "(answers so far are cached; re-run to continue)")
        for attempt in range(1, self.max_retries + 2):
            t0 = self.clock()
            ctx, tokens = self.vlm.classify(jpeg)
            latency_ms = (self.clock() - t0) * 1000  # the final attempt only; sleeps are not timed
            exc = getattr(self.vlm, "last_exception", None)
            if getattr(self.vlm, "last_error", None) is None or not is_transient(exc) or attempt > self.max_retries:
                break
            self.n_retries += 1
            self.sleep(retry_delay_s(exc, attempt))
        self.last_cached = False
        self.last_latency_ms = latency_ms
        self.last_usage = dict(getattr(self.vlm, "last_usage", {}) or {})
        self.last_error = getattr(self.vlm, "last_error", None)
        if self.last_error is not None or not self.last_usage.get("calls", 1):
            self.n_errors += 1
            self._consecutive_errors += 1
            if self.max_consecutive_errors and self._consecutive_errors >= self.max_consecutive_errors:
                raise SystemExit(f"{self._consecutive_errors} cloud requests failed in a row; last error: "
                                 f"{self.last_error} (answers so far are cached). Check the CLOUD_VLM_* "
                                 "variables; a 400 often means the provider does not accept "
                                 "response_format json_schema: try --response-format json_object or none")
            return ctx, tokens
        self._consecutive_errors = 0
        self.n_fresh += 1
        self.cache.put(key, {**self.fingerprint,
                             "context": ctx.model_dump() if ctx is not None else None,
                             "tokens": int(tokens), "usage": self.last_usage,
                             "latency_ms": latency_ms, "created_unix": time.time()})
        return ctx, tokens

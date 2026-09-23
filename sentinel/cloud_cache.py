"""Response cache for the cloud-only baseline, so re-runs never re-bill the same request.

A JSONL file (default data/cloud_cache.jsonl, which is git-ignored) keyed by
sha256(model + response_format + image bytes). Each line stores the parsed context (or null for a
parse failure), the provider-reported usage and the latency measured when the call was really made.
Request errors (no response at all) are not cached, so they are retried on the next run.

`CachedVLM` wraps a ContextVLM. After each classify() it exposes `last_cached` (True for a cache
hit), `last_latency_ms` (for a hit: the latency measured when the call was originally made, which
must be reported separately from fresh latency), `last_usage` and `last_error`.
"""
import hashlib
import json
import time
from pathlib import Path

from sentinel.schema import ContextResult


def cache_key(model: str, mode: str, jpeg: bytes) -> str:
    h = hashlib.sha256()
    for part in (model.encode(), b"\0", mode.encode(), b"\0", jpeg):
        h.update(part)
    return h.hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except ValueError:  # a line cut short by an interrupted run
                    continue
                if isinstance(rec, dict) and "key" in rec:
                    self.records[rec["key"]] = rec

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


class CachedVLM:
    def __init__(self, vlm, cache: ResponseCache, clock=time.perf_counter,
                 max_fresh_calls: int | None = None, max_consecutive_errors: int | None = 5):
        self.vlm = vlm
        self.cache = cache
        self.clock = clock
        self.max_fresh_calls = max_fresh_calls
        self.max_consecutive_errors = max_consecutive_errors
        self.model = vlm.model
        self.mode = getattr(vlm, "response_format", "json_schema")
        self.n_fresh = self.n_cached = self.n_errors = 0
        self._consecutive_errors = 0
        self.last_cached = False
        self.last_latency_ms: float | None = None
        self.last_usage: dict = {}
        self.last_error: str | None = None

    def classify(self, jpeg: bytes) -> tuple[ContextResult | None, int]:
        key = cache_key(self.model, self.mode, jpeg)
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
        t0 = self.clock()
        ctx, tokens = self.vlm.classify(jpeg)
        latency_ms = (self.clock() - t0) * 1000
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
        self.cache.put(key, {"model": self.model, "mode": self.mode,
                             "context": ctx.model_dump() if ctx is not None else None,
                             "tokens": int(tokens), "usage": self.last_usage,
                             "latency_ms": latency_ms, "created_unix": time.time()})
        return ctx, tokens

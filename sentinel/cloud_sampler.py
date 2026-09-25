"""Measured cloud samples for the console's Edge vs Cloud page (optional, off without a provider key).

The shadow ledger (sentinel/shadow.py) MODELS what cloud-only would do with every frame. This puts
real numbers next to it: for a few events (each incident's opening frame, at most `cap_per_hour`)
the same full frame goes to the hosted model a cloud-only design would use, and the answer, latency
and tokens are compared with the edge's. Calls run in the background, never in a decision's path,
and go through CachedVLM, so the same frame is never billed twice. Without the CLOUD_VLM_* settings
(environment, or ~/.cloud_vlm.env) there is no sampler and the page says "modeled"."""
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Callable, Mapping

from sentinel.imaging import to_jpeg

log = logging.getLogger(__name__)

ENV_FILE = Path.home() / ".cloud_vlm.env"
CACHE_PATH = "data/cloud_cache.jsonl"
FULL_FRAME_MAX_SIDE = 1280
SAMPLES_SHOWN = 10


def _full_frame_jpeg(frame) -> bytes:
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    import cv2
    h, w = frame.shape[:2]
    s = min(1.0, FULL_FRAME_MAX_SIDE / max(h, w, 1))
    if s < 1:
        frame = cv2.resize(frame, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return to_jpeg(frame)


def _thread(fn, *args) -> None:
    threading.Thread(target=fn, args=args, name="cloud-sample", daemon=True).start()


class CloudSampler:
    def __init__(self, vlm, cap_per_hour: int = 20, clock: Callable[[], float] = time.time,
                 run: Callable = _thread, host: str | None = None):
        self.vlm = vlm
        self.cap_per_hour = cap_per_hour
        self.clock = clock
        self.run = run
        self.host = host or getattr(getattr(vlm, "vlm", vlm), "host", None) or "cloud"
        self.lock = threading.Lock()
        self.seen: set[str] = set()
        self.calls: deque = deque()           # start times of calls in the last hour
        self.samples: list[dict] = []
        self.skipped_cap = 0
        self.errors = 0

    def maybe_sample(self, key: str, frame, edge_ctx: dict | None) -> bool:
        """Send this event's frame once, within the hourly cap. Returns whether a call was started."""
        now = self.clock()
        with self.lock:
            if key in self.seen:
                return False
            while self.calls and now - self.calls[0] > 3600:
                self.calls.popleft()
            if len(self.calls) >= self.cap_per_hour:
                self.skipped_cap += 1
                return False
            self.seen.add(key)
            self.calls.append(now)
        self.run(self._sample, key, _full_frame_jpeg(frame), edge_ctx or {})
        return True

    def _sample(self, key: str, jpeg: bytes, edge_ctx: dict) -> None:
        t0 = time.perf_counter()
        try:
            ctx, tokens = self.vlm.classify(jpeg)
        except BaseException as exc:  # noqa: BLE001 - CachedVLM may raise SystemExit; never kill the server
            with self.lock:
                self.errors += 1
            log.warning("cloud sample %s failed: %s", key, type(exc).__name__)
            return
        ms = getattr(self.vlm, "last_latency_ms", None)
        rec = {"key": key, "at": self.clock(), "cloud_ms": ms if ms is not None else (time.perf_counter() - t0) * 1000,
               "cloud_tokens": int(tokens or 0), "cloud_source_type": ctx.source_type if ctx else None,
               "edge_source_type": edge_ctx.get("source_type"), "cached": bool(getattr(self.vlm, "last_cached", False)),
               "jpeg_bytes": len(jpeg)}
        rec["agree"] = rec["cloud_source_type"] is not None and rec["cloud_source_type"] == rec["edge_source_type"]
        with self.lock:
            if ctx is None:
                self.errors += 1
            self.samples.append(rec)

    def summary(self) -> dict:
        with self.lock:
            done = [s for s in self.samples if s["cloud_source_type"] is not None]
            recent = list(reversed(self.samples[-SAMPLES_SHOWN:]))
            errors, capped = self.errors, self.skipped_cap
        return {"enabled": True, "host": self.host, "n": len(done), "errors": errors, "skipped_cap": capped,
                "cap_per_hour": self.cap_per_hour,
                "agreement_pct": 100 * sum(s["agree"] for s in done) / len(done) if done else None,
                "cloud_ms_p50": float(median(s["cloud_ms"] for s in done)) if done else None,
                "cloud_tokens_p50": float(median(s["cloud_tokens"] for s in done)) if done else None,
                "cached": sum(s["cached"] for s in done), "samples": recent}


def read_env_file(path: Path) -> dict[str, str]:
    """KEY=VALUE lines (optionally `export KEY=VALUE`, quotes stripped); nothing is exported or logged."""
    out = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


def sampler_from_env(env: Mapping[str, str] | None = None, env_file: Path | None = ENV_FILE,
                     cache_path: str = CACHE_PATH, cap_per_hour: int = 20) -> CloudSampler | None:
    """A sampler when CLOUD_VLM_BASE_URL, _MODEL and _API_KEY are set (environment first, then the file)."""
    merged = {**(read_env_file(env_file) if env_file else {}), **(os.environ if env is None else env)}
    if not all((merged.get(f"CLOUD_VLM_{k}") or "").strip() for k in ("BASE_URL", "MODEL", "API_KEY")):
        return None
    from sentinel.cloud_cache import CachedVLM, ResponseCache
    from sentinel.vlm_client import CLOUD_FULL_FRAME_PROMPT, cloud_vlm_from_env
    vlm = cloud_vlm_from_env(env=merged, system_prompt=CLOUD_FULL_FRAME_PROMPT)
    return CloudSampler(CachedVLM(vlm, ResponseCache(cache_path), max_consecutive_errors=None),
                        cap_per_hour=cap_per_hour, host=vlm.host)


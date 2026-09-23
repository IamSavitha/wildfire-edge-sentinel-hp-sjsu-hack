"""Live model-economics dashboard for the zrt-served vLLM backends on the Nano.
`python -m sentinel.monitor` then open http://localhost:8090 (SSH-tunnel the port from a laptop)."""
import argparse
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from sentinel.live_metrics import (TOTAL_KEYS, accumulate_totals, discover_models, economics,
                                   effective_prices, histogram_buckets, model_snapshot,
                                   parse_prometheus, rates, system_stats, windowed_quantiles)

PAGE = Path(__file__).parent / "static" / "monitor.html"
FETCH_TIMEOUT_S = 1.0
POLL_INTERVAL_S = 2.0

SNAPSHOT_KEYS = ("requests", "prompt_tokens", "generation_tokens", "running", "kv_cache_pct",
                 "latency_p50_s", "latency_p95_s", "ttft_p50_s", "latency_window")
RATE_KEYS = ("prompt_tok_per_s", "gen_tok_per_s", "req_per_s")

log = logging.getLogger(__name__)


def fetch_uds_metrics(sock_path: str) -> str:
    """GET /metrics from a vLLM backend over its zrt unix socket."""
    with httpx.Client(transport=httpx.HTTPTransport(uds=sock_path), timeout=FETCH_TIMEOUT_S) as c:
        r = c.get("http://localhost/metrics")
        r.raise_for_status()
        return r.text


def fetch_json(url: str) -> dict:
    r = httpx.get(url, timeout=FETCH_TIMEOUT_S)
    r.raise_for_status()
    return r.json()


class PriceUpdate(BaseModel):
    """Session-only price overrides. All values are user-supplied assumptions."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    usd_per_mtok_in: float | None = Field(None, ge=0)
    usd_per_mtok_out: float | None = Field(None, ge=0)
    usd_per_mtok: float | None = Field(None, ge=0)
    usd_per_gb: float | None = Field(None, ge=0)
    tokens_per_full_frame: float | None = Field(None, ge=0)
    bytes_per_frame: float | None = Field(None, ge=0)


def _load_prices(path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:200]


def _outcome(f, timeout: BaseException):
    """A finished future's result or exception; the timeout error if it is still running."""
    if not f.done():
        f.cancel()
        return timeout
    if f.cancelled():
        return timeout
    return f.exception() or f.result()


class _Model:
    """Per-label state carried between polls."""

    def __init__(self):
        self.ts: float | None = None
        self.raw: dict | None = None        # raw counters from the previous poll (for rates/resets)
        self.hists: dict | None = None      # cumulative histogram buckets from the previous poll
        self.windowed: dict | None = None   # last windowed quantiles (held while idle)
        self.offsets: dict = {}             # counter-reset offsets so totals never decrease
        self.last_good: dict | None = None  # last adjusted snapshot (for economics when unreachable)


class Monitor:
    """Polls every backend in parallel off the request path; requests read the stored payload."""

    def __init__(self, run_dir, prices: dict, pipeline_url: str | None, fetch, clock,
                 fetch_pipeline, system, interval_s: float = POLL_INTERVAL_S,
                 fetch_timeout_s: float = FETCH_TIMEOUT_S):
        self.run_dir, self.prices, self.pipeline_url = run_dir, prices, pipeline_url
        self.fetch, self.clock, self.fetch_pipeline, self.system = fetch, clock, fetch_pipeline, system
        self.interval_s, self.fetch_timeout_s = interval_s, fetch_timeout_s
        self.lock = threading.Lock()
        self.models: dict[str, _Model] = {}
        self.payload: dict | None = None
        self.last_system: dict = {}
        self.pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="monitor-fetch")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- polling (network I/O outside the lock) ----
    def poll_once(self) -> dict:
        now = self.clock()
        metas = discover_models(self.run_dir)
        futures = {m["label"]: self.pool.submit(self.fetch, m["sock"]) for m in metas if m.get("sock")}
        pipe_f = self.pool.submit(self.fetch_pipeline, self.pipeline_url) if self.pipeline_url else None
        sys_f = self.pool.submit(self.system)
        pending = list(futures.values()) + [f for f in (pipe_f, sys_f) if f]
        wait(pending, timeout=self.fetch_timeout_s)

        timeout = TimeoutError(f"no response in {self.fetch_timeout_s:g} s")
        texts = {label: _outcome(f, timeout) for label, f in futures.items()}
        pipeline = None
        state = _outcome(pipe_f, timeout) if pipe_f is not None else None
        if isinstance(state, dict):
            metrics = state.get("metrics", state)
            pipeline = metrics if isinstance(metrics, dict) else None
        system = _outcome(sys_f, timeout)

        with self.lock:
            if isinstance(system, dict):
                self.last_system = system     # a slow nvidia-smi keeps the previous reading
            system = self.last_system
            rows = [self._update(m, texts.get(m["label"]), now) for m in metas]
            self.payload = {"ts": now, "models": rows, "system": system, "pipeline": pipeline}
            return self._with_economics()

    def _update(self, meta: dict, text, now: float) -> dict:
        label = meta["label"]
        row = {k: meta.get(k) for k in ("label", "model_uri", "gpu_memory_fraction", "started_at")}
        st = self.models.setdefault(label, _Model())
        if not isinstance(text, str):
            err = text if isinstance(text, BaseException) else FileNotFoundError("no socket in run dir")
            row.update({k: None for k in SNAPSHOT_KEYS + RATE_KEYS})
            row.update(status="unreachable", error=_err(err))
            return row
        samples = parse_prometheus(text)
        names = {lab.get("model_name") for _, lab, _ in samples}
        # one backend per socket: if the served name differs from the zrt label, take everything
        name = label if label in names else None
        snap = model_snapshot(samples, name)
        raw = {k: snap[k] for k in TOTAL_KEYS}
        hists = histogram_buckets(samples, name)

        r = rates(st.raw, raw, now - st.ts if st.ts is not None else 0.0)
        windowed = windowed_quantiles(st.hists, hists, st.windowed)
        adjusted, st.offsets = accumulate_totals(st.raw, raw, st.offsets)
        st.ts, st.raw, st.hists, st.windowed = now, raw, hists, windowed

        snap.update(windowed)
        snap.update(adjusted)
        st.last_good = snap
        row.update(snap)
        row.update(r)
        row["status"] = "ok"
        return row

    def _with_economics(self) -> dict:
        """Stored payload + economics at the current prices (so a price change shows at once)."""
        good = {label: st.last_good for label, st in self.models.items() if st.last_good}
        return {**self.payload,
                "economics": economics(good, self.payload["pipeline"], self.prices),
                "prices": effective_prices(self.prices)}

    def latest(self) -> dict:
        with self.lock:
            if self.payload is not None:
                return self._with_economics()
        return self.poll_once()

    def set_prices(self, update: dict) -> dict:
        with self.lock:
            self.prices.update(update)
            return effective_prices(self.prices)

    # ---- background poller ----
    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - keep polling whatever one cycle hit
                log.exception("monitor poll failed")
            self._stop.wait(self.interval_s)

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="monitor-poller", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.pool.shutdown(wait=False, cancel_futures=True)


def create_monitor_app(run_dir, prices_path, pipeline_url: str | None = None,
                       fetch: Callable[[str], str] | None = None,
                       clock: Callable[[], float] = time.time,
                       fetch_pipeline: Callable[[str], dict] | None = None,
                       system: Callable[[], dict] | None = None,
                       interval_s: float = POLL_INTERVAL_S,
                       fetch_timeout_s: float = FETCH_TIMEOUT_S) -> FastAPI:
    mon = Monitor(run_dir, _load_prices(prices_path), pipeline_url, fetch or fetch_uds_metrics, clock,
                  fetch_pipeline or fetch_json, system or system_stats, interval_s, fetch_timeout_s)

    @asynccontextmanager
    async def lifespan(app):
        mon.start()
        yield
        mon.stop()

    app = FastAPI(title="Live Model Economics", lifespan=lifespan)
    app.state.monitor = mon

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text()

    @app.get("/api/live")
    def live():
        return mon.latest()

    @app.post("/api/prices")
    def set_prices(update: PriceUpdate):
        return mon.set_prices(update.model_dump(exclude_none=True))

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--run-dir", default="/opt/hp/zrt/run", help="zrt run dir with vllm-*.json/.sock")
    ap.add_argument("--prices", default="config/cost_inputs.json", help="price/cost inputs JSON")
    ap.add_argument("--pipeline-url", default=None,
                    help="demo app state endpoint, e.g. http://localhost:8080/api/state")
    args = ap.parse_args(argv)
    import uvicorn
    uvicorn.run(create_monitor_app(args.run_dir, args.prices, args.pipeline_url),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

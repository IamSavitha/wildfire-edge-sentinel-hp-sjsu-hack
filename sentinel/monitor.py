"""Live model-economics dashboard for the zrt-served vLLM backends on the Nano.
`python -m sentinel.monitor` then open http://localhost:8090 (SSH-tunnel the port from a laptop)."""
import argparse
import json
import threading
import time
from pathlib import Path
from typing import Callable

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from sentinel.live_metrics import (discover_models, economics, effective_prices, model_snapshot,
                                   parse_prometheus, rates, system_stats)

PAGE = Path(__file__).parent / "static" / "monitor.html"
TIMEOUT_S = 2.0
MIN_RATE_WINDOW_S = 1.0   # several viewers polling at once must not produce ~0 s rate windows

SNAPSHOT_KEYS = ("requests", "prompt_tokens", "generation_tokens", "running", "kv_cache_pct",
                 "latency_p50_s", "latency_p95_s", "ttft_p50_s")
RATE_KEYS = ("prompt_tok_per_s", "gen_tok_per_s", "req_per_s")


def fetch_uds_metrics(sock_path: str) -> str:
    """GET /metrics from a vLLM backend over its zrt unix socket."""
    with httpx.Client(transport=httpx.HTTPTransport(uds=sock_path), timeout=TIMEOUT_S) as c:
        r = c.get("http://localhost/metrics")
        r.raise_for_status()
        return r.text


def fetch_json(url: str) -> dict:
    r = httpx.get(url, timeout=TIMEOUT_S)
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


def create_monitor_app(run_dir, prices_path, pipeline_url: str | None = None,
                       fetch: Callable[[str], str] | None = None,
                       clock: Callable[[], float] = time.time,
                       fetch_pipeline: Callable[[str], dict] | None = None,
                       system: Callable[[], dict] | None = None) -> FastAPI:
    fetch = fetch or fetch_uds_metrics
    fetch_pipeline = fetch_pipeline or fetch_json
    system = system or system_stats
    prices = _load_prices(prices_path)
    lock = threading.Lock()
    # label -> (ts, snapshot) used as the start of the next rate window
    rate_base: dict[str, tuple[float, dict]] = {}
    last_rates: dict[str, dict] = {}
    last_good: dict[str, dict] = {}   # label -> last reachable snapshot (tokens don't un-happen)

    app = FastAPI(title="Live Model Economics")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text()

    def poll_model(meta: dict, now: float) -> dict:
        row = {k: meta.get(k) for k in ("label", "model_uri", "gpu_memory_fraction", "started_at")}
        label = meta["label"]
        try:
            if not meta.get("sock"):
                raise FileNotFoundError("no socket in run dir")
            samples = parse_prometheus(fetch(meta["sock"]))
        except Exception as e:  # noqa: BLE001 - any transport failure means unreachable
            row.update({k: None for k in SNAPSHOT_KEYS + RATE_KEYS})
            row.update(status="unreachable", error=f"{type(e).__name__}: {e}"[:200])
            return row
        names = {lab.get("model_name") for _, lab, _ in samples}
        # one backend per socket: if the served name differs from the zrt label, take everything
        snap = model_snapshot(samples, label if label in names else None)
        base = rate_base.get(label)
        if base is None or now - base[0] >= MIN_RATE_WINDOW_S:
            last_rates[label] = rates(base[1] if base else None, snap, now - base[0] if base else 0.0)
            rate_base[label] = (now, snap)
        last_good[label] = snap
        row.update(snap)
        row.update(last_rates.get(label) or {k: None for k in RATE_KEYS})
        row["status"] = "ok"
        return row

    @app.get("/api/live")
    def live():
        now = clock()
        pipeline = None
        if pipeline_url:
            try:
                state = fetch_pipeline(pipeline_url)
                pipeline = state.get("metrics", state) if isinstance(state, dict) else None
            except Exception:  # noqa: BLE001 - demo app not running is normal
                pipeline = None
        with lock:
            models = [poll_model(m, now) for m in discover_models(run_dir)]
            econ = economics(dict(last_good), pipeline, prices)
            eff = effective_prices(prices)
        return {"ts": now, "models": models, "system": system(), "pipeline": pipeline,
                "economics": econ, "prices": eff}

    @app.post("/api/prices")
    def set_prices(update: PriceUpdate):
        with lock:
            prices.update(update.model_dump(exclude_none=True))
            return effective_prices(prices)

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

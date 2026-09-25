"""Sentinel Console: the whole edge system as one server and one screen.

`python -m sentinel.console` then open http://localhost:8080 (on the Nano, over the local network with
--host 0.0.0.0, or through an SSH tunnel). It needs no internet: the models run on the device, the map
and the UI are served from it, and ALERTs wait in the outbox until the uplink returns.

One process shares one set of services (sentinel.services): one detector load, one VLM client, one
uplink, one outbox (phone push + dispatch) and one cloud-only shadow ledger. The incident map is
mounted at /ops, the image/mobile-camera app at /lab, and the console's own API at /api. A fire seen
by any camera (towers, the mobile camera, a field photo) becomes one incident, one alert and one push."""
import argparse
import json
import logging
import threading
import time
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from sentinel.live import LIVE_TOWER_ID
from sentinel.live_metrics import effective_prices
from sentinel.monitor import PriceUpdate, _load_prices
from sentinel.netprofile import PROFILES
from sentinel.notify import NotifyError
from sentinel.services import Services
from sentinel.webapp_logic import load_benchmarks

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static" / "console"
OFFICER_EXTRAS = ('<link rel="stylesheet" href="/console/officer_ext.css">'
                  '<script src="/console/officer_ext.js"></script>')
HOME = Path.home()
FLUSH_EVERY_S = 2.0
MOBILE_ID = "mobile-1"
ALERTS_SHOWN = 50
DEFAULT_MOBILE = {"lat": 33.2745, "lon": -116.9650}      # a patrol position between the Palomar towers
DEPLOYED = {
    "detector": {"name": "YOLO11s joint", "version": "v1.3.0-joint", "trained_on": "D-Fire + tower (HPWREN/pyro-sdis)",
                 "results": {"tower": "detector_tower_stage3.json", "dfire": "detector_dfire_stage3.json",
                             "before": "detector_before_yoloworld.json"}},
    "vlm": {"name": "Qwen2.5-VL-7B + context LoRA", "version": "v1.3.0-joint",
            "distilled_from": "Qwen2.5-VL-32B-AWQ teacher",
            "results": {"before": "context_before_base7b.json", "context": "context_after_lora7b.json",
                        "context_v2": "context_after_lora7b_v2.json"}},
}
BENCH_FILES = {"edge": "bench_after.json", "edge_recheck30": "bench_after_recheck30.json",
               "cloud": "bench_cloud_qwen7b.json", "outage": "outage_after.json"}


class Uplink(BaseModel):
    online: bool


class ModelChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=80)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _pick(d: dict | None, keys) -> dict | None:
    return {k: d[k] for k in keys if k in d} if d else None


def mobile_camera(lat: float, lon: float, name: str = "Mobile unit 1") -> dict:
    """The mobile camera (laptop webcam / phone) as a camera the map knows: incidents open at its site."""
    return {"id": MOBILE_ID, "name": name, "site": name, "lat": lat, "lon": lon, "elev_m": None,
            "azimuth_deg": 0, "hfov_deg": 60, "wx_station": None, "recordings": [], "mobile": True}


def create_console_app(*, services: Services, map_app: FastAPI, web_app: FastAPI,
                       results_dir: str | Path = "results", prices_path: str | Path = "config/cost_inputs.json",
                       assets_dir: str | Path | None = None, detector_weights: str | None = None,
                       monitor=None, cloud_sampler=None,
                       system_stats: Callable[[], dict] | None = None,
                       flush_every_s: float = FLUSH_EVERY_S) -> FastAPI:
    """Mount the two apps (built on `services`) under one console. Use `build_console` to make them."""
    results_dir = Path(results_dir)
    prices = _load_prices(prices_path)
    clock = services.clock
    delivery = services.delivery
    if system_stats is None:
        from sentinel.live_metrics import system_stats

    # ------------------------------------------------------------ views

    def incidents() -> list[dict]:
        return map_app.state.store.all()

    def incident_by_event() -> dict[str, dict]:
        out = {}
        for inc in incidents():
            eid = (inc.get("escalation") or {}).get("event_id")
            if eid:
                out[eid] = {"id": inc["id"], "name": inc.get("name"), "severity": inc.get("severity")}
        return out

    def alert_rows(limit: int = ALERTS_SHOWN) -> list[dict]:
        with delivery.lock:
            ids = list(delivery.events)[-limit:]
        by_event = incident_by_event()
        rows = []
        for eid in reversed(ids):
            ev = delivery.event(eid)
            if ev is None:
                continue
            rows.append({**ev, "source": "mobile" if ev.get("tower_id") in (LIVE_TOWER_ID, MOBILE_ID) else
                         "field" if ev.get("tower_id") == "field" else "tower",
                         "incident": by_event.get(eid)})
        return rows

    def overview() -> dict:
        incs = incidents()
        active = [i for i in incs if i.get("status") == "active"]
        rows = alert_rows(1)
        s = services.shadow.summary(prices, "lte")
        return {
            "node": services.node_name, "time": clock(),
            "uplink": {"online": services.link.online, "queued": delivery.outbox.pending_count()},
            "incidents": {"active": len(active),
                          "alert": sum(1 for i in active if i.get("severity") == "ALERT"),
                          "monitor": sum(1 for i in active if i.get("severity") == "MONITOR")},
            "last_alert": rows[0] if rows else None,
            "savings": {"bytes": s["savings"]["bytes"], "bytes_x": s["savings"]["bytes_x"],
                        "usd": s["savings"]["usd"], "prices_set": s["prices_set"],
                        "offline_decisions": s["edge_offline_decisions"], "frames": s["edge"]["frames"]},
        }

    def benchmarks() -> dict:
        out = {"before_after": load_benchmarks(results_dir)}
        keys = ("precision", "recall", "false_alarms", "missed", "n_clips", "tokens_per_vlm_call",
                "frames_per_vlm_call", "time_to_decision_s_p50", "time_to_decision_s_p95")
        for name, fname in BENCH_FILES.items():
            out[name] = _pick(_read_json(results_dir / fname), keys + ("bytes_up", "usd", "prompt_tokens"))
        return out

    def model_info() -> dict:
        served, err = services.served_models()
        det = DEPLOYED["detector"]
        dres = {k: _pick(_read_json(results_dir / f), ("map50", "map50_95", "precision", "recall", "ms_per_image",
                                                       "n_images"))
                for k, f in det["results"].items()}
        vres = {k: _pick(_read_json(results_dir / f), ("source_type_acc", "group_acc", "parse_fail_rate",
                                                       "tokens_per_call", "latency_ms_p50", "n", "per_source"))
                for k, f in DEPLOYED["vlm"]["results"].items()}
        active = map_app.state.model_name()
        return {
            "detector": {**{k: v for k, v in det.items() if k != "results"}, "weights": detector_weights,
                         "available": bool(detector_weights and Path(detector_weights).exists()),
                         "headline": {"label": "mAP50 · tower val", "value": (dres.get("tower") or {}).get("map50")},
                         "results": dres},
            "vlm": {**{k: v for k, v in DEPLOYED["vlm"].items() if k != "results"}, "active": active,
                    "served": served, "error": err,
                    "options": [m for m in served if m.startswith("context")] or [active],
                    "headline": {"label": "Source-type agreement with the teacher",
                                 "value": (vres.get(active) or vres.get("context") or {}).get("source_type_acc")},
                    "results": vres},
        }

    def health() -> list[dict]:
        served, err = services.served_models()
        active = map_app.state.model_name()
        pending = delivery.outbox.pending_count()
        det_ok = bool(detector_weights and Path(detector_weights).exists())
        rows = [
            {"id": "detector", "label": "Detector", "state": "ok" if det_ok else "down",
             "note": "Loaded on demand" if det_ok else "Weights not found"},
            {"id": "vlm", "label": "Context VLM",
             "state": "ok" if active in served else "warn" if served else "down",
             "note": f"Serving {active}" if active in served else (err or f"{active} not served")[:80]},
            {"id": "uplink", "label": "Uplink", "state": "ok" if services.link.online else "warn",
             "note": "Online" if services.link.online else "Offline: deciding locally"},
            {"id": "outbox", "label": "Alert outbox", "state": "ok" if not pending else "warn",
             "note": f"{pending} queued" if pending else "Empty"},
            {"id": "phone", "label": "Phone alerts", "state": "ok" if delivery.notifier is not None else "off",
             "note": getattr(delivery.notifier, "host", None) or "Not configured"},
            {"id": "dispatch", "label": "Dispatch", "state": "ok" if delivery.dispatch_fn is not None else "off",
             "note": "Configured" if delivery.dispatch_fn is not None else "Phone only"},
        ]
        return rows

    # ------------------------------------------------------------ app

    @asynccontextmanager
    async def lifespan(app):
        stop = threading.Event()

        def retry_loop():
            while not stop.wait(flush_every_s):
                if services.link.online and delivery.outbox.pending_count():
                    services.flush()
        async with AsyncExitStack() as stack:
            # Starlette does not run a mounted app's lifespan: enter both here
            await stack.enter_async_context(map_app.router.lifespan_context(map_app))
            await stack.enter_async_context(web_app.router.lifespan_context(web_app))
            if monitor is not None:
                monitor.start()
            threading.Thread(target=retry_loop, name="outbox-retry", daemon=True).start()
            app.state.started = True
            try:
                yield
            finally:
                stop.set()
                if monitor is not None:
                    monitor.stop()

    app = FastAPI(title="Sentinel Console", lifespan=lifespan)
    app.state.services = services
    app.state.map = map_app
    app.state.web = web_app
    app.state.started = False

    @app.get("/api/overview")
    def get_overview():
        return overview()

    @app.get("/api/uplink")
    def get_uplink():
        return {"online": services.link.online, "queued": delivery.outbox.pending_count()}

    @app.post("/api/uplink")
    def post_uplink(u: Uplink):
        sent = services.set_online(u.online)
        return {"online": services.link.online, "queued": delivery.outbox.pending_count(), "flushed": sent}

    @app.get("/api/alerts")
    def get_alerts():
        s = delivery.summary()
        return {"alerts": alert_rows(), "online": services.link.online, "queued": s["pending"],
                "delivered": s["delivered"],
                "phone": {"configured": delivery.notifier is not None,
                          "host": getattr(delivery.notifier, "host", None),
                          **({k: v for k, v in s["phone"].items() if k in ("sent", "suppressed", "failed", "deferred")}
                             if delivery.notifier is not None else {})},
                "dispatch_configured": delivery.dispatch_fn is not None}

    @app.post("/api/alerts/test")
    def test_alert():
        if delivery.notifier is None:
            raise HTTPException(409, "Phone alerts are not configured (NTFY_TOPIC_URL).")
        if not services.link.online:
            raise HTTPException(409, "The uplink is down: a test push cannot leave the device.")
        try:
            return {"status": delivery.notifier.send_test()}
        except NotifyError as exc:            # already redacted
            raise HTTPException(502, str(exc)) from None

    @app.get("/api/edge-cloud")
    def edge_cloud(profile: str = "lte"):
        if profile not in PROFILES:
            raise HTTPException(400, f"profile must be one of {sorted(PROFILES)}")
        return {"summary": services.shadow.summary(prices, profile), "profiles": services.shadow.profiles(),
                "prices": effective_prices(prices), "benchmarks": benchmarks(),
                "measured": cloud_sampler.summary() if cloud_sampler is not None else {"enabled": False}}

    @app.post("/api/edge-cloud/prices")
    def set_prices(update: PriceUpdate):
        prices.update(update.model_dump(exclude_none=True))
        return effective_prices(prices)

    @app.get("/api/edge-cloud/fleet")
    def fleet(towers: int = 20, fps: float = 1.0, days: int = 30):
        if not (1 <= towers <= 10_000 and 0 < fps <= 30 and 1 <= days <= 3650):
            raise HTTPException(400, "towers 1-10000, fps 0-30, days 1-3650")
        return services.shadow.fleet(towers, fps, days, prices)

    @app.post("/api/edge-cloud/reset")
    def reset_ledger():
        services.shadow.reset()
        return {"ok": True}

    @app.get("/api/models")
    def get_models():
        return model_info()

    @app.post("/api/models/vlm")
    def set_model(choice: ModelChoice):
        served, _ = services.served_models()
        if choice.model not in served:
            raise HTTPException(400, f"{choice.model!r} is not served; served: {served}")
        services.active_vlm = choice.model
        return model_info()["vlm"]

    @app.get("/api/system")
    def get_system():
        try:
            stats = system_stats()
        except Exception:  # noqa: BLE001 - no GPU tools: the page still works
            stats = {}
        served, err = services.served_models()
        return {"node": services.node_name, "uptime_s": clock() - services.started_at, "health": health(),
                "stats": stats, "served_models": served, "models_error": err,
                "monitor": monitor.latest() if monitor is not None else None}

    app.mount("/ops", map_app)
    app.mount("/lab", web_app)
    if assets_dir is not None:
        for mount, sub in (("/assets", Path(assets_dir) / "web"), ("/tiles", Path(assets_dir) / "maps")):
            if sub.is_dir():
                app.mount(mount, StaticFiles(directory=sub), name=mount.strip("/"))
    if STATIC.is_dir():
        app.mount("/console", StaticFiles(directory=STATIC, html=True), name="console")
    # the officer console (Kruthika's incident map page, with the console's extra tabs) is the home page;
    # mounted last, so every console route above wins, and its own /api/* names do not collide with them
    app.mount("/", map_app)

    return app


def build_console(*, services: Services, cameras: dict[str, dict], mobile: dict | None = None,
                  map_kwargs: dict | None = None, web_kwargs: dict | None = None, cloud_sampler=None,
                  **console_kwargs) -> FastAPI:
    """The two apps on shared `services`, the mobile camera registered as a camera, and the console."""
    from sentinel.config import Tower
    from sentinel.incident_map import create_map_app
    from sentinel.webapp import create_web_app

    mobile = mobile or mobile_camera(**DEFAULT_MOBILE)
    if cloud_sampler is not None:
        services.on_incident_opened = cloud_sampler.maybe_sample
    cameras = {**cameras, mobile["id"]: mobile}
    map_kwargs = dict(map_kwargs or {})
    map_kwargs.setdefault("page_extras", OFFICER_EXTRAS)
    web_kwargs = dict(web_kwargs or {})
    map_app = create_map_app(cameras=cameras, services=services, **map_kwargs)

    def on_live_event(ev: dict, frame: np.ndarray) -> None:
        """A mobile-camera MONITOR/ALERT opens (or updates) an incident at the unit's position. Its ALERT
        was already delivered by the live pipeline through the shared outbox: no second push."""
        eid = ev["id"]
        sev = ev["severity"]
        detected = datetime.fromtimestamp(ev.get("opened_at") or services.clock(), tz=timezone.utc).isoformat()
        report = {"event_id": eid, "tower_id": mobile["id"], "tower_name": mobile["name"], "lat": mobile["lat"],
                  "lon": mobile["lon"], "detected_at": detected, "severity": sev,
                  "confidence": float(ev.get("confidence") or 0), "trend": ev.get("trend"), "temp_c": 25.0,
                  "source_type": ev.get("source_type") or "unknown", "context": None,
                  "description": ev.get("description") or "Smoke seen by the mobile unit.", "forecast": None,
                  "forecast_status": "not_requested", "thumbnail_jpeg_b64": ev.get("thumbnail_b64") or ""}
        esc = None
        if sev == "ALERT":
            st = services.delivery.event(eid) or {}
            esc = {"decision": "sent" if st.get("delivered") else "queued", "event_id": eid,
                   "payload_bytes": st.get("bytes", 0), "bytes_up": st.get("bytes", 0) if st.get("delivered") else 0,
                   "at": services.clock(),
                   "note": "Delivered by the mobile unit's pipeline." if st.get("delivered") else
                   "Held in the outbox until the uplink returns."}
        ctx = {"source_type": ev.get("source_type") or "unknown", "description": ev.get("description")}
        view, created = map_app.state.ingest_external(mobile["id"], severity=sev, context=ctx, report=report,
                                      conf=float(ev.get("confidence") or 0), thumbnail_b64=ev.get("thumbnail_b64") or "",
                                      image=frame, escalation=esc, name="mobile camera")
        if created:
            services.incident_opened(view["id"], frame, ctx)

    web_kwargs.setdefault("tower", Tower(mobile["id"], mobile["name"], mobile["lat"], mobile["lon"], "", temp_c=25.0))
    web_kwargs.setdefault("live_lat", mobile["lat"])
    web_kwargs.setdefault("live_lon", mobile["lon"])
    web_app = create_web_app(services=services, on_live_event=on_live_event, **web_kwargs)
    return create_console_app(services=services, map_app=map_app, web_app=web_app, cloud_sampler=cloud_sampler,
                              detector_weights=map_kwargs.get("detector_weights"),
                              assets_dir=map_kwargs.get("assets_dir"), **console_kwargs)


def main(argv=None):
    from sentinel.detector import imgsz_for
    from sentinel.incident_map import DEFAULT_ASSETS, load_cameras
    from sentinel.services import build_services
    from sentinel.webapp import DETECTORS, PIPELINES, detector_specs
    from sentinel.wind import SensorEmulator

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 serves the local network (no internet needed)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--node-name", default="Sentinel Edge 01")
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS), help="offline map, web libs and state (outside git)")
    ap.add_argument("--state-dir", default=str(HOME / "sentinel-state" / "console"),
                    help="incidents, frames and the alert outbox (never inside the shared assets folder)")
    ap.add_argument("--data-dir", default=str(HOME / "sentinel" / "data"))
    ap.add_argument("--figlib-dir", default=None)
    ap.add_argument("--cameras", default="config/cameras.json")
    ap.add_argument("--detector-weights", default=str(HOME / "sentinel" / "models" / "joint_yolo.pt"))
    ap.add_argument("--photo-detector-weights", default=str(HOME / "sentinel" / "models" / "smoke_yolo.pt"))
    ap.add_argument("--before-weights", default=DETECTORS["yoloworld"]["weights"])
    ap.add_argument("--vlm-base-url", default="unix:///opt/hp/zrt/run/vllm-base7b.sock")
    ap.add_argument("--vlm-model", default="context")
    ap.add_argument("--before-vlm", default="base7b")
    ap.add_argument("--assistant-model", default="base7b")
    ap.add_argument("--vlm-timeout", type=float, default=60.0)
    ap.add_argument("--run-dir", default="/opt/hp/zrt/run", help="zrt run dir (live vLLM metrics)")
    ap.add_argument("--prices", default="config/cost_inputs.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--mobile-lat", type=float, default=DEFAULT_MOBILE["lat"])
    ap.add_argument("--mobile-lon", type=float, default=DEFAULT_MOBILE["lon"])
    ap.add_argument("--live-recheck-s", type=float, default=10.0)
    ap.add_argument("--dispatch-url", default=None)
    ap.add_argument("--start-offline", action="store_true", help="start with the uplink down")
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--no-sensor", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    assets = Path(a.assets)
    state = Path(a.state_dir)
    dispatch_fn = None
    if a.dispatch_url:
        from functools import partial

        from sentinel.cloud import send_dispatch
        dispatch_fn = partial(send_dispatch, a.dispatch_url)
    services = build_services(outbox_path=str(state / "outbox.db"), dispatch_fn=dispatch_fn,
                              vlm_base_url=a.vlm_base_url, vlm_timeout_s=a.vlm_timeout,
                              start_online=not a.start_offline, node_name=a.node_name)
    print(f"phone alerts: {'configured (' + services.delivery.notifier.host + ')' if services.delivery.notifier else 'not configured'}",
          flush=True)
    seed = assets / "state" / "wx_seed.json"
    sensor = None if a.no_sensor or not seed.exists() else SensorEmulator(seed)
    map_kwargs = dict(assets_dir=assets, data_dir=a.data_dir, state_dir=state, figlib_dir=a.figlib_dir,
                      detector_weights=a.detector_weights, photo_detector_weights=a.photo_detector_weights,
                      detector_imgsz=imgsz_for(a.detector_weights),
                      photo_detector_imgsz=imgsz_for(a.photo_detector_weights),
                      vlm_model=a.vlm_model, vlm_base_url=a.vlm_base_url, vlm_timeout_s=a.vlm_timeout,
                      sensor=sensor, start_online=not a.start_offline, assistant_model=a.assistant_model)
    detectors = detector_specs(a.before_weights, a.detector_weights)
    pipelines = {"before": {**PIPELINES["before"], "vlm": a.before_vlm},
                 "after": {**PIPELINES["after"], "vlm": a.vlm_model}}
    web_kwargs = dict(prices_path=a.prices, results_dir=a.results, detectors=detectors, pipelines=pipelines,
                      vlm_base_url=a.vlm_base_url, vlm_timeout_s=a.vlm_timeout, warmup=True,
                      sample_dirs=["data/dfire/test/images", "data/demo", "data/benign"],
                      live_recheck_s=a.live_recheck_s)
    monitor = None
    if not a.no_monitor:
        from sentinel.live_metrics import system_stats
        from sentinel.monitor import Monitor, fetch_json, fetch_uds_metrics
        monitor = Monitor(a.run_dir, _load_prices(a.prices), None, fetch_uds_metrics, time.time, fetch_json,
                          system_stats)
    from sentinel.cloud_sampler import sampler_from_env
    cloud_sampler = sampler_from_env()
    print(f"cloud samples: {'on (' + cloud_sampler.host + ')' if cloud_sampler else 'off (no CLOUD_VLM_* settings)'}",
          flush=True)
    app = build_console(services=services, cameras=load_cameras(a.cameras),
                        mobile=mobile_camera(a.mobile_lat, a.mobile_lon), map_kwargs=map_kwargs,
                        web_kwargs=web_kwargs, cloud_sampler=cloud_sampler, results_dir=a.results,
                        prices_path=a.prices, monitor=monitor)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()

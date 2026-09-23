"""Wires real components and runs the replay loop + dashboard. `python -m sentinel.main`"""
import logging
import threading
import time

import uvicorn

from sentinel.app import Runtime, create_app
from sentinel.cloud import fetch_burn_schedule, fetch_forecast, send_dispatch
from sentinel.config import Settings, load_settings, load_towers
from sentinel.detector import YoloDetector
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.replayer import frames
from sentinel.vlm_client import ContextVLM, ensure_served

FLUSH_EVERY_S = 2.0


def run_loop(rt: Runtime, streams: dict, settings: Settings, stop: threading.Event) -> None:
    # flush runs on this thread; Metrics has a single writer.
    log = logging.getLogger(__name__)
    period = 1.0 / settings.fps
    last_flush = 0.0
    while not stop.is_set():
        tick = time.time()
        for tid, stream in list(streams.items()):
            try:  # isolate per tower: one bad source or frame must not skip the others
                frame = next(stream)
                rt.pipeline.process(tid, frame, tick)  # takes pipeline.lock itself; released during the VLM call
            except StopIteration:  # a non-looping source ran out: drop it instead of logging every tick
                log.error("tower %s stream ended", tid)
                streams.pop(tid)
            except Exception:
                log.exception("tower %s failed", tid)
        if tick - last_flush >= FLUSH_EVERY_S:
            try:
                if rt.link.online:
                    rt.pipeline.burn_towers = fetch_burn_schedule()
                rt.escalator.flush(tick)  # network I/O: never under the lock
            except Exception:
                log.exception("burn schedule / outbox flush failed")
            last_flush = tick
        time.sleep(max(0.0, period - (time.time() - tick)))


def warn_if_not_served(vlm: ContextVLM, model: str, base_url: str) -> bool:
    """Warn (don't exit) when the VLM model isn't served: the live demo still runs on the fallback."""
    try:
        ensure_served(vlm.client, model, base_url)
    except SystemExit as exc:
        logging.getLogger(__name__).warning("%s — the demo will run on the detector-only fallback", exc)
        return False
    return True


def main() -> None:
    settings = load_settings()
    towers = load_towers()
    metrics, link = Metrics(), Link(online=False)
    escalator = Escalator(Outbox(settings.db_path), link, fetch_forecast,
                          lambda payload: send_dispatch(settings.dispatch_url, payload), metrics)
    vlm = ContextVLM(settings.vlm_model, settings.vlm_base_url, settings.vlm_timeout_s)
    warn_if_not_served(vlm, settings.vlm_model, settings.vlm_base_url)
    pipeline = Pipeline(towers, YoloDetector(settings.detector_weights, classes=settings.detector_classes),
                        vlm, escalator, settings, metrics)
    rt = Runtime(pipeline, escalator, link)
    stop = threading.Event()
    streams = {tid: frames(t.source, settings.fps) for tid, t in towers.items()}
    for stream in streams.values():
        next(stream)  # generators run lazily: surface bad tower sources at startup, not per tick
    threading.Thread(target=run_loop, args=(rt, streams, settings, stop), daemon=True).start()
    uvicorn.run(create_app(rt), host="0.0.0.0", port=settings.dashboard_port)


if __name__ == "__main__":
    main()

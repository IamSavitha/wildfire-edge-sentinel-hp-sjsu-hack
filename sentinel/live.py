"""Live camera for the demo app: webcam frames through the real `Pipeline` (persistence gate, one VLM
call per event, trend re-check, per-tower latch), and real delivery of ALERTs (phone push via ntfy +
optional dispatch POST) behind the durable `Outbox`, retried by `Escalator.flush`.

No I/O of its own: detector, VLM, notifier, dispatch and forecast are injected (fakes in tests)."""
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime
from typing import Callable

import numpy as np

from sentinel.config import Settings, Tower
from sentinel.escalation import FORECAST_UPDATE, Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.schema import Severity

log = logging.getLogger(__name__)

LIVE_TOWER_ID = "live-camera"
MAX_TRACKED_EVENTS = 50
EVENTS_SHOWN = 4
MAX_ALERT_AGE_S = 600   # a queued ALERT older than this is dropped, not pushed (e.g. after a restart)


class _NoDispatch(Exception):
    """Raised instead of fetching a forecast nobody would receive."""


def _alert_age_s(payload: dict, now: float) -> float | None:
    try:
        return now - datetime.fromisoformat(payload["detected_at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


class Delivery:
    """Sends ALERT reports for real. Every ALERT goes through the outbox: `handle` enqueues it (this is
    the escalator interface `Pipeline` calls), `flush` delivers what is due while the link is up.
    A push or dispatch failure raises inside the escalator's send, so the report stays queued and is
    retried with the outbox backoff (<= 10 s); a retry only redoes the channel that failed, so the
    phone never buzzes twice for one alert. An ALERT still queued MAX_ALERT_AGE_S after detection is
    marked "expired" and not sent. The forecast update only goes to dispatch, so without a dispatch
    URL no forecast is fetched."""

    def __init__(self, outbox: Outbox, link: Link, forecast_fn: Callable[[float, float], dict],
                 notifier=None, dispatch_fn: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.time):
        self.outbox = outbox
        self.link = link
        self.notifier = notifier
        self.dispatch_fn = dispatch_fn
        self.clock = clock
        self.forecast_fn = forecast_fn
        self.escalator = Escalator(outbox, link, self._forecast, self._send, Metrics())
        self.events: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()          # guards `events`
        self.flush_lock = threading.Lock()    # one flush at a time (request threads + the retry thread)

    @property
    def configured(self) -> bool:
        return self.notifier is not None or self.dispatch_fn is not None

    # ------------------------------------------------------------ escalator interface

    def handle(self, report: dict, severity: Severity, now: float) -> None:
        if severity == Severity.ALERT:
            self._track(report)
        self.escalator.handle(report, severity, now)

    def deliver(self, report: dict, now: float) -> dict:
        """Queue one ALERT report and, if the link is up, send it now."""
        self.handle(report, Severity.ALERT, now)
        self.flush(now)
        return self.event(report["event_id"])

    def flush(self, now: float) -> int:
        with self.flush_lock:
            return self.escalator.flush(now)

    # ------------------------------------------------------------ state

    def event(self, event_id: str) -> dict | None:
        with self.lock:
            st = self.events.get(event_id)
            if st is None:
                return None
            st = dict(st)
        if st["delivered"]:
            state = "sent"
        elif not self.link.online:
            state = "queued"
        elif st["error"]:
            state = "retrying"
        else:
            state = "sending"
        return {**st, "state": state}

    def summary(self) -> dict:
        with self.lock:
            ids = list(self.events)
            delivered = sum(1 for s in self.events.values() if s["delivered"])
        phone = {"configured": self.notifier is not None}
        if self.notifier is not None:
            phone.update(self.notifier.stats())
        return {"online": self.link.online, "pending": self.outbox.pending_count(), "delivered": delivered,
                "phone": phone, "dispatch_configured": self.dispatch_fn is not None,
                "events": [self.event(e) for e in reversed(ids[-10:])]}

    def _track(self, report: dict) -> dict:
        eid = report["event_id"]
        with self.lock:
            if eid not in self.events:
                self.events[eid] = {"event_id": eid, "tower_name": report.get("tower_name"),
                                    "severity": report.get("severity"), "phone": "pending",
                                    "dispatch": "pending", "delivered": False, "attempts": 0,
                                    "error": None, "sent_at": None}
                while len(self.events) > MAX_TRACKED_EVENTS:
                    self.events.popitem(last=False)
            return self.events[eid]

    # ------------------------------------------------------------ send (runs inside Escalator.flush)

    def _forecast(self, lat: float, lon: float) -> dict:
        if self.dispatch_fn is None:
            raise _NoDispatch()      # the escalator then skips the forecast update
        return self.forecast_fn(lat, lon)

    def _send(self, payload: dict) -> None:
        if payload.get("type") == FORECAST_UPDATE:   # dispatch wants it; the phone already buzzed
            with self.lock:
                st = self.events.get(payload.get("event_id"))
            if self.dispatch_fn is not None and not (st and st["phone"] == "expired"):
                self.dispatch_fn(payload)
            return
        st = self._track(payload)                    # also covers alerts queued by an earlier run
        st["attempts"] += 1
        age = _alert_age_s(payload, self.clock())
        if age is not None and age > MAX_ALERT_AGE_S:
            log.warning("alert %s expired: detected %.0f s ago (> %d s); not sent", st["event_id"], age,
                        MAX_ALERT_AGE_S)
            st.update(phone="expired", dispatch="expired", delivered=True, error=None, sent_at=self.clock())
            return
        try:
            if st["phone"] in ("pending", "failed"):
                if self.notifier is None:
                    st["phone"] = "off"
                else:
                    try:
                        st["phone"] = self.notifier.notify_alert(payload)
                    except Exception:
                        st["phone"] = "failed"
                        raise
            if st["dispatch"] in ("pending", "failed"):
                if self.dispatch_fn is None:
                    st["dispatch"] = "off"
                else:
                    try:
                        self.dispatch_fn(payload)
                        st["dispatch"] = "sent"
                    except Exception:
                        st["dispatch"] = "failed"
                        raise
        except Exception as exc:
            st["error"] = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("alert %s not delivered (attempt %d): %s", st["event_id"], st["attempts"], st["error"])
            raise
        st.update(delivered=True, error=None, sent_at=self.clock())


def _det_dict(d, min_conf: float) -> dict:
    return {"cls": d.cls, "conf": round(float(d.conf), 4), "box": [round(float(v), 1) for v in d.box],
            "above_gate": d.conf >= min_conf}


class _VLMProbe:
    """The VLM as `Pipeline` sees it: resolved per call (None = not served -> detector-only fallback),
    serialised on the shared GPU lock, and timed for the live status strip."""

    def __init__(self, cam: "LiveCamera"):
        self.cam = cam

    def classify(self, jpeg: bytes):
        f = self.cam._frame
        vlm = self.cam.vlm_fn()
        if vlm is None:
            f["vlm_status"] = "unavailable"
            return None, 0
        t0 = time.perf_counter()
        f["vlm_status"] = "failed"
        try:
            with self.cam.gpu_lock:
                ctx, tokens = vlm.classify(jpeg)
        finally:
            f["vlm_ms"] = (time.perf_counter() - t0) * 1000
        f["vlm_status"] = "ok" if ctx is not None else "failed"
        f["tokens"] = int(tokens or 0)
        return ctx, tokens


class LiveCamera:
    """One persistent `Pipeline` for the camera stream, driven one frame at a time by the web app.
    `busy` is held by the caller while a frame is processed (frames arriving meanwhile are skipped)."""

    def __init__(self, tower: Tower, detector_fn: Callable[[], Callable], vlm_fn: Callable[[], object | None],
                 delivery: Delivery, settings: Settings, gpu_lock: threading.Lock):
        self.tower = tower
        self.detector_fn = detector_fn
        self.vlm_fn = vlm_fn
        self.delivery = delivery
        self.settings = settings
        self.gpu_lock = gpu_lock
        self.busy = threading.Lock()
        self._frame: dict = {}
        self.reset()

    def reset(self) -> None:
        self.pipeline = Pipeline({self.tower.id: self.tower}, self._detect, _VLMProbe(self), self.delivery,
                                 self.settings, Metrics())
        self.frames = 0
        self.stream: str | None = None

    def _detect(self, frame: np.ndarray):
        t0 = time.perf_counter()
        with self.gpu_lock:
            dets = list(self.detector_fn()(frame))
        self._frame["detect_ms"] = (time.perf_counter() - t0) * 1000
        self._frame["dets"] = dets
        return dets

    def process(self, image: np.ndarray, now: float, stream: str | None = None) -> dict:
        t0 = time.perf_counter()
        tid = self.tower.id
        if stream is not None and stream != self.stream:
            if self.stream is not None:          # another camera/session: its frames don't add up
                self.pipeline.gate.reset_streak(tid)
            self.stream = stream
        self._frame = {"dets": [], "detect_ms": None, "vlm_ms": None, "vlm_status": None, "tokens": 0}
        seen = len(self.pipeline.history)
        self.pipeline.process(tid, image, now)
        self.frames += 1
        new = self.pipeline.history[seen:]
        if any(e.severity == Severity.ALERT for e in new) and self.delivery.link.online:
            self.delivery.flush(now)             # network I/O: outside the pipeline lock
        out = self.view(now, new)
        h, w = image.shape[:2]
        f = self._frame
        out.update(
            frame={"width": int(w), "height": int(h)},
            detections=[_det_dict(d, self.settings.min_conf) for d in sorted(f["dets"], key=lambda d: -d.conf)],
            vlm={"status": f["vlm_status"], "tokens": f["tokens"]},
            timings={"detect_ms": f["detect_ms"], "vlm_ms": f["vlm_ms"],
                     "total_ms": (time.perf_counter() - t0) * 1000})
        return out

    def _event_view(self, ev) -> dict:
        d = ev.to_dict()
        ctx = d.pop("context") or {}
        d.update(source_type=ctx.get("source_type"), description=ctx.get("description"),
                 confidence=round(float(ev.confidence), 3),
                 delivery=self.delivery.event(ev.id) if ev.severity == Severity.ALERT else None)
        return d

    def view(self, now: float, new=()) -> dict:
        p, tid = self.pipeline, self.tower.id
        with p.lock:
            ev = p.active.get(tid)
            event = None
            if ev is not None:
                event = {"id": ev.id, "status": "classifying" if ev.severity is None else "monitoring",
                         "severity": ev.severity.name if ev.severity is not None else None,
                         "source_type": ev.ctx.source_type if ev.ctx else None,
                         "description": ev.ctx.description if ev.ctx else None, "trend": ev.trend,
                         "rechecks": ev.rechecks, "max_rechecks": self.settings.max_rechecks,
                         "next_check_in_s": max(0.0, ev.next_check_at - now),
                         "confidence": round(float(ev.confidence), 3)}
            latched_at = p.latched.get(tid)
            streak = p.gate.streak(tid)
            history = list(p.history[-EVENTS_SHOWN:])
        alert = next((e for e in new if e.severity == Severity.ALERT), None)
        return {
            "frames": self.frames, "stream": self.stream,
            "gate": {"streak": streak, "needed": self.settings.min_frames, "min_conf": self.settings.min_conf},
            "event": event,
            "latched": {"active": latched_at is not None,
                        "clears_in_s": (max(0.0, self.settings.cooldown_s - (now - latched_at))
                                        if latched_at is not None else None)},
            "new_events": [self._event_view(e) for e in new],
            "new_alert": self._event_view(alert) if alert is not None else None,
            "events": [self._event_view(e) for e in reversed(history)],
            "delivery": self.delivery.summary(),
        }

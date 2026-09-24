"""Delivery (phone + dispatch behind the outbox) and the live-camera pipeline wrapper, with fakes."""
import logging
import threading
from datetime import datetime, timezone

import numpy as np
import pytest

from sentinel.config import Settings, Tower
from sentinel.escalation import FORECAST_UPDATE, Link
from sentinel.live import LIVE_TOWER_ID, Delivery, LiveCamera
from sentinel.notify import NotifyError
from sentinel.outbox import Outbox
from sentinel.schema import ContextResult, Detection, Severity

WILD = ContextResult(source_type="wildland", smoke_color="black", attended="no", near_structures=True,
                     near_road=False, size_estimate="medium", description="Black smoke near a cabin.")
SMALL = ContextResult(source_type="wildland", smoke_color="grey", attended="no", near_structures=False,
                      near_road=False, size_estimate="small", description="A thin grey plume.")
SMOKE = [Detection("smoke", 0.9, (40, 40, 200, 160)), Detection("smoke", 0.2, (0, 0, 10, 10))]
TOWER = Tower(LIVE_TOWER_ID, "Live camera", 37.1, -121.9, "")


def report(eid="ev1", **kw):
    return {"event_id": eid, "tower_id": LIVE_TOWER_ID, "tower_name": "Live camera", "lat": 37.1,
            "lon": -121.9, "severity": "ALERT", "confidence": 0.9, "trend": None, "temp_c": 25.0,
            "source_type": "wildland", "description": "Smoke.", "forecast": None,
            "forecast_status": "not_requested", "thumbnail_jpeg_b64": "", **kw}


class FakeNotifier:
    def __init__(self, results=("sent",)):
        self.results = list(results)
        self.alerts = []
        self.tests = 0
        self.target = "ntfy.example/abcd…"

    def _next(self):
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(r, Exception):
            raise r
        return r

    def notify_alert(self, rep):
        self.alerts.append(rep["event_id"])
        return self._next()

    def send_test(self):
        self.tests += 1
        return self._next()

    def stats(self):
        return {"sent": len(self.alerts), "suppressed": 0, "failed": 0, "deferred": 0, "host": "ntfy.example",
                "min_interval_s": 30, "last_error": None}


def delivery(notifier=None, dispatch=None, online=True, forecasts=None, clock=lambda: 0.0):
    forecasts = [] if forecasts is None else forecasts

    def forecast(lat, lon):
        forecasts.append((lat, lon))
        return {"temp_c": [30.0]}
    return Delivery(Outbox(":memory:"), Link(online=online), forecast, notifier=notifier,
                    dispatch_fn=dispatch, clock=clock)


# ---------------------------------------------------------------- delivery

def test_offline_alert_is_queued_then_delivered_once_when_online():
    n = FakeNotifier()
    d = delivery(n, online=False)
    st = d.deliver(report(), now=100.0)
    assert st["state"] == "queued" and n.alerts == []
    assert d.summary()["pending"] == 1
    d.link.online = True
    assert d.flush(101.0) == 1
    assert n.alerts == ["ev1"]
    st = d.event("ev1")
    assert st["state"] == "sent" and st["phone"] == "sent" and st["dispatch"] == "off"
    assert d.flush(200.0) == 0 and n.alerts == ["ev1"]
    assert d.summary()["pending"] == 0 and d.summary()["delivered"] == 1


def test_push_failure_stays_queued_and_is_retried():
    n = FakeNotifier([NotifyError("ntfy down"), "sent"])
    d = delivery(n)
    st = d.deliver(report(), now=100.0)
    assert st["state"] == "retrying" and "ntfy down" in st["error"]
    assert d.summary()["pending"] == 1
    assert d.flush(100.5) == 0            # backoff not over yet
    assert d.flush(103.0) == 1
    assert d.event("ev1")["state"] == "sent" and n.alerts == ["ev1", "ev1"]


def test_retry_only_redoes_the_channel_that_failed():
    n = FakeNotifier()
    calls = []

    def dispatch(payload):
        calls.append(payload.get("type", "alert"))
        if len(calls) == 1:
            raise OSError("dispatch down")
    d = delivery(n, dispatch)
    assert d.deliver(report(), now=100.0)["state"] == "retrying"
    assert d.event("ev1")["phone"] == "sent" and d.event("ev1")["dispatch"] == "failed"
    d.flush(103.0)
    assert n.alerts == ["ev1"]                 # the phone did not buzz twice
    assert d.event("ev1")["dispatch"] == "sent" and d.event("ev1")["state"] == "sent"


def test_burst_deferred_push_stays_queued_and_goes_out_later():
    n = FakeNotifier([NotifyError("burst limit: more than 5 alert pushes in 60 s; deferred"), "sent"])
    d = delivery(n)
    st = d.deliver(report(), now=100.0)
    assert st["state"] == "retrying" and "burst limit" in st["error"] and d.summary()["pending"] == 1
    assert d.flush(103.0) == 1 and d.event("ev1")["phone"] == "sent"


def test_forecast_update_goes_to_dispatch_only():
    n, sent = FakeNotifier(), []
    forecasts = []
    d = delivery(n, sent.append, forecasts=forecasts)
    d.deliver(report(), now=100.0)
    assert [p.get("type") for p in sent] == [None, FORECAST_UPDATE]
    assert n.alerts == ["ev1"] and forecasts == [(37.1, -121.9)]


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def test_stale_queued_alert_expires_instead_of_buzzing(caplog):
    n, sent, forecasts = FakeNotifier(), [], []
    d = delivery(n, sent.append, online=False, forecasts=forecasts, clock=lambda: 10_000.0)
    d.deliver(report(detected_at=iso(10_000.0 - 700)), now=9_300.0)
    d.link.online = True
    with caplog.at_level(logging.WARNING, logger="sentinel.live"):
        d.flush(10_000.0)
    st = d.event("ev1")
    assert st["state"] == "sent" and st["phone"] == "expired" and st["dispatch"] == "expired"
    assert n.alerts == [] and sent == [] and d.summary()["pending"] == 0
    assert "expired" in caplog.text


def test_recent_queued_alert_is_still_delivered():
    n = FakeNotifier()
    d = delivery(n, online=False, clock=lambda: 10_000.0)
    d.deliver(report(detected_at=iso(10_000.0 - 120)), now=9_880.0)
    d.link.online = True
    d.flush(10_000.0)
    assert n.alerts == ["ev1"] and d.event("ev1")["phone"] == "sent"


def test_no_forecast_fetch_without_dispatch():
    forecasts = []
    d = delivery(FakeNotifier(), forecasts=forecasts)
    d.deliver(report(), now=100.0)
    assert forecasts == [] and d.summary()["pending"] == 0


def test_nothing_configured_is_a_simulated_delivery():
    d = delivery()
    st = d.deliver(report(), now=100.0)
    assert st["state"] == "sent" and st["phone"] == "off" and st["dispatch"] == "off"
    s = d.summary()
    assert s["phone"]["configured"] is False and s["dispatch_configured"] is False


def test_only_alerts_are_tracked():
    d = delivery(FakeNotifier())
    d.handle(report("m1", severity="MONITOR"), Severity.MONITOR, 1.0)
    assert d.event("m1") is None and d.summary()["pending"] == 0


# ---------------------------------------------------------------- live camera

class Det:
    def __init__(self):
        self.dets = list(SMOKE)
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.dets


class VLM:
    def __init__(self, ctx=WILD):
        self.ctx = ctx
        self.calls = 0

    def classify(self, jpeg):
        self.calls += 1
        return self.ctx, 300


def live(det=None, vlm=None, notifier=None, online=True, recheck_s=10.0, served=True):
    det, vlm = det or Det(), vlm or VLM()
    d = delivery(notifier or FakeNotifier(), online=online)
    cam = LiveCamera(TOWER, lambda: det, lambda: vlm if served else None, d,
                     Settings(min_conf=0.4, min_frames=3, recheck_s=recheck_s, cooldown_s=120.0),
                     gpu_lock=threading.Lock())
    return cam, det, vlm, d


FRAME = np.zeros((480, 640, 3), np.uint8)


def test_gate_needs_three_frames_before_the_vlm():
    cam, det, vlm, d = live()
    r1 = cam.process(FRAME, 0.0)
    assert r1["gate"] == {"streak": 1, "needed": 3, "min_conf": 0.4} and vlm.calls == 0
    assert r1["frame"] == {"width": 640, "height": 480}
    assert r1["detections"][0] == {"cls": "smoke", "conf": 0.9, "box": [40.0, 40.0, 200.0, 160.0],
                                   "above_gate": True}
    assert r1["detections"][1]["above_gate"] is False
    assert r1["new_alert"] is None and r1["timings"]["detect_ms"] is not None
    cam.process(FRAME, 1.0)
    assert vlm.calls == 0
    r3 = cam.process(FRAME, 2.0)
    assert vlm.calls == 1 and r3["vlm"]["status"] == "ok" and r3["vlm"]["tokens"] == 300
    a = r3["new_alert"]
    assert a["severity"] == "ALERT" and a["source_type"] == "wildland"
    assert a["delivery"]["state"] == "sent" and a["delivery"]["phone"] == "sent"
    assert "[ALERT]" in a["report_text"] and a["thumbnail_b64"]
    assert r3["events"][0]["id"] == a["id"] and r3["latched"]["active"] is True


def test_one_alert_per_fire_while_smoke_persists():
    n = FakeNotifier()
    cam, det, vlm, d = live(notifier=n)
    for t in range(12):
        r = cam.process(FRAME, float(t))
    assert vlm.calls == 1 and len(n.alerts) == 1 and r["new_alert"] is None
    assert r["latched"]["active"] is True and r["gate"]["streak"] == 0


def test_monitor_event_is_reported_as_active():
    cam, det, vlm, d = live(vlm=VLM(SMALL))
    for t in range(3):
        r = cam.process(FRAME, float(t))
    ev = r["event"]
    assert ev["status"] == "monitoring" and ev["severity"] == "MONITOR"
    assert ev["source_type"] == "wildland" and ev["description"] == "A thin grey plume."
    assert ev["next_check_in_s"] == pytest.approx(10.0)
    assert r["new_alert"] is None


def test_vlm_not_served_falls_back_without_calling_it():
    cam, det, vlm, d = live(served=False)
    for t in range(3):
        r = cam.process(FRAME, float(t))
    assert vlm.calls == 0 and r["vlm"]["status"] == "unavailable"
    assert r["event"]["severity"] == "MONITOR"          # fallback: never silently ignore


def test_offline_alert_queued_in_the_live_response():
    cam, det, vlm, d = live(online=False)
    for t in range(3):
        r = cam.process(FRAME, float(t))
    assert r["new_alert"]["delivery"]["state"] == "queued"
    assert r["delivery"]["pending"] == 1


def test_reset_clears_latch_and_history():
    cam, det, vlm, d = live()
    for t in range(3):
        cam.process(FRAME, float(t))
    cam.reset()
    r = cam.process(FRAME, 10.0)
    assert r["events"] == [] and r["latched"]["active"] is False and r["gate"]["streak"] == 1
    cam.process(FRAME, 11.0)
    cam.process(FRAME, 12.0)
    assert vlm.calls == 2


def test_a_new_stream_restarts_the_persistence_count():
    cam, det, vlm, d = live()
    cam.process(FRAME, 0.0, stream="a")
    cam.process(FRAME, 1.0, stream="a")
    r = cam.process(FRAME, 2.0, stream="b")
    assert r["gate"]["streak"] == 1 and vlm.calls == 0


def test_no_smoke_resets_the_streak():
    cam, det, vlm, d = live()
    cam.process(FRAME, 0.0)
    det.dets = []
    r = cam.process(FRAME, 1.0)
    assert r["gate"]["streak"] == 0 and r["detections"] == []

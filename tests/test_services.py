"""Services: the console's shared detector/VLM caches, served-model list, uplink and delivery, with fakes."""
import os
import threading
import time

import pytest

from sentinel.escalation import Link
from sentinel.live import Delivery
from sentinel.outbox import Outbox
from sentinel.schema import Severity
from sentinel.services import MODELS_TTL_S, Services, build_services


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def notify_alert(self, rep):
        self.alerts.append(rep["event_id"])
        return "sent"

    def stats(self):
        return {"sent": len(self.alerts)}


def no_forecast(lat, lon):
    raise AssertionError("no dispatch configured: no forecast should be fetched")


def services(tmp_path=None, *, detector_factory=None, vlm_factory=None, model_lister=None, notifier=None,
             online=True, clock=None):
    clock = clock or Clock()
    link = Link(online=online)
    return Services(link=link,
                    delivery=Delivery(Outbox(":memory:"), link, no_forecast, notifier=notifier, clock=clock),
                    detector_factory=detector_factory or (lambda w, sz: object()),
                    vlm_factory=vlm_factory or (lambda m: object()),
                    model_lister=model_lister or (lambda: []), clock=clock)


def alert(eid="ev1"):
    return {"event_id": eid, "tower_id": "t1", "tower_name": "Tower 1", "lat": 37.1, "lon": -121.9,
            "severity": "ALERT", "forecast": None}


# ---------------------------------------------------------------- detectors

def test_one_detector_load_per_weights_file_with_8_concurrent_callers(tmp_path):
    weights = tmp_path / "joint_yolo.pt"
    weights.write_bytes(b"w")
    loads = []

    def factory(w, imgsz):
        loads.append((w, imgsz))
        time.sleep(0.05)              # a slow load: every caller arrives while it runs
        return object()

    svc = services(detector_factory=factory)
    got, start = [], threading.Barrier(8)

    def call():
        start.wait()
        got.append(svc.detector(str(weights)))
    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loads == [(str(weights), 960)]           # tower/joint weights default to their 960 px
    assert len(got) == 8 and all(d is got[0] for d in got)


def test_detector_size_defaults_per_weights_and_an_explicit_size_is_its_own_model(tmp_path):
    loads = []
    svc = services(detector_factory=lambda w, sz: loads.append((w, sz)) or object())
    small = str(tmp_path / "smoke_yolo.pt")
    a = svc.detector(small)
    assert svc.detector(small, 640) is a            # same weights, same resolved size: one load
    svc.detector(small, 1280)
    assert loads == [(small, 640), (small, 1280)]


def test_weights_rewritten_on_disk_are_reloaded(tmp_path):
    weights = tmp_path / "smoke_yolo.pt"
    weights.write_bytes(b"v1")
    svc = services(detector_factory=lambda w, sz: object())
    first = svc.detector(str(weights))
    st = weights.stat()
    os.utime(weights, (st.st_atime, st.st_mtime + 10))
    second = svc.detector(str(weights))
    assert second is not first
    assert svc.detector(str(weights)) is second


def test_gpu_lock_is_one_lock_and_not_held_by_a_load(tmp_path):
    svc = services()
    assert isinstance(svc.gpu_lock, type(threading.Lock()))
    with svc.gpu_lock:                             # callers hold gpu_lock and may load under it
        svc.detector(str(tmp_path / "x.pt"))


# ---------------------------------------------------------------- VLM

def test_vlm_client_is_cached_per_model():
    made = []
    svc = services(vlm_factory=lambda m: made.append(m) or object())
    a = svc.vlm("context")
    assert svc.vlm("context") is a
    assert svc.vlm("context_v2") is not a
    assert made == ["context", "context_v2"]


def test_served_models_ttl_holds():
    clock, calls = Clock(), []
    svc = services(clock=clock, model_lister=lambda: calls.append(1) or ["context", "base7b"])
    assert svc.served_models() == (["context", "base7b"], None)
    clock.t += MODELS_TTL_S - 0.1
    svc.served_models()
    assert len(calls) == 1
    clock.t += 0.2
    svc.served_models()
    assert len(calls) == 2


def test_served_models_error_means_nothing_served_and_is_cached_too():
    clock, calls = Clock(), []

    def down():
        calls.append(1)
        raise ConnectionError("refused")
    svc = services(clock=clock, model_lister=down)
    assert svc.served_models() == ([], "ConnectionError: refused")
    assert svc.served_models() == ([], "ConnectionError: refused")
    assert len(calls) == 1


# ---------------------------------------------------------------- uplink

def test_outage_then_restore_flushes_a_queued_alert_exactly_once_and_calls_callbacks():
    n = FakeNotifier()
    svc = services(notifier=n, online=True)
    seen = []
    svc.on_link_change(seen.append)
    assert svc.set_online(False) == 0
    svc.delivery.handle(alert("ev1"), Severity.ALERT, svc.clock())
    assert svc.delivery.flush(svc.clock()) == 0 and n.alerts == []     # down: stays queued
    assert svc.set_online(True) == 1
    assert n.alerts == ["ev1"] and svc.delivery.outbox.pending_count() == 0
    assert svc.set_online(True) == 0 and n.alerts == ["ev1"]           # nothing twice
    assert seen == [False, True]                                       # one call per change
    assert svc.delivery.event("ev1")["state"] == "sent"


def test_a_failing_callback_does_not_break_the_toggle():
    svc = services(online=False)
    seen = []

    def boom(online):
        raise RuntimeError("listener bug")
    svc.on_link_change(boom)
    svc.on_link_change(seen.append)
    assert svc.set_online(True) == 0
    assert svc.link.online is True and seen == [True]


def test_flush_tells_listeners_only_when_something_went_out():
    n = FakeNotifier()
    svc = services(notifier=n, online=True)
    seen = []
    svc.on_link_change(seen.append)
    assert svc.flush() == 0 and seen == []
    svc.delivery.handle(alert("ev2"), Severity.ALERT, svc.clock())
    assert svc.flush() == 1 and seen == [True] and n.alerts == ["ev2"]


def test_started_at_comes_from_the_clock():
    assert services(clock=Clock(42.0)).started_at == 42.0


# ---------------------------------------------------------------- build_services

def test_build_services_wires_one_link_one_outbox_and_the_notifier(tmp_path):
    n = FakeNotifier()
    svc = build_services(outbox_path=str(tmp_path / "state" / "outbox.db"), notifier=n,
                         forecast_fn=no_forecast, detector_factory=lambda w, sz: object(),
                         vlm_factory=lambda m: object(), model_lister=lambda: ["context"], clock=Clock(7.0))
    assert svc.delivery.link is svc.link and svc.link.online is True
    assert svc.delivery.notifier is n and svc.delivery.dispatch_fn is None
    assert (tmp_path / "state" / "outbox.db").exists()                  # durable outbox on disk
    assert svc.served_models() == (["context"], None) and svc.started_at == 7.0


def test_build_services_reads_the_notifier_from_env(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC_URL", raising=False)
    assert build_services(forecast_fn=no_forecast).delivery.notifier is None
    monkeypatch.setenv("NTFY_TOPIC_URL", "https://ntfy.example/secret-topic")
    assert build_services(forecast_fn=no_forecast).delivery.notifier.host == "ntfy.example"
    assert build_services(forecast_fn=no_forecast, notifier=None).delivery.notifier is None


def test_build_services_default_vlm_is_a_context_vlm_on_the_given_base_url():
    svc = build_services(forecast_fn=no_forecast, notifier=None, vlm_base_url="unix:///tmp/vllm.sock")
    from sentinel.vlm_client import ContextVLM
    v = svc.vlm("context")
    assert isinstance(v, ContextVLM) and v.model == "context"


def test_build_services_can_start_offline():
    assert build_services(forecast_fn=no_forecast, notifier=None, start_online=False).link.online is False


def test_services_module_does_not_import_ultralytics():
    import subprocess
    import sys
    code = "import sys, sentinel.services; print('ultralytics' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("weights,size", [("models/joint_yolo.pt", 960), ("models/smoke_yolo.pt", 640)])
def test_default_detector_factory_uses_the_trained_size(monkeypatch, weights, size):
    import sentinel.detector
    made = {}

    class FakeYolo:
        def __init__(self, w, conf, imgsz):
            made.update(w=w, conf=conf, imgsz=imgsz)
    monkeypatch.setattr(sentinel.detector, "YoloDetector", FakeYolo)
    from sentinel.services import default_detector_factory
    default_detector_factory(weights)
    assert made == {"w": weights, "conf": 0.1, "imgsz": size}

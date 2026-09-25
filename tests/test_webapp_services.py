"""The web app (compare on an image, mobile camera) on the console's shared services."""
import json

from fastapi.testclient import TestClient

from sentinel.services import build_services
from sentinel.webapp import create_web_app
from tests.test_webapp import TOWER, FakeVLM, jpeg
from tests.test_webapp_live import SMOKE, Clock, Det, FakeNotifier, frames


def shared(online=True, clock=None):
    n, det, vlms = FakeNotifier(), Det(list(SMOKE)), {}

    def vlm_factory(model):
        vlms[model] = v = FakeVLM(model)
        return v
    svc = build_services(notifier=n, forecast_fn=lambda la, lo: {"temp_c": [30.0]},
                         detector_factory=lambda w, sz: det, vlm_factory=vlm_factory,
                         model_lister=lambda: ["base7b", "context", "context_v2"], start_online=online,
                         clock=clock or Clock())
    return svc, n, det, vlms


def app_on(tmp_path, svc, clock, events=None):
    prices = tmp_path / "cost_inputs.json"
    prices.write_text(json.dumps({"usd_per_mtok_in": 0, "usd_per_mtok_out": 0, "usd_per_gb": 0}))
    return create_web_app(detector_factory=lambda spec: Det(list(SMOKE)), monitor=None, prices_path=prices,
                          sample_dirs=[], results_dir=tmp_path, tower=TOWER,
                          detector_available=lambda spec: True, clock=clock, services=svc,
                          on_live_event=(lambda ev, img: events.append((ev, img.shape))) if events is not None
                          else None)


def test_live_alert_uses_the_shared_outbox_and_calls_the_event_hook_once(tmp_path):
    clock = Clock()
    svc, n, det, vlms = shared(online=True, clock=clock)
    events = []
    with TestClient(app_on(tmp_path, svc, clock, events)) as c:
        r = frames(c, clock, 3)[-1]
        assert r["new_alert"]["severity"] == "ALERT" and r["new_alert"]["delivery"]["state"] == "sent"
        frames(c, clock, 5)                                        # latched: no second event
    assert len(n.alerts) == 1 and det.calls == 8                   # the shared detector ran every frame
    alerts = [e for e, _ in events if e["severity"] == "ALERT"]
    assert len(alerts) == 1 and events[0][1] == (480, 640, 3)
    assert svc.delivery.event(alerts[0]["id"])["state"] == "sent"
    assert "context" in vlms


def test_web_network_flips_the_shared_link_and_flushes_the_shared_outbox(tmp_path):
    clock = Clock()
    svc, n, _, _ = shared(online=True, clock=clock)
    with TestClient(app_on(tmp_path, svc, clock)) as c:
        assert c.post("/api/network", json={"online": False}).json() == {"online": False}
        assert svc.link.online is False
        frames(c, clock, 3)
        assert n.alerts == [] and svc.delivery.outbox.pending_count() == 1
        assert c.post("/api/network", json={"online": True}).json() == {"online": True, "flushed": 1}
    assert len(n.alerts) == 1 and svc.link.online is True


def test_live_frames_are_mobile_frames_in_the_shadow(tmp_path):
    clock = Clock()
    svc, _, _, _ = shared(online=False, clock=clock)
    with TestClient(app_on(tmp_path, svc, clock)) as c:
        frames(c, clock, 4)
    s = svc.shadow.summary()
    assert s["by_source"]["mobile"]["frames"] == 4 and s["cloud_blind_frames"] == 4
    assert s["edge"]["vlm_calls"] == 1
    svc.set_online(True)
    assert svc.shadow.summary()["by_source"]["mobile"]["edge_bytes_up"] > 0   # counted when really sent


def test_analyze_counts_the_after_pass_only(tmp_path):
    clock = Clock()
    svc, _, _, _ = shared(online=True, clock=clock)
    with TestClient(app_on(tmp_path, svc, clock)) as c:
        r = c.post("/api/analyze", data={"pipelines": "before,after"},
                   files={"image": ("f.jpg", jpeg(1920, 1080), "image/jpeg")})
        assert r.status_code == 200, r.text
    s = svc.shadow.summary()
    assert s["by_source"]["field"]["frames"] == 1 and s["cloud"]["frames"] == 1


def test_model_switch_applies_to_the_live_camera(tmp_path):
    clock = Clock()
    svc, _, _, vlms = shared(online=True, clock=clock)
    svc.active_vlm = "context_v2"
    with TestClient(app_on(tmp_path, svc, clock)) as c:
        frames(c, clock, 3)
    assert "context_v2" in vlms and vlms["context_v2"].calls == 1 and "context" not in vlms

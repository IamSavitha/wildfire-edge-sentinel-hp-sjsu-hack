"""The incident map on the console's shared services: real delivery through the one outbox, the one
uplink, the cloud-only shadow, and sightings from other sources."""
import numpy as np
from fastapi.testclient import TestClient

from sentinel.incident_map import create_map_app
from sentinel.services import build_services
from tests.test_incident_map import CAMS, COUNTIES, FORECAST, Clock, FakeDetector, FakeVLM, WILD, jpeg, post
from tests.test_incident_map import wait_done, watch_env


class FakeNotifier:
    def __init__(self):
        self.alerts = []

    def notify_alert(self, report):
        self.alerts.append(report["event_id"])
        return "sent"

    def stats(self):
        return {"sent": len(self.alerts)}


def shared(online=False, clock=None):
    n = FakeNotifier()
    det = FakeDetector()
    svc = build_services(notifier=n, forecast_fn=lambda la, lo: FORECAST, detector_factory=lambda w, sz: det,
                         vlm_factory=FakeVLM, model_lister=lambda: ["context", "context_v2"],
                         start_online=online, clock=clock or Clock())
    return svc, n, det


def app_on(tmp_path, svc, cams=CAMS, clock=None):
    FakeVLM.ctx = WILD
    return create_map_app(cameras=dict(cams), assets_dir=tmp_path / "assets", data_dir=tmp_path / "data",
                          detector_weights="w.pt", forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES,
                          clock=clock or Clock(), services=svc)


def test_tower_alert_offline_is_one_outbox_entry_and_is_sent_once_when_the_link_returns(tmp_path):
    svc, n, _ = shared(online=False)
    with TestClient(app_on(tmp_path, svc)) as c:
        inc = post(c, camera_id="rm-e").json()["incident"]
        assert inc["severity"] == "ALERT" and inc["escalation"]["decision"] == "queued"
        assert svc.delivery.outbox.pending_count() == 1 and n.alerts == []
        assert c.get("/api/state").json()["online"] is False
        assert svc.set_online(True) == 1
        assert n.alerts == [inc["escalation"]["event_id"]]
        got = c.get(f"/api/incidents/{inc['id']}").json()
        assert got["escalation"]["decision"] == "sent" and "queued_at" in got["escalation"]
        s = c.get("/api/state").json()["summary"]
        assert s["alerts_sent"] == 1 and s["alerts_queued"] == 0 and s["outbox"] == 0
        svc.set_online(False)
        svc.set_online(True)
        assert len(n.alerts) == 1                                  # never twice


def test_map_network_toggle_is_the_shared_uplink(tmp_path):
    svc, n, _ = shared(online=False)
    with TestClient(app_on(tmp_path, svc)) as c:
        post(c, camera_id="rm-e")
        r = c.post("/api/network", json={"online": True}).json()
        assert r == {"online": True, "flushed": 1} and svc.link.online is True and len(n.alerts) == 1
        assert c.get("/api/config").json()["online"] is True


def test_online_alert_is_pushed_immediately(tmp_path):
    svc, n, _ = shared(online=True)
    with TestClient(app_on(tmp_path, svc)) as c:
        inc = post(c, camera_id="rm-e").json()["incident"]
        assert inc["escalation"]["decision"] == "sent" and inc["escalation"]["bytes_up"] > 0
        assert len(n.alerts) == 1 and svc.delivery.outbox.pending_count() == 0


def test_report_upload_is_one_field_frame_in_the_shadow(tmp_path):
    svc, _, _ = shared(online=True)
    with TestClient(app_on(tmp_path, svc)) as c:
        post(c, camera_id="rm-e", image=jpeg(1600, 900))
    s = svc.shadow.summary()
    assert s["by_source"]["field"]["frames"] == 1 and s["edge"]["vlm_calls"] == 1
    assert s["cloud"]["frames"] == 1 and s["edge"]["bytes_up"] > 0     # the ALERT report went out


def test_offline_report_is_cloud_blind(tmp_path):
    svc, _, _ = shared(online=False)
    with TestClient(app_on(tmp_path, svc)) as c:
        post(c, camera_id="rm-e")
    s = svc.shadow.summary()
    assert s["cloud_blind_frames"] == 1 and s["edge_offline_decisions"] == 1 and s["edge"]["bytes_up"] == 0
    svc.set_online(True)
    assert svc.shadow.summary()["edge"]["bytes_up"] > 0                  # delivered later, counted then


def test_detector_and_vlm_come_from_the_shared_caches_and_the_model_switch(tmp_path):
    svc, _, det = shared(online=True)
    app = app_on(tmp_path, svc)
    with TestClient(app) as c:
        post(c, camera_id="rm-e")
        assert svc.vlm("context") is not None
        svc.active_vlm = "context_v2"
        assert c.get("/api/config").json()["vlm_model"] == "context_v2"
        assert app.state.model_name() == "context_v2"


def test_ingest_external_opens_an_incident_without_another_outbox_entry(tmp_path):
    svc, n, _ = shared(online=False)
    cams = {**CAMS, "mobile-1": {"id": "mobile-1", "name": "Mobile unit 1", "site": "Mobile unit",
                                 "lat": 33.3, "lon": -117.1, "azimuth_deg": 0, "hfov_deg": 60, "wx_station": None}}
    app = app_on(tmp_path, svc, cams)
    with TestClient(app) as c:
        report = {"event_id": "live1", "tower_id": "mobile-1", "tower_name": "Mobile unit 1", "lat": 33.3,
                  "lon": -117.1, "detected_at": "2026-09-24T00:00:00+00:00", "severity": "ALERT",
                  "confidence": 0.8, "trend": "growing", "temp_c": 25.0, "source_type": "wildland",
                  "context": None, "description": "Smoke.", "forecast": None, "thumbnail_jpeg_b64": ""}
        view, created = app.state.ingest_external(
            "mobile-1", severity="ALERT", context={"source_type": "wildland", "description": "Smoke."},
            report=report, conf=0.8, image=np.zeros((360, 640, 3), np.uint8),
            detections=[{"box": [10, 10, 100, 100], "conf": 0.8}],
            escalation={"decision": "queued", "event_id": "live1", "payload_bytes": 900, "at": 1.0})
        assert created and view["camera_id"] == "mobile-1" and view["loc_source"] == "camera_site"
        assert svc.delivery.outbox.pending_count() == 0 and n.alerts == []
        got = c.get(f"/api/incidents/{view['id']}").json()
        assert got["severity"] == "ALERT" and len(got["frames"]) == 1 and got["updates"][0]["kind"] == "opened"


def test_tower_watch_frames_are_recorded_in_the_shadow(tmp_path):
    import tests.test_incident_map as tim
    svc, _, _ = shared(online=True)
    base = watch_env(tmp_path)                  # builds the recording on disk
    del base
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "assets", data_dir=tmp_path / "data",
                         detector_weights="w.pt", detector_factory=lambda w: tim.ScriptedDetector(),
                         forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES, clock=Clock(), services=svc)
    with TestClient(app) as c:
        c.post("/api/watch", json={"recording": "fire_x", "interval_s": 0, "batch_frames": 4,
                                   "time_mode": "live", "start_offset_s": None})
        [s] = wait_done(c)
    sh = svc.shadow.summary()
    assert sh["by_source"]["tower"]["frames"] == s["frames_done"] == 14
    assert sh["edge"]["vlm_calls"] == s["vlm_calls"] and sh["cloud"]["vlm_calls"] == 14


def test_without_services_nothing_changes(tmp_path):
    FakeVLM.ctx = WILD
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "assets", detector_weights="w.pt",
                         detector_factory=lambda w: FakeDetector(), vlm_factory=FakeVLM,
                         model_lister=lambda: ["context"], forecast_fn=lambda la, lo: FORECAST,
                         counties=COUNTIES, clock=Clock())
    with TestClient(app) as c:
        inc = post(c, camera_id="rm-e").json()["incident"]
        assert inc["escalation"]["decision"] == "queued"
        assert c.post("/api/network", json={"online": True}).json()["flushed"] == 1

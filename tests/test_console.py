"""Sentinel Console: one server over the incident map and the web app, on shared services (fakes only)."""
import json

import pytest
from fastapi.testclient import TestClient

from sentinel.console import MOBILE_ID, build_console, mobile_camera
from sentinel.services import build_services
from tests.test_incident_map import CAMS, COUNTIES, FORECAST, FakeVLM, WILD, jpeg
from tests.test_webapp import TOWER
from tests.test_webapp_live import SMOKE, Clock, Det

TOPIC = "https://ntfy.example.org/very-secret-topic-name"


class FakeNotifier:
    host = "ntfy.example.org"

    def __init__(self):
        self.alerts, self.tests = [], 0

    def notify_alert(self, report):
        self.alerts.append(report)
        return "sent"

    def send_test(self):
        self.tests += 1
        return "sent"

    def stats(self):
        return {"sent": len(self.alerts), "suppressed": 0, "failed": 0, "deferred": 0, "host": self.host,
                "target": TOPIC}


class FakeSampler:
    def __init__(self):
        self.sampled = []

    def maybe_sample(self, key, frame, ctx):
        self.sampled.append((key, ctx))
        return True

    def summary(self):
        return {"enabled": True, "n": len(self.sampled)}


def console(tmp_path, online=True, sampler=None):
    FakeVLM.ctx = WILD
    clock, n, det = Clock(), FakeNotifier(), Det(list(SMOKE))
    svc = build_services(notifier=n, forecast_fn=lambda la, lo: FORECAST, detector_factory=lambda w, sz: det,
                         vlm_factory=FakeVLM, model_lister=lambda: ["base7b", "context", "context_v2"],
                         start_online=online, clock=clock, node_name="Test node")
    weights = tmp_path / "joint_yolo.pt"
    weights.write_bytes(b"w")
    results = tmp_path / "results"
    results.mkdir()
    (results / "detector_tower_stage3.json").write_text(json.dumps({"map50": 0.718, "precision": 0.69}))
    (results / "context_after_lora7b.json").write_text(json.dumps({"source_type_acc": 0.736, "group_acc": 0.756}))
    (results / "context_after_lora7b_v2.json").write_text(json.dumps({"source_type_acc": 0.744}))
    (results / "bench_after.json").write_text(json.dumps({"precision": 0.65, "recall": 0.73, "n_clips": 25}))
    prices = tmp_path / "cost_inputs.json"
    prices.write_text(json.dumps({"usd_per_mtok_in": 0.2, "usd_per_mtok_out": 0.6, "usd_per_gb": 5}))
    map_kwargs = dict(assets_dir=tmp_path / "assets", data_dir=tmp_path / "data", state_dir=tmp_path / "state",
                      detector_weights=str(weights), photo_detector_weights=None,
                      forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES, clock=clock)
    web_kwargs = dict(detector_factory=lambda spec: Det(list(SMOKE)), monitor=None, prices_path=prices,
                      sample_dirs=[], results_dir=results, detector_available=lambda spec: True, clock=clock)
    app = build_console(services=svc, cameras=CAMS, mobile=mobile_camera(33.3, -117.1), map_kwargs=map_kwargs,
                        web_kwargs=web_kwargs, cloud_sampler=sampler, results_dir=results, prices_path=prices,
                        system_stats=lambda: {"gpu_util_pct": 12.0, "mem_used_gb": 40.0, "mem_total_gb": 128.0})
    return app, svc, n, clock


def live_frames(c, clock, k):
    out = []
    for _ in range(k):
        r = c.post("/lab/api/live/frame", content=jpeg(640, 480), headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 200, r.text
        out.append(r.json())
        clock.t += 1.0
    return out


def test_one_server_answers_every_part_and_runs_the_sub_app_lifespans(tmp_path):
    app, _, _, _ = console(tmp_path)
    with TestClient(app) as c:
        assert app.state.started
        assert c.get("/ops/api/state").status_code == 200
        assert c.get("/lab/api/config").status_code == 200
        o = c.get("/api/overview").json()
        assert o["node"] == "Test node" and o["uplink"] == {"online": True, "queued": 0}
        assert o["incidents"]["active"] == 0 and o["last_alert"] is None
        assert {m["id"] for m in c.get("/ops/api/config").json()["cameras"]} >= {"rm-e", MOBILE_ID}


def test_mobile_camera_alert_is_one_incident_one_outbox_entry_one_push(tmp_path):
    sampler = FakeSampler()
    app, svc, n, clock = console(tmp_path, sampler=sampler)
    with TestClient(app) as c:
        live_frames(c, clock, 3)
        live_frames(c, clock, 5)                       # same fire: latched
        incs = c.get("/ops/api/state").json()["incidents"]
        assert len(incs) == 1 and incs[0]["camera_id"] == MOBILE_ID and incs[0]["severity"] == "ALERT"
        assert incs[0]["escalation"]["decision"] == "sent"
        assert len(n.alerts) == 1 and svc.delivery.outbox.pending_count() == 0
        o = c.get("/api/overview").json()
        assert o["incidents"]["alert"] == 1 and o["last_alert"]["source"] == "mobile"
        assert o["last_alert"]["incident"]["id"] == incs[0]["id"]
        assert o["savings"]["frames"] == 8
        assert [k for k, _ in sampler.sampled] == [incs[0]["id"]]      # one cloud sample per incident


def test_outage_queues_alerts_from_every_source_and_restoring_sends_each_once(tmp_path):
    app, svc, n, clock = console(tmp_path, online=True)
    with TestClient(app) as c:
        assert c.post("/api/uplink", json={"online": False}).json()["online"] is False
        live_frames(c, clock, 3)                                          # mobile camera ALERT
        r = c.post("/ops/api/report", data={"camera_id": "rm-e"},
                   files={"image": ("f.jpg", jpeg(), "image/jpeg")})       # tower ALERT
        assert r.json()["incident"]["escalation"]["decision"] == "queued"
        assert n.alerts == [] and c.get("/api/uplink").json() == {"online": False, "queued": 2}
        o = c.get("/api/overview").json()
        assert o["uplink"]["queued"] == 2 and o["savings"]["offline_decisions"] == 4
        r = c.post("/api/uplink", json={"online": True}).json()
        assert r["flushed"] == 2 and r["queued"] == 0 and len(n.alerts) == 2
        c.post("/api/uplink", json={"online": False})
        c.post("/api/uplink", json={"online": True})
        assert len(n.alerts) == 2
        states = {i["camera_id"]: i["escalation"]["decision"] for i in c.get("/ops/api/state").json()["incidents"]}
        assert states == {MOBILE_ID: "sent", "rm-e": "sent"}
        alerts = c.get("/api/alerts").json()
        assert len(alerts["alerts"]) == 2 and {a["state"] for a in alerts["alerts"]} == {"sent"}
        assert {a["source"] for a in alerts["alerts"]} == {"mobile", "tower"}


def test_alerts_and_system_never_show_the_topic(tmp_path):
    app, _, n, clock = console(tmp_path)
    with TestClient(app) as c:
        live_frames(c, clock, 3)
        for path in ("/api/alerts", "/api/system", "/api/overview"):
            body = c.get(path).text
            assert "very-secret-topic-name" not in body
        assert c.get("/api/alerts").json()["phone"]["host"] == "ntfy.example.org"
        assert c.post("/api/alerts/test").json() == {"status": "sent"} and n.tests == 1
        c.post("/api/uplink", json={"online": False})
        assert c.post("/api/alerts/test").status_code == 409


def test_model_switch_applies_to_map_and_live_camera(tmp_path):
    app, svc, _, _ = console(tmp_path)
    with TestClient(app) as c:
        m = c.get("/api/models").json()
        assert m["vlm"]["active"] == "context" and m["vlm"]["options"] == ["context", "context_v2"]
        assert m["vlm"]["headline"]["value"] == 0.736 and m["detector"]["headline"]["value"] == 0.718
        assert c.post("/api/models/vlm", json={"model": "teacher32b"}).status_code == 400
        v = c.post("/api/models/vlm", json={"model": "context_v2"}).json()
        assert v["active"] == "context_v2" and v["headline"]["value"] == 0.744
        assert c.get("/ops/api/config").json()["vlm_model"] == "context_v2"
        assert svc.active_vlm == "context_v2"


def test_edge_cloud_page_data(tmp_path):
    app, _, _, clock = console(tmp_path)
    with TestClient(app) as c:
        live_frames(c, clock, 4)
        d = c.get("/api/edge-cloud?profile=satellite_geo").json()
        s = d["summary"]
        assert s["profile"] == "satellite_geo" and s["prices_set"] is True and s["edge"]["frames"] == 4
        assert s["cloud"]["bytes_up"] > s["edge"]["bytes_up"] and s["savings"]["usd"] > 0
        assert d["benchmarks"]["edge"]["precision"] == 0.65 and d["measured"] == {"enabled": False}
        assert {p["name"] for p in d["profiles"]} >= {"fiber", "outage"}
        assert c.get("/api/edge-cloud?profile=pigeon").status_code == 400
        f = c.get("/api/edge-cloud/fleet?towers=50&fps=1&days=30").json()
        assert len(f["rows"]) == 3 and f["towers"] == 50
        assert c.get("/api/edge-cloud/fleet?towers=0").status_code == 400
        assert c.post("/api/edge-cloud/prices", json={"usd_per_gb": 9}).json()["usd_per_gb"] == 9
        c.post("/api/edge-cloud/reset")
        assert c.get("/api/edge-cloud").json()["summary"]["edge"]["frames"] == 0


def test_system_health(tmp_path):
    app, _, _, _ = console(tmp_path, online=False)
    with TestClient(app) as c:
        s = c.get("/api/system").json()
        h = {r["id"]: r["state"] for r in s["health"]}
        assert h == {"detector": "ok", "vlm": "ok", "uplink": "warn", "outbox": "ok", "phone": "ok",
                     "dispatch": "off"}
        assert s["stats"]["gpu_util_pct"] == 12.0 and s["served_models"] == ["base7b", "context", "context_v2"]


@pytest.mark.parametrize("path", ["/api/uplink", "/api/models/vlm"])
def test_bad_bodies_are_422(tmp_path, path):
    app, _, _, _ = console(tmp_path)
    with TestClient(app) as c:
        assert c.post(path, json={"nope": 1}).status_code == 422


def test_mobile_monitor_plume_is_on_the_map_while_it_is_rechecked(tmp_path):
    from sentinel.schema import ContextResult
    app, svc, n, clock = console(tmp_path)
    FakeVLM.ctx = ContextResult(source_type="wildland", smoke_color="grey", attended="no", near_structures=False,
                                near_road=False, size_estimate="small", description="Grey smoke on a slope.")
    try:
        with TestClient(app) as c:
            r = live_frames(c, clock, 3)[-1]
            assert r["event"]["severity"] == "MONITOR" and r["new_alert"] is None
            incs = c.get("/ops/api/state").json()["incidents"]
            assert len(incs) == 1 and incs[0]["camera_id"] == MOBILE_ID and incs[0]["severity"] == "MONITOR"
            live_frames(c, clock, 3)                     # still MONITOR: no second incident, no push
            assert len(c.get("/ops/api/state").json()["incidents"]) == 1 and n.alerts == []
    finally:
        FakeVLM.ctx = WILD

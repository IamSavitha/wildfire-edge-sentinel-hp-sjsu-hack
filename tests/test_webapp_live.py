"""Live camera, phone-alert delivery and notify endpoints of the demo web app (fakes only)."""
import json

import pytest
from fastapi.testclient import TestClient

from sentinel.notify import NotifyError
from sentinel.schema import Detection
from sentinel.webapp import create_web_app, main as webapp_main
from tests.test_webapp import TOWER, FakeVLM, jpeg

SMOKE = [Detection("smoke", 0.9, (40, 40, 200, 160))]
SECRET_TOPIC = "https://ntfy.example.org/very-secret-topic-name"


class FakeNotifier:
    def __init__(self, results=("sent",)):
        self.results = list(results)
        self.alerts = []
        self.tests = 0
        self.target = "ntfy.example.org/very…"

    def _next(self):
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(r, Exception):
            raise r
        return r

    def notify_alert(self, report):
        self.alerts.append(report)
        return self._next()

    def send_test(self):
        self.tests += 1
        return self._next()

    def stats(self):
        return {"sent": len(self.alerts), "suppressed": 0, "failed": 0, "deferred": 0, "host": "ntfy.example",
                "min_interval_s": 30, "last_error": None}


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


class Det:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.dets


def make_live_app(tmp_path, notifier=None, dets=None, **kw):
    made = {"detectors": {}, "vlms": {}}
    dets = list(SMOKE) if dets is None else dets

    def detector_factory(spec):
        made["detectors"][spec["weights"]] = d = Det(dets)
        return d

    def vlm_factory(model):
        made["vlms"][model] = v = FakeVLM(model)
        return v
    prices = tmp_path / "cost_inputs.json"
    prices.write_text(json.dumps({"usd_per_mtok_in": 0, "usd_per_mtok_out": 0, "usd_per_gb": 0}))
    clock = kw.pop("clock", None) or Clock()
    app = create_web_app(detector_factory=detector_factory, vlm_factory=vlm_factory,
                         model_lister=lambda: ["base7b", "context"],
                         forecast_fn=lambda lat, lon: {"temp_c": [30.0]}, monitor=None,
                         prices_path=prices, sample_dirs=[], results_dir=tmp_path, tower=TOWER,
                         detector_available=lambda spec: True, notifier=notifier, clock=clock, **kw)
    return TestClient(app), made, clock, app


def post_frame(client, body=None, stream=None, ctype="image/jpeg"):
    url = "/api/live/frame" + (f"?stream={stream}" if stream else "")
    return client.post(url, content=jpeg(640, 480) if body is None else body, headers={"Content-Type": ctype})


def frames(client, clock, n):
    out = []
    for _ in range(n):
        out.append(post_frame(client).json())
        clock.t += 1.0
    return out


def test_live_gate_needs_three_frames_then_one_alert_is_pushed(tmp_path):
    n = FakeNotifier()
    client, made, clock, _ = make_live_app(tmp_path, n)
    r1, r2, r3 = frames(client, clock, 3)
    assert r1["gate"]["streak"] == 1 and r2["gate"]["streak"] == 2
    assert r1["frame"] == {"width": 640, "height": 480}
    assert r1["detections"][0]["box"] == [40.0, 40.0, 200.0, 160.0]
    assert made["vlms"].get("context") is not None and made["vlms"]["context"].calls == 1
    assert "base7b" not in made["vlms"]                       # AFTER VLM only
    assert r3["new_alert"]["severity"] == "ALERT"
    assert r3["new_alert"]["delivery"]["state"] == "sent" and r3["new_alert"]["delivery"]["phone"] == "sent"
    assert len(n.alerts) == 1 and n.alerts[0]["tower_name"] == "Live camera"
    assert n.alerts[0]["tower_id"] == "live-camera"
    later = frames(client, clock, 8)                          # same fire: latched
    assert made["vlms"]["context"].calls == 1 and len(n.alerts) == 1
    assert all(r["new_alert"] is None for r in later) and later[-1]["latched"]["active"]


def test_live_uses_the_after_detector(tmp_path):
    client, made, clock, _ = make_live_app(tmp_path)
    post_frame(client)
    assert list(made["detectors"]) == ["models/smoke_yolo.pt"]


def test_live_offline_alert_is_queued_and_delivered_once_when_online(tmp_path):
    n = FakeNotifier()
    client, made, clock, _ = make_live_app(tmp_path, n)
    client.post("/api/network", json={"online": False})
    r = frames(client, clock, 3)[-1]
    assert r["new_alert"]["delivery"]["state"] == "queued" and r["delivery"]["pending"] == 1
    assert n.alerts == []
    resp = client.post("/api/network", json={"online": True}).json()
    assert resp["online"] is True and resp["flushed"] == 1
    assert len(n.alerts) == 1
    client.post("/api/network", json={"online": False})
    client.post("/api/network", json={"online": True})
    assert len(n.alerts) == 1
    d = client.get("/api/session").json()["delivery"]
    assert d["pending"] == 0 and d["events"][0]["state"] == "sent"


def test_live_failed_push_stays_queued(tmp_path):
    n = FakeNotifier([NotifyError("ntfy 500"), "sent"])
    client, made, clock, app = make_live_app(tmp_path, n)
    r = frames(client, clock, 3)[-1]
    assert r["new_alert"]["delivery"]["state"] == "retrying" and r["delivery"]["pending"] == 1
    clock.t += 5
    assert app.state.delivery.flush(clock()) == 1
    assert client.get("/api/session").json()["delivery"]["pending"] == 0


def test_live_busy_returns_202_immediately(tmp_path):
    client, made, clock, app = make_live_app(tmp_path)
    assert app.state.live.busy.acquire(blocking=False)
    try:
        r = post_frame(client)
        assert r.status_code == 202 and r.json() == {"busy": True}
    finally:
        app.state.live.busy.release()
    assert post_frame(client).status_code == 200


def test_live_rejects_oversize_wrong_type_and_bad_stream(tmp_path):
    client, made, clock, _ = make_live_app(tmp_path, max_upload_bytes=1000)
    assert post_frame(client, body=b"\xff" * 2000).status_code == 413
    client2, *_ = make_live_app(tmp_path)
    assert post_frame(client2, ctype="image/png").status_code == 415
    assert post_frame(client2, body=b"not a jpeg").status_code == 415
    assert post_frame(client2, body=b"").status_code == 400
    assert post_frame(client2, stream="../etc").status_code == 400
    assert post_frame(client2, stream="cam-1_A").status_code == 200


def test_live_reset_clears_the_latch(tmp_path):
    client, made, clock, _ = make_live_app(tmp_path)
    frames(client, clock, 3)
    r = client.post("/api/live/reset").json()
    assert r["events"] == [] and r["latched"]["active"] is False and r["frames"] == 0
    frames(client, clock, 3)
    assert made["vlms"]["context"].calls == 2


def test_live_recheck_interval_is_configurable(tmp_path):
    client, made, clock, app = make_live_app(tmp_path, live_recheck_s=4.0)
    assert client.get("/api/config").json()["live"]["recheck_s"] == 4.0


def test_notify_test_endpoint(tmp_path):
    client, *_ = make_live_app(tmp_path)
    r = client.post("/api/notify/test")
    assert r.status_code == 409 and "NTFY_TOPIC_URL" in r.json()["detail"]
    n = FakeNotifier(["sent", "suppressed", NotifyError("ntfy ntfy.example.org/very… answered HTTP 500")])
    client, *_ = make_live_app(tmp_path, n)
    assert client.post("/api/notify/test").json()["status"] == "sent"
    assert client.post("/api/notify/test").json()["status"] == "suppressed"
    r = client.post("/api/notify/test")
    assert r.status_code == 502 and "HTTP 500" in r.json()["detail"]


def test_config_reports_phone_alerts_without_the_topic(tmp_path, monkeypatch):
    from sentinel.notify import NtfyNotifier
    n = NtfyNotifier(SECRET_TOPIC, post=lambda *a, **k: None)
    client, *_ = make_live_app(tmp_path, n)
    text = client.get("/api/config").text
    c = json.loads(text)
    assert c["phone"] == {"configured": True, "target": "ntfy.example.org/very…"}
    assert c["live"]["tower"]["name"] == "Live camera" and c["live"]["min_frames"] == 3
    assert "very-secret" not in text
    assert "very-secret" not in client.get("/api/session").text


def test_analyze_alert_is_pushed_once_per_request(tmp_path):
    n = FakeNotifier()
    client, made, clock, _ = make_live_app(tmp_path, n)
    img = {"image": ("x.jpg", jpeg(), "image/jpeg")}
    r = client.post("/api/analyze", files=img, data={"pipelines": "before,after"}).json()
    assert [x["severity"] for x in r["results"]] == ["ALERT", "ALERT"]
    assert len(n.alerts) == 1                     # one photo, one push (the AFTER pass's report)
    after = r["results"][1]
    assert after["escalation"]["decision"] == "sent"
    assert after["escalation"]["delivery"]["phone"] == "sent"
    assert "delivery" not in r["results"][0]["escalation"]


def test_analyze_offline_alert_waits_for_the_link(tmp_path):
    n = FakeNotifier()
    client, made, clock, _ = make_live_app(tmp_path, n)
    client.post("/api/network", json={"online": False})
    img = {"image": ("x.jpg", jpeg(), "image/jpeg")}
    r = client.post("/api/analyze", files=img, data={"pipelines": "after"}).json()
    assert r["results"][0]["escalation"]["decision"] == "queued"
    assert r["results"][0]["escalation"]["delivery"]["state"] == "queued" and n.alerts == []
    client.post("/api/network", json={"online": True})
    assert len(n.alerts) == 1


def test_analyze_without_delivery_configured_is_unchanged(tmp_path):
    client, made, clock, _ = make_live_app(tmp_path)
    img = {"image": ("x.jpg", jpeg(), "image/jpeg")}
    r = client.post("/api/analyze", files=img, data={"pipelines": "after"}).json()
    assert "delivery" not in r["results"][0]["escalation"]
    assert client.get("/api/session").json()["delivery"]["pending"] == 0


def test_outbox_persists_under_the_state_dir(tmp_path):
    n = FakeNotifier()
    db = tmp_path / "state" / "live_outbox.db"
    client, made, clock, _ = make_live_app(tmp_path, n, outbox_path=str(db))
    client.post("/api/network", json={"online": False})
    frames(client, clock, 3)
    assert db.exists()
    n2 = FakeNotifier()
    client2, *_ = make_live_app(tmp_path, n2, outbox_path=str(db),   # restart: the queued alert survives
                                clock=Clock(clock.t + 60))
    client2.post("/api/network", json={"online": False})
    client2.post("/api/network", json={"online": True})
    assert len(n2.alerts) == 1


def test_cli_help_lists_live_flags(capsys):
    with pytest.raises(SystemExit):
        webapp_main(["--help"])
    out = capsys.readouterr().out
    for flag in ("--live-recheck-s", "--live-lat", "--live-lon", "--dispatch-url", "--state-dir"):
        assert flag in out


def test_background_retry_delivers_a_failed_push(tmp_path):
    import time
    n = FakeNotifier([NotifyError("ntfy down"), "sent"])
    client, made, clock, app = make_live_app(tmp_path, n, flush_every_s=0.01)
    with client:                                   # runs the lifespan: starts the retry thread
        frames(client, clock, 3)
        clock.t += 5                               # past the outbox backoff
        for _ in range(300):
            if app.state.delivery.outbox.pending_count() == 0:
                break
            time.sleep(0.01)
    assert len(n.alerts) == 2 and app.state.delivery.outbox.pending_count() == 0

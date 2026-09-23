import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentinel.config import Tower
from sentinel.schema import ContextResult, Detection
from sentinel.webapp import create_web_app, parse_multipart, scan_samples

WILD = ContextResult(source_type="wildland", smoke_color="black", attended="no", near_structures=True,
                     near_road=False, size_estimate="medium", description="Smoke near a cabin.")
TOWER = Tower("t1", "Test Lookout", 37.1, -121.9, "", temp_c=30.0)


def jpeg(w=320, h=240, value=120):
    ok, buf = cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))
    return buf.tobytes()


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.dets


class FakeVLM:
    def __init__(self, model):
        self.model = model
        self.calls = 0

    def classify(self, jpeg_bytes):
        self.calls += 1
        self.last_usage = {"prompt_tokens": 380, "completion_tokens": 40, "calls": 1}
        return WILD, 420


@pytest.fixture
def samples(tmp_path):
    dfire = tmp_path / "data" / "dfire" / "test"
    (dfire / "images").mkdir(parents=True)
    (dfire / "labels").mkdir()
    for i in range(5):
        (dfire / "images" / f"img{i}.jpg").write_bytes(jpeg())
        (dfire / "labels" / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n" if i % 2 == 0 else "")
    demo = tmp_path / "data" / "demo" / "seq1"
    demo.mkdir(parents=True)
    (demo / "1465065268_-00120.jpg").write_bytes(jpeg())
    (demo / "1465065268_+00300.jpg").write_bytes(jpeg())
    benign = tmp_path / "data" / "benign" / "fog"
    benign.mkdir(parents=True)
    (benign / "fog1.png").write_bytes(cv2.imencode(".png", np.zeros((50, 60, 3), np.uint8))[1].tobytes())
    (benign / "notes.txt").write_text("not an image")
    return [dfire / "images", tmp_path / "data" / "demo", tmp_path / "data" / "benign"]


def make_app(tmp_path, samples=(), served=("base7b", "context"), dets=None, available=None, **kw):
    made = {"detectors": {}, "vlms": {}, "forecasts": []}
    dets = [Detection("smoke", 0.9, (40, 40, 200, 160))] if dets is None else dets

    def detector_factory(spec):
        d = FakeDetector(dets)
        made["detectors"][spec["weights"]] = d
        return d

    def vlm_factory(model):
        v = FakeVLM(model)
        made["vlms"][model] = v
        return v

    def forecast(lat, lon):
        made["forecasts"].append((lat, lon))
        return {"time": ["t"], "temp_c": [35.0], "wind_mph": [9.0], "wind_dir_deg": [180.0]}

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "detector_after_yolo11s.json").write_text(json.dumps(
        {"map50": 0.787, "precision": 0.776, "recall": 0.719, "ms_per_image": 3.3}))
    prices = tmp_path / "cost_inputs.json"
    prices.write_text(json.dumps({"usd_per_mtok_in": 0, "usd_per_mtok_out": 0, "usd_per_gb": 0}))
    app = create_web_app(
        detector_factory=detector_factory, vlm_factory=vlm_factory,
        model_lister=lambda: list(served), forecast_fn=forecast, monitor=kw.pop("monitor", None),
        prices_path=prices, sample_dirs=list(samples), results_dir=results, tower=TOWER,
        detector_available=available or (lambda spec: True), **kw)
    return TestClient(app), made


def test_index_serves_two_pane_page(tmp_path):
    client, _ = make_app(tmp_path)
    html = client.get("/").text
    assert "Try the models" in html and "Usage &amp; economics" in html


def test_config_marks_available_pipelines(tmp_path):
    client, _ = make_app(tmp_path, served=["base7b"],
                         available=lambda spec: spec["weights"] != "yolov8s-worldv2.pt")
    c = client.get("/api/config").json()
    assert c["served_models"] == ["base7b"]
    p = {x["name"]: x for x in c["pipelines"]}
    assert p["before"]["detector"] == "yoloworld" and p["before"]["vlm"] == "base7b"
    assert p["before"]["detector_available"] is False and p["before"]["vlm_served"] is True
    assert p["before"]["available"] is False
    assert p["after"]["detector_available"] is True and p["after"]["vlm_served"] is False
    assert p["after"]["available"] is True  # runs detector-only; the card says the VLM is not served
    assert p["teacher"]["vlm_served"] is False and p["teacher"]["available"] is False
    assert c["benchmarks"]["detector"]["after"]["map50"] == 0.787
    assert c["benchmarks"]["context"]["after"] is None
    assert c["prices"]["usd_per_mtok_in"] == 0 and c["online"] is True
    assert c["min_conf"] == 0.4


def test_config_survives_a_failing_model_lister(tmp_path):
    def boom():
        raise OSError("connection refused")
    app = create_web_app(detector_factory=lambda s: FakeDetector([]), vlm_factory=FakeVLM,
                         model_lister=boom, forecast_fn=lambda a, b: {}, monitor=None,
                         prices_path=tmp_path / "none.json", sample_dirs=[], results_dir=tmp_path,
                         tower=TOWER, detector_available=lambda s: True)
    c = TestClient(app).get("/api/config").json()
    assert c["served_models"] == [] and "connection refused" in c["models_error"]


def test_scan_samples_groups_hints_and_cap(samples):
    items = scan_samples(samples, cap=60)
    groups = {s["group"] for s in items}
    assert groups == {"dfire_test", "tower_fire_sequence", "benign_tower"}
    assert len(items) == 8 and all(len(s["id"]) == 12 for s in items)
    by_name = {s["name"]: s for s in items}
    assert by_name["img0.jpg"]["truth"] == "wildfire" and "smoke" in by_name["img0.jpg"]["hint"]
    assert by_name["img1.jpg"]["truth"] == "no_wildfire"
    assert by_name["1465065268_-00120.jpg"]["truth"] == "no_wildfire"
    assert by_name["1465065268_+00300.jpg"]["truth"] == "wildfire"
    assert by_name["fog1.png"]["truth"] == "no_wildfire"
    assert scan_samples(samples, cap=60) == items  # deterministic
    assert len(scan_samples(samples, cap=3)) == 3


def test_samples_listing_and_bytes(tmp_path, samples):
    client, _ = make_app(tmp_path, samples)
    items = client.get("/api/samples").json()["samples"]
    assert len(items) == 8 and "path" not in items[0]
    r = client.get(f"/api/sample/{items[0]['id']}")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")
    t = client.get(f"/api/sample/{items[0]['id']}?thumb=1")
    assert t.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(t.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 200


@pytest.mark.parametrize("bad", ["..%2F..%2Fetc%2Fpasswd", "abcdef012345", "ABCDEF012345", "x" * 12,
                                 "%2e%2e", "notes.txt"])
def test_sample_rejects_unknown_ids_and_traversal(tmp_path, samples, bad):
    client, _ = make_app(tmp_path, samples)
    assert client.get(f"/api/sample/{bad}").status_code == 404


def test_analyze_sample_runs_both_pipelines(tmp_path, samples):
    client, made = make_app(tmp_path, samples)
    sid = client.get("/api/samples").json()["samples"][0]["id"]
    r = client.post("/api/analyze", data={"sample_id": sid, "truth": "wildfire",
                                          "pipelines": "before,after"})
    assert r.status_code == 200, r.text
    body = r.json()
    names = [x["pipeline"] for x in body["results"]]
    assert names == ["before", "after"]
    after = body["results"][1]
    assert after["vlm"] == "context" and after["vlm_called"] is True
    assert after["severity"] == "ALERT" and after["correct"] is True
    assert after["escalation"]["decision"] == "sent" and after["escalation"]["bytes_up"] > 0
    assert "report" not in after and after["report_text"].startswith("[ALERT]")
    assert set(made["vlms"]) == {"base7b", "context"}
    assert set(made["detectors"]) == {"yolov8s-worldv2.pt", "models/smoke_yolo.pt"}
    s = body["session"]["pipelines"]["after"]
    assert s["images"] == 1 and s["vlm_calls"] == 1 and s["accuracy"] == 1.0
    assert body["image"]["source"] == "sample" and body["image"]["bytes"] > 0
    assert body["economics"]["pipelines"]["after"]["edge_tokens"] == 420


def test_detectors_are_loaded_once_and_cached(tmp_path, samples):
    client, made = make_app(tmp_path, samples)
    sid = client.get("/api/samples").json()["samples"][0]["id"]
    for _ in range(2):
        client.post("/api/analyze", data={"sample_id": sid, "pipelines": "after"})
    assert made["detectors"]["models/smoke_yolo.pt"].calls == 2
    assert made["vlms"]["context"].calls == 2


def test_analyze_upload_and_repeated_pipeline_fields(tmp_path):
    client, _ = make_app(tmp_path)
    r = client.post("/api/analyze", files={"image": ("cam.jpg", jpeg(640, 480), "image/jpeg")},
                    data={"pipelines": ["after", "before"], "truth": "no_wildfire"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [x["pipeline"] for x in body["results"]] == ["after", "before"]
    assert body["image"] == {"source": "upload", "name": "cam.jpg", "width": 640, "height": 480,
                             "bytes": len(jpeg(640, 480))}
    assert body["results"][0]["correct"] is False


def test_analyze_cascade_skip_and_force_vlm(tmp_path):
    client, made = make_app(tmp_path, dets=[])
    img = {"image": ("x.jpg", jpeg(), "image/jpeg")}
    r = client.post("/api/analyze", files=img, data={"pipelines": "after"}).json()["results"][0]
    assert r["vlm_called"] is False and r["vlm_status"] == "skipped_no_detection"
    assert r["severity"] == "IGNORE" and r["escalation"]["decision"] == "ignored"
    r = client.post("/api/analyze", files=img,
                    data={"pipelines": "after", "force_vlm": "true"}).json()["results"][0]
    assert r["vlm_called"] is True and r["vlm_input"] == "full_frame"
    s = client.get("/api/session").json()["session"]["pipelines"]["after"]
    assert s["calls_avoided"] == 1 and s["vlm_calls"] == 1


def test_analyze_vlm_not_served_runs_detector_only(tmp_path):
    client, made = make_app(tmp_path, served=["base7b"])
    r = client.post("/api/analyze", files={"image": ("x.jpg", jpeg(), "image/jpeg")},
                    data={"pipelines": "after"}).json()["results"][0]
    assert r["vlm_status"] == "unavailable" and r["vlm_served"] is False
    assert r["severity"] == "MONITOR" and "context" not in made["vlms"]


def test_analyze_unavailable_detector_returns_error_entry(tmp_path):
    client, _ = make_app(tmp_path, available=lambda spec: False)
    r = client.post("/api/analyze", files={"image": ("x.jpg", jpeg(), "image/jpeg")},
                    data={"pipelines": "before"})
    assert r.status_code == 200
    res = r.json()["results"][0]
    assert res["pipeline"] == "before" and "not found" in res["error"]
    assert r.json()["session"]["pipelines"] == {}


@pytest.mark.parametrize("data,files,code", [
    ({"pipelines": "after"}, None, 400),                                         # no image
    ({"pipelines": "nope"}, {"image": ("x.jpg", jpeg(), "image/jpeg")}, 400),    # unknown pipeline
    ({"pipelines": "after", "truth": "maybe"}, {"image": ("x.jpg", jpeg(), "image/jpeg")}, 400),
    ({"pipelines": "after", "sample_id": "../../etc"}, None, 404),
    ({"pipelines": "after"}, {"image": ("x.txt", b"hello world", "text/plain")}, 415),
])
def test_analyze_rejects_bad_requests(tmp_path, data, files, code):
    client, _ = make_app(tmp_path)
    assert client.post("/api/analyze", data=data, files=files).status_code == code


def test_analyze_rejects_oversized_upload(tmp_path):
    client, _ = make_app(tmp_path, max_upload_bytes=1000)
    r = client.post("/api/analyze", files={"image": ("x.jpg", jpeg(640, 480, 7) + b"\0" * 2000, "image/jpeg")},
                    data={"pipelines": "after"})
    assert r.status_code == 413


def test_analyze_requires_multipart(tmp_path):
    client, _ = make_app(tmp_path)
    assert client.post("/api/analyze", json={"sample_id": "x"}).status_code == 415


def test_network_toggle_changes_the_decision(tmp_path):
    client, made = make_app(tmp_path)
    img = {"image": ("x.jpg", jpeg(), "image/jpeg")}
    assert client.post("/api/network", json={"online": False}).json() == {"online": False}
    r = client.post("/api/analyze", files=img, data={"pipelines": "after"}).json()
    esc = r["results"][0]["escalation"]
    assert esc["decision"] == "queued" and esc["bytes_up"] == 0 and made["forecasts"] == []
    assert r["online"] is False
    client.post("/api/network", json={"online": True})
    esc = client.post("/api/analyze", files=img, data={"pipelines": "after"}).json()["results"][0]["escalation"]
    assert esc["decision"] == "sent" and esc["forecast"]["temp_c"] == [35.0]
    assert made["forecasts"] == [(37.1, -121.9)]
    n = client.get("/api/session").json()["session"]["network"]
    assert n["offline_requests"] == 1 and n["cloud_only_blind"] == 1
    assert n["alerts_sent"] == 1 and n["alerts_queued"] == 1


def test_prices_update_economics_and_validate(tmp_path):
    client, _ = make_app(tmp_path)
    client.post("/api/analyze", files={"image": ("x.jpg", jpeg(), "image/jpeg")}, data={"pipelines": "after"})
    assert client.get("/api/session").json()["economics"]["priced"] is False
    r = client.post("/api/prices", json={"usd_per_mtok_in": 2.0, "usd_per_mtok_out": 8.0, "usd_per_gb": 1.0})
    assert r.status_code == 200 and r.json()["usd_per_mtok_in"] == 2.0
    e = client.get("/api/session").json()["economics"]
    assert e["priced"] is True and e["pipelines"]["after"]["cloud_usd"] > 0
    assert client.post("/api/prices", json={"usd_per_mtok_in": -1}).status_code == 422
    assert client.post("/api/prices", json={"bogus": 1}).status_code == 422


def test_session_reset(tmp_path):
    client, _ = make_app(tmp_path)
    client.post("/api/analyze", files={"image": ("x.jpg", jpeg(), "image/jpeg")}, data={"pipelines": "after"})
    assert client.get("/api/session").json()["session"]["pipelines"]["after"]["images"] == 1
    client.post("/api/session/reset")
    s = client.get("/api/session").json()
    assert s["session"]["pipelines"] == {} and s["session"]["network"]["requests"] == 0


def test_session_includes_live_monitor_data(tmp_path):
    class FakeMonitor:
        started = stopped = False

        def start(self):
            FakeMonitor.started = True

        def stop(self):
            FakeMonitor.stopped = True

        def latest(self):
            return {"ts": 1.0, "models": [{"label": "context", "status": "ok", "gen_tok_per_s": 30.0}],
                    "system": {"gpu_util_pct": 55.0}, "economics": {}, "prices": {}}
    client, _ = make_app(tmp_path, monitor=FakeMonitor())
    with client:
        live = client.get("/api/session").json()["live"]
        assert live["models"][0]["label"] == "context" and live["system"]["gpu_util_pct"] == 55.0
        assert "economics" not in live
        assert FakeMonitor.started
    assert FakeMonitor.stopped


def test_session_without_monitor(tmp_path):
    client, _ = make_app(tmp_path)
    assert client.get("/api/session").json()["live"] is None


def test_parse_multipart_keeps_binary_bytes():
    blob = bytes(range(256)) * 4 + b"\r\n--notboundary\r\n\n\r"
    boundary = "XyZ123"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"truth\"\r\n\r\nwildfire\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"pipelines\"\r\n\r\nbefore\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"pipelines\"\r\n\r\nafter\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    fields, files = parse_multipart(body, f"multipart/form-data; boundary={boundary}")
    assert fields == {"truth": ["wildfire"], "pipelines": ["before", "after"]}
    assert files["image"] == ("a.jpg", blob)



def test_warmup_runs_each_available_detector_once(tmp_path):
    client, made = make_app(tmp_path, warmup=True,
                            available=lambda spec: spec["weights"] != "yolov8s-worldv2.pt")
    with client:
        client.app.state.warmup_thread.join(timeout=5)
        assert set(made["detectors"]) == {"models/smoke_yolo.pt"}
        assert made["detectors"]["models/smoke_yolo.pt"].calls == 1
        # a later pass reuses the warmed instance
        client.post("/api/analyze", files={"image": ("x.jpg", jpeg(), "image/jpeg")}, data={"pipelines": "after"})
        assert made["detectors"]["models/smoke_yolo.pt"].calls == 2


def test_no_warmup_by_default(tmp_path):
    client, made = make_app(tmp_path)
    with client:
        assert client.app.state.warmup_thread is None
    assert made["detectors"] == {}


def test_warmup_failure_is_contained(tmp_path):
    def broken(spec):
        raise RuntimeError("CUDA not ready")
    app = create_web_app(detector_factory=broken, vlm_factory=FakeVLM, model_lister=lambda: [],
                         forecast_fn=lambda a, b: {}, monitor=None, prices_path=tmp_path / "p.json",
                         sample_dirs=[], results_dir=tmp_path, tower=TOWER,
                         detector_available=lambda s: True, warmup=True)
    with TestClient(app) as client:
        app.state.warmup_thread.join(timeout=5)
        assert client.get("/api/config").status_code == 200


def test_content_type_check_is_case_insensitive(tmp_path):
    client, _ = make_app(tmp_path)
    boundary = "b0undary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"pipelines\"\r\n\r\nafter\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.jpg\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + jpeg() + f"\r\n--{boundary}--\r\n".encode()
    r = client.post("/api/analyze", content=body,
                    headers={"Content-Type": f"Multipart/Form-Data; boundary={boundary}"})
    assert r.status_code == 200, r.text


def test_pass_card_fields_include_native_full_frame(tmp_path):
    client, _ = make_app(tmp_path)
    r = client.post("/api/analyze", files={"image": ("x.jpg", jpeg(1920, 1080), "image/jpeg")},
                    data={"pipelines": "after"}).json()["results"][0]
    assert r["full_frame_image_tokens"] == 26 * 46 and r["full_frame_image_tokens_native"] == 39 * 69

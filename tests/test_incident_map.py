import io
from pathlib import Path
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sentinel.incident_map import create_map_app, load_cameras, scan_recordings, scan_samples
from sentinel.schema import ContextResult, Detection
from sentinel.incidents import haversine_km
from sentinel.wind import SensorEmulator

WILD = ContextResult(source_type="wildland", smoke_color="black", attended="no", near_structures=True,
                     near_road=False, size_estimate="medium", description="Black smoke on a ridge.")
FOG = ContextResult(source_type="fog_dust_cloud", smoke_color="white", attended="unclear", near_structures=False,
                    near_road=False, size_estimate="large", description="Low cloud over the valley.")
CAMPFIRE = ContextResult(source_type="campfire", smoke_color="white", attended="yes", near_structures=False,
                         near_road=False, size_estimate="small", description="A small attended campfire.")
CAMS = {"rm-e": {"id": "rm-e", "name": "Red Mountain (east)", "site": "Red Mountain, Fallbrook",
                 "lat": 33.40079, "lon": -117.19047, "azimuth_deg": 90, "hfov_deg": 90,
                 "wx_station": "RM-WXT536", "figlib": "fire_x"},
        "sdsc-e": {"id": "sdsc-e", "name": "SDSC (east)", "site": "SDSC", "lat": 32.88, "lon": -117.24,
                   "azimuth_deg": 90, "hfov_deg": 90, "wx_station": None, "figlib": None}}
COUNTIES = {"features": [{"properties": {"name": "San Diego"}, "geometry": {"type": "Polygon", "coordinates": [
    [[-118, 32], [-116, 32], [-116, 34], [-118, 34], [-118, 32]]]}}]}
FORECAST = {"time": ["t0"], "temp_c": [30.0], "wind_mph": [20.0], "wind_dir_deg": [250.0]}


def jpeg(w=400, h=200, value=120):
    return cv2.imencode(".jpg", np.full((h, w, 3), value, np.uint8))[1].tobytes()


class FakeDetector:
    def __init__(self):
        self.dets = [Detection(box=(300, 50, 400, 150), conf=0.8, cls="smoke")]   # right half: bearing > 90

    def __call__(self, frame):
        return self.dets


class FakeVLM:
    ctx = WILD

    def __init__(self, model):
        self.model = model

    def classify(self, jpeg_bytes):
        self.last_usage = {"prompt_tokens": 300, "completion_tokens": 40, "calls": 1}
        return FakeVLM.ctx, 340


class Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"source": "t", "captured_utc": "x", "readings": {
        "RM-WXT536": "1|Dn=250D,Dm=270D,Dx=290D,Sn=4.0M,Sm=5.0M,Sx=7.0M,Ta=30.0C,Ua=10.0P"}}))
    clock = Clock()
    det = FakeDetector()
    calls = {"forecast": 0}

    def forecast(lat, lon):
        calls["forecast"] += 1
        return FORECAST

    seq = tmp_path / "data" / "demo" / "fire_x"
    seq.mkdir(parents=True)
    (seq / "1_-00060.jpg").write_bytes(jpeg())
    (seq / "2_+00120.jpg").write_bytes(jpeg())
    FakeVLM.ctx = WILD
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "assets", data_dir=tmp_path / "data",
                         detector_weights="w.pt", detector_factory=lambda w: det, vlm_factory=FakeVLM,
                         model_lister=lambda: ["context"], forecast_fn=forecast,
                         sensor=SensorEmulator(seed, clock=clock), counties=COUNTIES, clock=clock)
    with TestClient(app) as c:
        yield c, det, calls, clock


def post(c, **fields):
    files = {"image": ("f.jpg", fields.pop("image", jpeg()), "image/jpeg")} if "sample_id" not in fields else None
    return c.post("/api/report", data=fields, files=files)


def test_tower_alert_is_placed_along_the_bearing_and_queued_offline(env):
    c, det, calls, _ = env
    r = post(c, camera_id="rm-e")
    assert r.status_code == 200, r.text
    inc = r.json()["incident"]
    assert inc["severity"] == "ALERT" and inc["loc_source"] == "camera_bearing"
    assert 90 < inc["bearing_deg"] < 135 and inc["bearing_lo"] < inc["bearing_deg"] < inc["bearing_hi"]
    assert inc["lon"] > CAMS["rm-e"]["lon"] and inc["county"] == "San Diego"
    assert inc["escalation"]["decision"] == "queued" and calls["forecast"] == 0
    assert inc["wind"]["source"] == "sensor" and inc["wind"]["emulated"]
    assert inc["cone"]["toward_deg"] == pytest.approx(90, abs=31)
    assert inc["wedges"][0][0] == [CAMS["rm-e"]["lon"], CAMS["rm-e"]["lat"]]
    s = c.get("/api/state").json()
    assert s["summary"]["outbox"] == 1 and s["summary"]["active_fires"] == 1
    assert "thumbnail_b64" not in s["incidents"][0] and "report" not in s["incidents"][0]
    assert c.get(f"/api/incidents/{inc['id']}/thumb").headers["content-type"] == "image/jpeg"


def test_link_up_flushes_the_outbox_once_and_caches_the_forecast(env):
    c, _, calls, clock = env
    inc = post(c, camera_id="rm-e").json()["incident"]
    r = c.post("/api/network", json={"online": True}).json()
    assert r == {"online": True, "flushed": 1} and calls["forecast"] == 1
    got = c.get(f"/api/incidents/{inc['id']}").json()
    assert got["escalation"]["decision"] == "sent" and got["escalation"]["bytes_up"] > 0
    assert "queued_at" in got["escalation"]
    assert c.post("/api/network", json={"online": True}).json()["flushed"] == 0
    s = c.get("/api/state").json()["summary"]
    assert s["alerts_sent"] == 1 and s["alerts_queued"] == 0 and s["outbox"] == 0
    # the next frame of the same fire merges and does not re-send
    clock.t += 60
    again = post(c, camera_id="rm-e").json()
    assert not again["created"] and again["incident"]["detections"] == 2 and calls["forecast"] == 1


def test_no_detection_is_ignored_and_stores_nothing(env):
    c, det, _, _ = env
    det.dets = []
    r = post(c, camera_id="rm-e").json()
    assert r["incident"] is None and r["result"]["severity"] == "IGNORE"
    s = c.get("/api/state").json()
    assert s["incidents"] == [] and s["summary"]["ignored"] == 1 and s["summary"]["frames"] == 1


def test_benign_sources_stay_on_the_device(env):
    c, _, calls, _ = env
    FakeVLM.ctx = FOG
    r = post(c, camera_id="rm-e").json()
    assert r["result"]["severity"] == "IGNORE" and r["incident"] is None
    FakeVLM.ctx = CAMPFIRE
    inc = post(c, camera_id="rm-e").json()["incident"]
    assert inc["severity"] == "LOG" and inc["escalation"]["decision"] == "logged"
    c.post("/api/network", json={"online": True})
    assert calls["forecast"] == 0 and c.get("/api/state").json()["summary"]["bytes_up"] == 0


def test_reported_coordinates_and_exif_gps_take_priority(env):
    c, _, _, _ = env
    inc = post(c, lat="33.0", lon="-116.9").json()["incident"]
    assert (inc["lat"], inc["lon"], inc["loc_source"]) == (33.0, -116.9, "reported_gps")
    # no tower of its own: wind comes from the nearest tower station within 40 km (Red Mountain, 26 km here)
    near = post(c, lat="33.2", lon="-117.05").json()["incident"]["wind"]
    assert near["station"] == "RM-WXT536" and "nearest tower station (Red Mountain, 26 km)" in near["label"]
    assert inc["wind"] is None                                   # 52 km from the only station: no reading
    assert inc["camera_id"] is None and inc["wedges"] == [] and inc["name"] == "San Diego Incident 1"
    im = Image.new("RGB", (400, 200))
    exif = Image.Exif()
    exif.get_ifd(0x8825).update({1: "N", 2: (32.0, 45.0, 0.0), 3: "W", 4: (117.0, 0.0, 0.0)})
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    inc = post(c, image=buf.getvalue(), camera_id="rm-e", lat="33.0", lon="-116.9").json()["incident"]
    assert (inc["lat"], inc["lon"], inc["loc_source"]) == (32.75, -117.0, "exif_gps")


def test_report_needs_a_location_source_and_valid_input(env):
    c, _, _, _ = env
    assert post(c).status_code == 400
    assert post(c, camera_id="nope").status_code == 400
    assert post(c, lat="95", lon="0").status_code == 400
    assert post(c, lat="abc", lon="0").status_code == 400
    assert post(c, camera_id="rm-e", image=b"junk").status_code == 415


def test_manual_report_frame_has_the_full_analysis(env):
    c, _, _, _ = env
    r = post(c, camera_id="rm-e").json()
    f = c.get(f"/api/incidents/{r['incident']['id']}").json()["frames"][r["frame_n"]]
    a = f["analysis"]
    assert a["context"]["source_type"] == "wildland" and a["severity"] == "ALERT" and a["tokens"] == 340
    assert a["escalation"]["decision"] == "queued" and "[ALERT]" in a["report_text"]
    assert f["dets"][0]["strong"] and f["source"] == "f.jpg"


def test_flame_photos_are_picked_from_dfire_labels(tmp_path):
    from sentinel.incident_map import pick_flame_photos
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    rows = {"a": "1 0.5 0.5 0.3 0.3\n0 0.5 0.2 0.4 0.2\n",      # big flames + smoke: picked
            "b": "1 0.5 0.5 0.05 0.05\n0 0.5 0.2 0.4 0.2\n",    # flames too small
            "c": "1 0.5 0.5 0.3 0.3\n",                          # no smoke
            "d": ""}
    for k, v in rows.items():
        (tmp_path / "labels" / f"{k}.txt").write_text(v)
        (tmp_path / "images" / f"{k}.jpg").write_bytes(jpeg())
    assert [p.stem for p in pick_flame_photos(tmp_path)] == ["a"]


def test_samples_carry_their_camera(env):
    c, _, _, _ = env
    samples = c.get("/api/samples").json()["samples"]
    assert {s["offset_s"] for s in samples} == {-60, 120} and {s["camera_id"] for s in samples} == {"rm-e"}
    sid = next(s["id"] for s in samples if s["offset_s"] == 120)
    inc = post(c, sample_id=sid).json()["incident"]
    assert inc["camera_id"] == "rm-e" and inc["frame_offset_s"] == 120
    assert c.get(f"/api/sample/{sid}").headers["content-type"] == "image/jpeg"
    assert c.get("/api/sample/000000000000").status_code == 404


def test_patch_pin_status_and_manual_wind(env):
    c, _, _, clock = env
    inc = post(c, camera_id="sdsc-e").json()["incident"]
    assert inc["wind"] is None and inc["cone"] is None          # no station at SDSC, no forecast yet
    r = c.patch(f"/api/incidents/{inc['id']}", json={"lat": 32.9, "lon": -117.1}).json()
    assert r["loc_source"] == "map_pin" and r["county"] == "San Diego"
    r = c.patch(f"/api/incidents/{inc['id']}", json={"wind_from_deg": 450, "wind_speed_mps": 5}).json()
    assert r["wind"]["source"] == "manual" and r["wind"]["dir_from_deg"] == 90 and r["cone"]["toward_deg"] == 270
    assert c.patch(f"/api/incidents/{inc['id']}", json={"lat": 32.9}).status_code == 400
    assert c.patch(f"/api/incidents/{inc['id']}", json={"wind_from_deg": 90}).status_code == 400
    assert c.patch(f"/api/incidents/{inc['id']}", json={"status": "burning"}).status_code == 400
    c.patch(f"/api/incidents/{inc['id']}", json={"status": "resolved"})
    assert c.get("/api/state").json()["summary"]["active"] == 0
    assert c.patch("/api/incidents/nope", json={"status": "resolved"}).status_code == 404


def test_config_state_and_reset(env):
    c, _, _, _ = env
    cfg = c.get("/api/config").json()
    assert cfg["online"] is False and cfg["vlm_served"] and cfg["sensor"]["emulated"]
    assert {cam["id"] for cam in cfg["cameras"]} == {"rm-e", "sdsc-e"}
    assert cfg["cameras"][0]["wind"]["station"] == "RM-WXT536" and len(cfg["cameras"][0]["fov"]) > 3
    post(c, camera_id="rm-e")
    assert c.get("/api/state?hours=100").json()["hours"] == 12.0
    c.post("/api/reset")
    assert c.get("/api/state").json()["incidents"] == []


def test_vlm_not_served_still_records_the_detection(tmp_path):
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "a2", detector_weights="w.pt",
                         detector_factory=lambda w: FakeDetector(), vlm_factory=FakeVLM,
                         model_lister=lambda: [], forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES)
    with TestClient(app) as c2:
        r = post(c2, camera_id="rm-e").json()
    assert r["result"]["vlm_status"] == "unavailable" and r["incident"]["severity"] == "MONITOR"


def test_load_cameras_real_config():
    cams = load_cameras("config/cameras.json")
    assert {"rm-e", "rm-s", "lp-w", "hp-n", "sdsc-e"} <= set(cams)
    assert all(-125 < c["lon"] < -114 and 32 < c["lat"] < 42 for c in cams.values())
    junction = [c for c in cams.values() if any(r["fire"] == "Junction Fire" for r in c["recordings"])]
    assert len(junction) == 6


def test_scan_samples_missing_dirs(tmp_path):
    assert scan_samples([tmp_path], None, CAMS) == {} and scan_recordings([tmp_path], CAMS) == {}


class AimedDetector:
    """Smoke from the 2nd frame on, placed in the frame where the camera's heading puts `fire`."""
    def __init__(self, cams, fire):
        from sentinel.geo import pixel_bearing
        self.cams, self.fire, self.pb = cams, fire, pixel_bearing
        self.seen = {}

    def aim(self, cam):
        from sentinel.geo import destination
        target = min(range(1000), key=lambda i: haversine_km(*destination(
            cam["lat"], cam["lon"], self.pb(cam["azimuth_deg"], cam["hfov_deg"], i / 1000),
            haversine_km(cam["lat"], cam["lon"], *self.fire)), *self.fire))
        return target / 1000

    def __call__(self, frame):
        cam = self.cams[int(frame[0, 0, 0]) // 100]       # the fixture paints the camera index into pixel 0,0
        n = self.seen[cam["id"]] = self.seen.get(cam["id"], 0) + 1
        if n < 2:
            return []
        x = self.aim(cam) * frame.shape[1]
        return [Detection(box=(x - 10, 50, x + 10, 70), conf=0.8, cls="smoke")]


def test_two_towers_seeing_one_fire_merge_and_triangulate(tmp_path):
    cams = {"rm-e": {**CAMS["rm-e"], "figlib": None, "recordings": [{"folder": "A_rm", "fire": "Test Fire", "date": "2026-01-01"}]},
            "hp-s": {"id": "hp-s", "name": "High Point (south)", "site": "High Point", "lat": 33.36302,
                     "lon": -116.83622, "azimuth_deg": 180, "hfov_deg": 90, "wx_station": None,
                     "recordings": [{"folder": "A_hp", "fire": "Test Fire", "date": "2026-01-01"}]}}
    fire = (33.2, -116.95)                    # east-southeast of Red Mountain, south-southwest of High Point
    order = list(cams.values())
    for k, (cid, folder) in enumerate((("rm-e", "A_rm"), ("hp-s", "A_hp"))):
        d = tmp_path / "figlib" / folder
        d.mkdir(parents=True)
        for i in range(8):
            img = np.full((400, 800, 3), 120, np.uint8)
            img[0, 0] = 100 * k
            (d / f"{1_700_000_000 + 60 * i + k}_{60 * i:+06d}.jpg").write_bytes(cv2.imencode(".png", img)[1].tobytes())
    FakeVLM.ctx = WILD
    app = create_map_app(cameras=cams, assets_dir=tmp_path / "assets", figlib_dir=tmp_path / "figlib",
                         detector_weights="w.pt", detector_factory=lambda w: AimedDetector(order, fire),
                         vlm_factory=FakeVLM, model_lister=lambda: ["context"],
                         forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES, clock=Clock())
    with TestClient(app) as c:
        rec = c.get("/api/recordings").json()["recordings"]
        assert rec == [{"key": "2026-01-01_Test Fire", "fire": "Test Fire", "date": "2026-01-01",
                        "cameras": ["rm-e", "hp-s"], "camera_names": ["Red Mountain (east)", "High Point (south)"]}]
        c.post("/api/watch", json={"recording": rec[0]["key"], "interval_s": 0, "batch_frames": 3,
                                   "time_mode": "recorded", "start_offset_s": None})
        [s] = wait_done(c)
        assert len(s["incident_ids"]) == 1                     # one fire, not one incident per tower
        inc = c.get(f"/api/incidents/{s['incident_ids'][0]}").json()
    assert inc["name"] == "Test Fire" and set(inc["sightings"]) == {"rm-e", "hp-s"}
    assert inc["loc_source"] == "triangulated" and haversine_km(inc["lat"], inc["lon"], *fire) < 1.5
    assert len(inc["wedges"]) == 2 and any("bearings cross" in u["text"] for u in inc["updates"])
    assert {f["camera_id"] for f in inc["frames"]} == {"rm-e", "hp-s"}


class ScriptedDetector:
    """Smoke appears from the 3rd frame and the box grows each frame."""
    def __init__(self):
        self.n = 0

    def __call__(self, frame):
        self.n += 1
        if self.n < 3:
            return []
        w = 20 + 15 * self.n
        return [Detection(box=(300, 50, 300 + w, 50 + w), conf=0.8, cls="smoke")]


def watch_env(tmp_path, n_frames=14, ctx=None):
    seq = tmp_path / "data" / "demo" / "fire_x"
    seq.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        (seq / f"{1_700_000_000 + 60 * i}_{60 * (i - 2):+06d}.jpg").write_bytes(jpeg(800, 400))
    FakeVLM.ctx = ctx or ContextResult(source_type="wildland", smoke_color="grey", attended="no",
                                       near_structures=False, near_road=False, size_estimate="small",
                                       description="Grey smoke on a slope.")
    return create_map_app(cameras=CAMS, assets_dir=tmp_path / "assets", data_dir=tmp_path / "data",
                          detector_weights="w.pt", detector_factory=lambda w: ScriptedDetector(),
                          vlm_factory=FakeVLM, model_lister=lambda: ["context"],
                          forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES, clock=Clock())


def wait_done(c, timeout=10):
    import time as _t
    end = _t.time() + timeout
    while _t.time() < end:
        s = c.get("/api/watch").json()["sessions"]
        if s and all(x["state"] != "running" for x in s):
            return s
        _t.sleep(0.05)
    raise AssertionError("watch did not finish")


def test_watch_opens_once_updates_per_batch_and_escalates_on_growth(tmp_path):
    app = watch_env(tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/recordings").json()["recordings"][0]["key"] == "fire_x"
        r = c.post("/api/watch", json={"recording": "fire_x", "interval_s": 0, "batch_frames": 4,
                                       "time_mode": "recorded", "start_offset_s": None})
        assert r.status_code == 200, r.text
        [s] = wait_done(c)
        assert s["state"] == "done" and s["frames_done"] == 14 and len(s["incident_ids"]) == 1
        # VLM: once to open, once per full batch after it (frames 5-8, 9-12), not per frame
        assert s["vlm_calls"] == 3
        inc = c.get(f"/api/incidents/{s['incident_ids'][0]}").json()
        # every frame from the gate onward is kept for the filmstrip, with boxes drawn
        assert len(inc["frames"]) == 12 and inc["frames"][0]["smoke"] and inc["frames"][0]["camera_id"] == "rm-e"
        assert c.get(f"/api/incidents/{inc['id']}/frames/11").headers["content-type"] == "image/jpeg"
        # the frame the VLM classified carries the full analysis for the large view
        opened = inc["frames"][1]["analysis"]
        assert opened["context"]["source_type"] == "wildland" and opened["severity"] == "MONITOR"
        assert opened["tokens_in"] == 300 and opened["tokens_out"] == 40 and opened["crop_image_tokens"] > 0
        assert inc["frames"][0]["dets"][0] == {"cls": "smoke", "conf": 0.8, "box": [300.0, 50.0, 365.0, 115.0], "strong": True}
        assert inc["frames"][0]["full_frame_image_tokens"] > 0 and "detect_ms" in inc["frames"][0]
        batch = [f["analysis"] for f in inc["frames"] if f.get("analysis") and "batch" in f["analysis"]]
        assert len(batch) == 2 and batch[0]["trend"] == "growing" and batch[0]["escalation"]["decision"] == "queued"
        assert c.get(f"/api/incidents/{inc['id']}/frames/12").status_code == 404
        assert "frames" not in c.get("/api/state").json()["incidents"][0]
    kinds = [u["kind"] for u in inc["updates"]]
    assert kinds[:3] == ["opened", "batch", "batch"] and kinds[-1] == "closed"
    assert inc["updates"][0]["severity"] == "MONITOR"               # small grey wildland, no trend yet
    growing = [u for u in inc["updates"] if u.get("trend") == "growing"]
    assert growing and "(raised to ALERT)" in growing[0]["text"]
    assert inc["severity"] == "ALERT" and inc["escalation"]["decision"] == "queued"
    assert inc["created_at"] == 1_700_000_000 + 60 * 3                 # recorded mode keeps the camera's time
    assert inc["status"] == "resolved"                                 # a replayed recording is archived
    assert len(inc["timeline"]) >= 10 and inc["detections"] >= 10


def test_watch_rejects_bad_requests_and_can_stop(tmp_path):
    app = watch_env(tmp_path, n_frames=40)
    with TestClient(app) as c:
        assert c.post("/api/watch", json={"recording": "nope"}).status_code == 404
        assert c.post("/api/watch", json={"recording": "fire_x", "time_mode": "x"}).status_code == 400
        sid = c.post("/api/watch", json={"recording": "fire_x", "interval_s": 0.05,
                                         "start_offset_s": None}).json()["sessions"][0]["id"]
        assert c.post("/api/watch", json={"recording": "fire_x"}).status_code == 409
        c.delete(f"/api/watch/{sid}")
        [s] = wait_done(c)
        assert s["state"] == "stopped" and s["frames_done"] < 40
        assert c.get(f"/api/watch/{sid}/frame/rm-e").headers["content-type"] == "image/jpeg"
        assert c.delete("/api/watch/nope").status_code == 404


def test_photos_use_the_photo_detector_and_tower_frames_the_tower_one(tmp_path):
    used = []

    def factory(weights):
        used.append(Path(weights).name)
        return FakeDetector()
    tower_w, photo_w = tmp_path / "tower_yolo.pt", tmp_path / "smoke_yolo.pt"
    tower_w.write_bytes(b"x")
    photo_w.write_bytes(b"x")
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "a", detector_weights=str(tower_w),
                         photo_detector_weights=str(photo_w), detector_factory=factory, vlm_factory=FakeVLM,
                         model_lister=lambda: ["context"], forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES)
    with TestClient(app) as c:
        r1 = post(c, camera_id="rm-e", image=jpeg()).json()
        r2 = post(c, lat="33.0", lon="-116.9").json()
        f1 = c.get(f"/api/incidents/{r1['incident']['id']}").json()["frames"][r1["frame_n"]]
        f2 = c.get(f"/api/incidents/{r2['incident']['id']}").json()["frames"][r2["frame_n"]]
    # an upload is a photo even with a camera chosen; a camera frame from a recording uses the tower model
    assert used == ["smoke_yolo.pt"] and f1["detector"] == f2["detector"] == "smoke_yolo"


def test_detector_reloads_when_weights_change_on_disk(tmp_path):
    import os
    loads = []
    w = tmp_path / "tower_yolo.pt"
    w.write_bytes(b"v1")

    def factory(weights):
        loads.append(Path(weights).read_bytes())
        return FakeDetector()
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "a", detector_weights=str(w), photo_detector_weights=None,
                         detector_factory=factory, vlm_factory=FakeVLM, model_lister=lambda: ["context"],
                         forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES)
    with TestClient(app) as c:
        post(c, camera_id="rm-e")
        post(c, camera_id="rm-e")
        w.write_bytes(b"v2")
        os.utime(w, (w.stat().st_atime, w.stat().st_mtime + 10))
        post(c, camera_id="rm-e")
    assert loads == [b"v1", b"v2"]


def test_a_tower_that_first_fired_on_haze_is_merged_once_its_bearing_moves_onto_the_fire(tmp_path):
    """The Junction Fire case: tower 1 opens early on haze in the wrong direction; tower 2 then sees the
    real plume and, not crossing tower 1's stale line, opens its own incident. When tower 1's next batch
    moves its bearing onto the plume, the two must become one triangulated incident."""
    cams = {"rm-e": {**CAMS["rm-e"], "figlib": None, "recordings": [{"folder": "B_rm", "fire": "Haze Fire", "date": "2026-01-02"}]},
            "hp-s": {"id": "hp-s", "name": "High Point (south)", "site": "High Point", "lat": 33.36302,
                     "lon": -116.83622, "azimuth_deg": 180, "hfov_deg": 90, "wx_station": None,
                     "recordings": [{"folder": "B_hp", "fire": "Haze Fire", "date": "2026-01-02"}]}}
    fire = (33.2, -116.95)
    order = list(cams.values())

    class HazeThenFire(AimedDetector):
        def __call__(self, frame):
            cam = self.cams[int(frame[0, 0, 0]) // 100]
            n = self.seen[cam["id"]] = self.seen.get(cam["id"], 0) + 1
            if cam["id"] == "rm-e":
                x = 0.1 * frame.shape[1] if n <= 8 else self.aim(cam) * frame.shape[1]   # haze far left first
            else:
                if n < 5:
                    return []
                x = self.aim(cam) * frame.shape[1]
            return [Detection(box=(x - 10, 50, x + 10, 70), conf=0.8, cls="smoke")]

    for k, folder in enumerate(("B_rm", "B_hp")):
        d = tmp_path / "figlib" / folder
        d.mkdir(parents=True)
        for i in range(16):
            img = np.full((400, 800, 3), 120, np.uint8)
            img[0, 0] = 100 * k
            (d / f"{1_700_000_000 + 60 * i + k}_{60 * i:+06d}.jpg").write_bytes(cv2.imencode(".png", img)[1].tobytes())
    FakeVLM.ctx = WILD
    app = create_map_app(cameras=cams, assets_dir=tmp_path / "assets", figlib_dir=tmp_path / "figlib",
                         detector_weights="w.pt", photo_detector_weights=None,
                         detector_factory=lambda w: HazeThenFire(order, fire),
                         vlm_factory=FakeVLM, model_lister=lambda: ["context"],
                         forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES, clock=Clock())
    with TestClient(app) as c:
        key = c.get("/api/recordings").json()["recordings"][0]["key"]
        c.post("/api/watch", json={"recording": key, "interval_s": 0, "batch_frames": 3,
                                   "time_mode": "recorded", "start_offset_s": None})
        [s] = wait_done(c)
        incs = c.get("/api/state").json()["incidents"]
        assert len(incs) == 1 and s["incident_ids"] == [incs[0]["id"]]
        inc = c.get(f"/api/incidents/{incs[0]['id']}").json()
    assert set(inc["sightings"]) == {"rm-e", "hp-s"} and inc["loc_source"] == "triangulated"
    assert haversine_km(inc["lat"], inc["lon"], *fire) < 1.5
    assert any(u["kind"] == "merged" for u in inc["updates"])
    assert {f["camera_id"] for f in inc["frames"]} == {"rm-e", "hp-s"}
    for f in inc["frames"]:                                       # every renumbered frame file is still there
        assert (tmp_path / "assets" / "state" / "frames" / inc["id"] / f"{f['n']}.jpg").exists()


def test_demo_scenarios_run_the_real_pipeline_on_mock_inputs(tmp_path):
    data = tmp_path / "data"
    (data / "imgs").mkdir(parents=True)
    (data / "imgs" / "fire.jpg").write_bytes(jpeg())
    (data / "imgs" / "fog.jpg").write_bytes(jpeg(value=200))
    cfg = tmp_path / "demo.json"
    cfg.write_text(json.dumps([
        {"id": "a", "kind": "photo", "title": "Fire", "image": "imgs/fire.jpg", "lat": 33.0, "lon": -116.9,
         "name": "Demo Fire", "expect": "ALERT", "expect_severity": "ALERT"},
        {"id": "b", "kind": "photo", "title": "Fog", "image": "imgs/fog.jpg", "lat": 33.1, "lon": -116.9,
         "expect": "IGNORE", "expect_severity": "IGNORE"},
        {"id": "c", "kind": "action", "action": "restore_link", "title": "Link", "expect": "sent"}]))

    class ByBrightness:
        def __call__(self, frame):      # bright image = fog: nothing above the gate
            return [] if frame.mean() > 150 else [Detection(box=(10, 10, 60, 60), conf=0.8, cls="smoke")]
    FakeVLM.ctx = WILD
    app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "a", data_dir=data, detector_weights="w.pt",
                         photo_detector_weights=None, detector_factory=lambda w: ByBrightness(), vlm_factory=FakeVLM,
                         model_lister=lambda: ["context"], forecast_fn=lambda la, lo: FORECAST, counties=COUNTIES,
                         demo_scenarios=cfg, start_online=True)
    with TestClient(app) as c:
        post(c, camera_id="rm-e")                                  # leftover incident: cleared by the load
        c.post("/api/demo/load")
        import time as _t
        for _ in range(100):
            d = c.get("/api/demo").json()
            if d["state"] != "running":
                break
            _t.sleep(0.05)
        a, b, link = d["scenarios"]
        state = c.get("/api/state").json()
    assert d["state"] == "done" and state["online"] is False
    assert a["result"]["severity"] == "ALERT" and a["result"]["escalation"] == "queued" and a["result"]["thumb_b64"] is None
    assert b["result"]["severity"] == "IGNORE" and b["result"]["incident_id"] is None and b["result"]["thumb_b64"]
    assert link["result"]["status"] == "ready"
    [inc] = state["incidents"]
    assert inc["name"] == "Demo Fire" and inc["demo"]["id"] == "a" and inc["key_frame"] == 0


def test_default_detectors_run_at_their_training_size(tmp_path, monkeypatch):
    """Tower frames go through the joint model at 960 px, photos through the D-Fire model at 640."""
    import sentinel.detector as det_mod
    made = []

    class FakeYolo(FakeDetector):
        def __init__(self, weights, conf=0.25, imgsz=640, classes=None, model=None):
            super().__init__()
            made.append((Path(weights).name, imgsz))
    monkeypatch.setattr(det_mod, "YoloDetector", FakeYolo)
    tower_w, photo_w = tmp_path / "joint_yolo.pt", tmp_path / "smoke_yolo.pt"
    tower_w.write_bytes(b"x")
    photo_w.write_bytes(b"x")
    seq = tmp_path / "data" / "demo" / "fire_x"
    seq.mkdir(parents=True)
    (seq / "1_+00120.jpg").write_bytes(jpeg())

    def run(**kw):
        made.clear()
        app = create_map_app(cameras=CAMS, assets_dir=tmp_path / "a", data_dir=tmp_path / "data",
                             state_dir=tmp_path / "s", detector_weights=str(tower_w),
                             photo_detector_weights=str(photo_w), vlm_factory=FakeVLM,
                             model_lister=lambda: ["context"], forecast_fn=lambda la, lo: FORECAST,
                             counties=COUNTIES, **kw)
        with TestClient(app) as c:
            [sid] = [s["id"] for s in c.get("/api/samples").json()["samples"] if s["camera_id"] == "rm-e"]
            assert post(c, sample_id=sid).status_code == 200
            assert post(c, camera_id="rm-e", image=jpeg()).status_code == 200
        return made[:]
    assert run() == [("joint_yolo.pt", 960), ("smoke_yolo.pt", 640)]
    assert run(detector_imgsz=1280, photo_detector_imgsz=800) == [("joint_yolo.pt", 1280), ("smoke_yolo.pt", 800)]
    assert (tmp_path / "s" / "incidents.db").exists() and not (tmp_path / "a" / "state").exists()


def test_main_defaults_and_overrides(monkeypatch, tmp_path):
    import sys
    import types
    import sentinel.incident_map as im
    seen = {}
    monkeypatch.setattr(im, "create_map_app", lambda **kw: seen.update(kw) or "app")
    monkeypatch.setattr(im, "load_cameras", lambda path: {})
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda app, **kw: seen.update(ran=app)))
    assets = tmp_path / "assets"
    im.main(["--assets", str(assets), "--no-sensor"])
    assert seen["ran"] == "app" and Path(seen["state_dir"]) == assets / "state"
    assert Path(seen["detector_weights"]).name == "joint_yolo.pt" and seen["detector_imgsz"] == 960
    assert Path(seen["photo_detector_weights"]).name == "smoke_yolo.pt" and seen["photo_detector_imgsz"] == 640
    seen.clear()
    im.main(["--assets", str(assets), "--no-sensor", "--state-dir", str(tmp_path / "state-core"),
             "--detector-weights", "models/smoke_yolo.pt", "--photo-detector-imgsz", "800"])
    assert Path(seen["state_dir"]) == tmp_path / "state-core"
    assert seen["detector_imgsz"] == 640 and seen["photo_detector_imgsz"] == 800
    seen.clear()
    im.main(["--no-sensor", "--detector-weights", "models/smoke_yolo.pt", "--detector-imgsz", "960"])
    assert seen["detector_imgsz"] == 960

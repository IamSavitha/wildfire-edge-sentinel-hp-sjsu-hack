import json

import cv2
import numpy as np
import pytest

from scripts.bench_cloud import bench_cloud, classify_clip, encode_frame, parse_args, simulate, usd
from sentinel.cloud_cache import CachedVLM, ResponseCache
from sentinel.netprofile import PROFILES, Outages, transfer_s
from sentinel.schema import ContextResult

FIRE = ContextResult(source_type="wildland", smoke_color="grey", attended="no", near_structures=True,
                     near_road=False, size_estimate="small", description="Smoke on a ridge.")   # ALERT
WATCH = FIRE.model_copy(update={"near_structures": False})                                       # MONITOR
FOG = FIRE.model_copy(update={"source_type": "fog_dust_cloud"})                                  # IGNORE


class ByBrightness:
    """Fake cloud VLM: answers by the frame's pixel value; value 0 -> request error."""

    model, response_format = "qwen7b", "json_object"

    def __init__(self, table):
        self.table, self.calls = table, 0
        self.last_usage, self.last_error = {}, None

    def classify(self, jpeg):
        self.calls += 1
        v = int(np.median(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)))
        key = min(self.table, key=lambda k: abs(k - v))
        if self.table[key] == "ERR":
            self.last_usage, self.last_error = {"calls": 0}, "APITimeoutError: slow"
            return None, 0
        self.last_usage, self.last_error = {"prompt_tokens": 1000, "completion_tokens": 50, "calls": 1}, None
        return self.table[key], 1050


class Clock:
    """Each classify() spans 0.5 s of measured latency."""

    def __init__(self):
        self.t, self.reads = 0.0, 0

    def __call__(self):
        self.reads += 1
        if self.reads % 2 == 0:
            self.t += 0.5
        return self.t


def write_clip(root, name, values, size=(64, 96)):
    d = root / name
    d.mkdir()
    for i, v in enumerate(values):
        cv2.imwrite(str(d / f"{i:03d}.jpg"), np.full((*size, 3), v, np.uint8))
    return str(d)


def cached(tmp_path, table):
    fake = ByBrightness(table)
    return CachedVLM(fake, ResponseCache(tmp_path / "cache.jsonl"), clock=Clock()), fake


def test_encode_frame_caps_the_longest_side():
    jpeg = encode_frame(np.zeros((1080, 1920, 3), np.uint8), 1280)
    h, w = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    assert (w, h) == (1280, 720)


def test_classify_clip_sends_every_stride_frame_with_measured_latency(tmp_path):
    clip = write_clip(tmp_path, "c", [200, 200, 30, 30, 200])
    vlm, fake = cached(tmp_path, {200: FIRE, 30: FOG})
    rows = classify_clip(clip, vlm, fps=2.0, stride=2)
    assert [r["i"] for r in rows] == [0, 2, 4] and [r["t"] for r in rows] == [0.0, 1.0, 2.0]
    assert [r["severity"] for r in rows] == ["ALERT", "IGNORE", "ALERT"]
    assert fake.calls == 2  # frames 0 and 4 are identical -> the second is a cache hit
    assert [r["cached"] for r in rows] == [False, False, True]
    assert all(r["latency_ms"] == pytest.approx(500) for r in rows)
    assert rows[0]["prompt_tokens"] == 1000 and rows[0]["bytes"] > 0


def F(t, sev="ALERT", nbytes=125_000, lat=500.0):
    return {"t": t, "bytes": nbytes, "severity": sev, "latency_ms": lat, "prompt_tokens": 1000,
            "completion_tokens": 50}


def test_simulate_on_fiber_adds_transfer_and_latency():
    s = simulate([F(0.0, "IGNORE"), F(0.5)], PROFILES["fiber"])
    up = transfer_s(125_000, PROFILES["fiber"])  # 0.04 s
    assert s["first_alert_s"] == pytest.approx(0.5 + up + 0.5)
    assert s["decided"] == 2 and s["bytes_up"] == 250_000 and s["prompt_tokens"] == 2000
    assert s["max_backlog_s"] == 0


def test_simulate_slow_uplink_builds_a_backlog():
    frames = [F(i * 0.5, "IGNORE") for i in range(4)] + [F(2.0)]
    s = simulate(frames, PROFILES["rural_cellular"])
    up = transfer_s(125_000, PROFILES["rural_cellular"])  # ~1.17 s > 0.5 s frame interval
    assert s["first_alert_s"] == pytest.approx(5 * up + 0.5)
    assert s["max_backlog_s"] == pytest.approx(4 * up - 2.0)


def test_simulate_outage_drop_vs_buffer():
    frames = [F(0.0), F(1.0, "IGNORE"), F(12.0)]
    o = Outages([(0, 10)])
    drop = simulate(frames, PROFILES["fiber"], o, "drop")
    up = transfer_s(125_000, PROFILES["fiber"])
    assert drop["dropped"] == 2 and drop["decided"] == 1 and drop["captured_during_outage"] == 2
    assert drop["first_alert_s"] == pytest.approx(12 + up + 0.5)
    buf = simulate(frames, PROFILES["fiber"], o, "buffer")
    assert buf["dropped"] == 0 and buf["decided"] == 3
    assert buf["first_alert_s"] == pytest.approx(10 + up + 0.5)  # backlog drains from t=10
    assert buf["max_backlog_s"] == pytest.approx(10)
    assert drop["decisions_during_outage"] == buf["decisions_during_outage"] == 0


def test_simulate_outage_profile_and_failed_requests():
    assert simulate([F(0.0)], PROFILES["outage"])["dropped"] == 1
    s = simulate([F(0.0, None, lat=None), F(1.0, "MONITOR")], PROFILES["fiber"], alert_on="monitor")
    assert s["failed"] == 1 and s["decided"] == 1 and s["first_alert_s"] is not None
    assert simulate([F(1.0, "MONITOR")], PROFILES["fiber"])["first_alert_s"] is None


def test_usd_uses_in_out_and_gb_prices():
    assert usd(2e6, 1e6, 1e9, {"usd_per_mtok_in": 0.2, "usd_per_mtok_out": 0.6, "usd_per_gb": 0.1}) == pytest.approx(1.1)
    assert usd(1e6, 1e6, 0, {}) == 0


def test_bench_cloud_scores_clips_per_profile(tmp_path):
    fire = write_clip(tmp_path, "fire", [30, 30, 200, 200])
    fog = write_clip(tmp_path, "fog", [30, 30, 30, 30])
    broken = write_clip(tmp_path, "broken", [0, 0])
    rows = [{"path": fire, "label": "alert"}, {"path": fog, "label": "no_alert"},
            {"path": broken, "label": "alert"}]
    vlm, _ = cached(tmp_path, {200: FIRE, 30: FOG, 0: "ERR"})
    vlm.max_consecutive_errors = None
    out = bench_cloud("t", rows, vlm, fps=2.0, profiles=[PROFILES["fiber"], PROFILES["satellite_geo"]],
                      prices={"usd_per_mtok_in": 1.0, "usd_per_mtok_out": 1.0})
    assert out["tp"] == 1 and out["fn"] == 1 and out["fp"] == 0
    assert out["frames_sent"] == 10 and out["frames_answered"] == 8
    assert out["n_request_errors"] == 2 and out["n_fresh"] == 2 and out["n_cached"] == 6
    assert out["cloud_latency_ms_p50"] == pytest.approx(500)  # fresh calls only
    assert out["billed_prompt_tokens"] == 2000
    fiber, sat = out["profiles"]["fiber"], out["profiles"]["satellite_geo"]
    assert fiber["recall"] == 0.5 and fiber["time_to_first_alert_s_p50"] < sat["time_to_first_alert_s_p50"]
    assert fiber["usd"] == pytest.approx(8 * 1050 / 1e6)
    assert fiber["frames_failed"] == 2 and fiber["bytes_per_1000_frames"] > 0
    assert out["clips"][0]["severities"]["ALERT"] == 2 and out["clips"][2]["severities"]["failed"] == 2
    json.dumps(out, allow_nan=False)  # strict JSON


def test_parse_args_validates_profiles_and_outages():
    a = parse_args(["--outage", "0,60", "--outage", "100,160", "--profiles", "fiber,lte"])
    assert a.outage_windows.windows == [(0, 60), (100, 160)] and [p.name for p in a.profile_list] == ["fiber", "lte"]
    for bad in (["--profiles", "dialup"], ["--outage", "5"], ["--frame-stride", "0"]):
        with pytest.raises(SystemExit):
            parse_args(bad)


def test_main_writes_results_without_leaking_the_key(tmp_path, monkeypatch, capsys):
    import scripts.bench_cloud as bc

    secret = "sk-test-SECRET-bench-42"
    monkeypatch.chdir(tmp_path)
    clip = write_clip(tmp_path, "fire", [200, 200])
    (tmp_path / "clips.csv").write_text(f"path,label\n{clip},alert\n")
    for k, v in {"CLOUD_VLM_BASE_URL": "https://api.example.com/v1", "CLOUD_VLM_MODEL": "qwen7b",
                 "CLOUD_VLM_API_KEY": secret}.items():
        monkeypatch.setenv(k, v)
    real = bc.cloud_vlm_from_env

    def fake_env(**kw):  # real env parsing; answers come from the fake instead of the network
        return _Proxy(real(**kw), ByBrightness({200: FIRE}))
    monkeypatch.setattr(bc, "cloud_vlm_from_env", fake_env)
    bc.main(["--clips", "clips.csv", "--profiles", "fiber", "--outage", "0,0.4",
             "--outage-policy", "buffer", "--settings", "none.json"])
    text = (tmp_path / "results" / "bench_cloud_cloud_qwen7b.json").read_text()
    out = json.loads(text)
    assert out["provider_host"] == "api.example.com" and out["recall"] == 1.0
    assert out["outages"] == [[0.0, 0.4]] and out["outage_policy"] == "buffer"
    assert secret not in text + capsys.readouterr().out
    with pytest.raises(SystemExit):  # results are never silently overwritten
        bc.main(["--clips", "clips.csv", "--settings", "none.json"])


class _Proxy:
    """A configured ContextVLM's identity (model, host, format) with a fake's answers."""

    def __init__(self, vlm, fake):
        self._vlm, self._fake = vlm, fake
        self.model, self.host, self.response_format = vlm.model, vlm.host, vlm.response_format

    def classify(self, jpeg):
        return self._fake.classify(jpeg)

    @property
    def last_usage(self):
        return self._fake.last_usage

    @property
    def last_error(self):
        return self._fake.last_error

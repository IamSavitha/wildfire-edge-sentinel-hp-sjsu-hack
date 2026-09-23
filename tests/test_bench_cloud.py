import json

import cv2
import numpy as np
import pytest

from scripts.bench_cloud import (bench_cloud, classify_clip, encode_frame, first_alert, frame_severity, parse_args,
                                 simulate, usd)
from sentinel.cloud_cache import CachedVLM, ResponseCache
from sentinel.netprofile import PROFILES, Outages, rtt_s, serialize_s
from sentinel.schema import ContextResult

FIRE = ContextResult(source_type="wildland", smoke_color="grey", attended="no", near_structures=True,
                     near_road=False, size_estimate="small", description="Smoke on a ridge.")   # ALERT
WATCH = FIRE.model_copy(update={"near_structures": False})                                       # MONITOR
FOG = FIRE.model_copy(update={"source_type": "fog_dust_cloud", "smoke_color": "none"})          # IGNORE


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


def C(**kw):
    base = dict(source_type="wildland", smoke_color="grey", attended="no", near_structures=True,
                near_road=False, size_estimate="small", description="d")
    return {**base, **kw}


SEV = {"ALERT": C(), "MONITOR": C(near_structures=False), "IGNORE": C(smoke_color="none",
                                                                    source_type="fog_dust_cloud")}


def F(t, sev="ALERT", nbytes=125_000, lat=500.0, ctx=None):
    c = ctx if ctx is not None else SEV.get(sev)
    return {"t": t, "i": int(round(t * 2)), "bytes": nbytes, "severity": sev, "ctx": c, "latency_ms": lat,
            "prompt_tokens": 1000, "completion_tokens": 50}


FIBER, RURAL = PROFILES["fiber"], PROFILES["rural_cellular"]


def test_no_smoke_answer_is_ignore():
    assert frame_severity(C(smoke_color="none")) == "IGNORE"
    assert frame_severity(C()) == "ALERT" and frame_severity(None) is None
    assert frame_severity(C(near_structures=False), "growing") == "ALERT"


def test_simulate_on_fiber_link_busy_only_while_serializing():
    s = simulate([F(0.0, "IGNORE"), F(0.5)], FIBER, policy="fifo")
    assert s["first_alert_s"] == pytest.approx(0.5 + serialize_s(125_000, FIBER) + rtt_s(FIBER) + 0.5)
    assert s["decided"] == 2 and s["bytes_up"] == 250_000 and s["prompt_tokens"] == 2000
    assert s["max_backlog_s"] == 0


def test_net_baseline_is_subtracted_and_floored_at_zero():
    s = simulate([F(0.0, lat=500.0)], FIBER, net_baseline_ms=200.0)
    assert s["first_alert_s"] == pytest.approx(serialize_s(125_000, FIBER) + rtt_s(FIBER) + 0.3)
    s = simulate([F(0.0, lat=100.0)], FIBER, net_baseline_ms=200.0)
    assert s["first_alert_s"] == pytest.approx(serialize_s(125_000, FIBER) + rtt_s(FIBER))


def test_fifo_on_a_slow_uplink_builds_a_backlog():
    frames = [F(i * 0.5, "IGNORE") for i in range(4)] + [F(2.0)]
    s = simulate(frames, RURAL, policy="fifo")
    up = serialize_s(125_000, RURAL)  # ~1.02 s > 0.5 s frame interval
    assert s["first_alert_s"] == pytest.approx(5 * up + rtt_s(RURAL) + 0.5)
    assert s["max_backlog_s"] == pytest.approx(4 * up - 2.0)


def test_latest_skips_stale_frames_and_stays_current():
    frames = [F(i * 0.5, "IGNORE") for i in range(4)] + [F(2.0)]
    s = simulate(frames, RURAL, policy="latest")
    up = serialize_s(125_000, RURAL)
    # t=0 sent; at ~1.02 frames 0.5 and 1.0 wait -> send 1.0, skip 0.5; at ~2.04 send 2.0 (the alert)
    assert s["sent"] == 3 and s["skipped"] == 2
    assert s["first_alert_s"] == pytest.approx(2 * up + up + rtt_s(RURAL) + 0.5)
    assert s["first_alert_s"] < simulate(frames, RURAL, policy="fifo")["first_alert_s"]


def test_outage_policies():
    frames = [F(0.0), F(1.0, "IGNORE"), F(12.0)]
    o = Outages([(0, 10)])
    up, rtt = serialize_s(125_000, FIBER), rtt_s(FIBER)
    drop = simulate(frames, FIBER, o, "drop")
    assert drop["dropped"] == 2 and drop["decided"] == 1 and drop["captured_during_outage"] == 2
    assert drop["first_alert_s"] == pytest.approx(12 + up + rtt + 0.5)
    fifo = simulate(frames, FIBER, o, "fifo")
    assert fifo["decided"] == 3 and fifo["first_alert_s"] == pytest.approx(10 + up + rtt + 0.5)
    assert fifo["max_backlog_s"] == pytest.approx(10)
    newest = simulate(frames, FIBER, o, "buffer_newest")
    assert newest["decided"] == 3 and newest["first_alert_s"] == pytest.approx(10 + 2 * up + rtt + 0.5)
    latest = simulate(frames, FIBER, o, "latest")
    assert latest["skipped"] == 1 and latest["decided"] == 2  # the stale t=0 frame is never sent
    assert drop["decisions_during_outage"] == fifo["decisions_during_outage"] == 0


def test_simulate_outage_profile_and_failed_requests():
    assert simulate([F(0.0)], PROFILES["outage"])["dropped"] == 1
    s = simulate([F(0.0, None, lat=None), F(1.0, "MONITOR")], FIBER, policy="fifo", alert_on="monitor")
    assert s["failed"] == 1 and s["decided"] == 1 and s["first_alert_s"] is not None
    assert simulate([F(1.0, "MONITOR")], FIBER)["first_alert_s"] is None


def test_temporal_rule_needs_k_consecutive_positive_frames():
    frames = [F(0.0), F(0.5, "IGNORE"), F(1.0), F(1.5), F(2.0)]
    dec = [{**f, "decided_at": f["t"] + 1} for f in frames]
    assert first_alert(dec) == pytest.approx(1.0)
    assert first_alert(dec, temporal={"persist": 3, "recheck_s": 5}) == pytest.approx(3.0)
    assert first_alert(dec[:4], temporal={"persist": 3, "recheck_s": 5}) is None


def test_temporal_trend_from_size_rank_escalates_monitor():
    small, big = C(near_structures=False), C(near_structures=False, size_estimate="medium")
    frames = [F(0.0, "MONITOR", ctx=small), F(5.0, "MONITOR", ctx=big)]
    dec = [{**f, "decided_at": f["t"]} for f in frames]
    assert first_alert(dec) is None  # per-frame: both MONITOR
    assert first_alert(dec, temporal={"persist": 1, "recheck_s": 5}) == pytest.approx(5.0)  # growing -> ALERT


def test_usd_uses_in_out_and_gb_prices():
    assert usd(2e6, 1e6, 1e9, {"usd_per_mtok_in": 0.2, "usd_per_mtok_out": 0.6, "usd_per_gb": 0.1}) == pytest.approx(1.1)
    assert usd(1e6, 1e6, 0, {}) == 0


def test_bench_cloud_scores_rules_cadence_and_profiles(tmp_path):
    fire = write_clip(tmp_path, "fire", [30, 30, 200, 200])
    fog = write_clip(tmp_path, "fog", [30, 30, 30, 30])
    broken = write_clip(tmp_path, "broken", [0, 0])
    rows = [{"path": fire, "label": "alert"}, {"path": fog, "label": "no_alert"},
            {"path": broken, "label": "alert"}]
    vlm, fake = cached(tmp_path, {200: FIRE, 30: FOG, 0: "ERR"})
    vlm.max_consecutive_errors = None
    out = bench_cloud("t", rows, vlm, fps=2.0, profiles=[FIBER, PROFILES["satellite_geo"]],
                      prices={"usd_per_mtok_in": 1.0, "usd_per_mtok_out": 1.0}, net_baseline_ms=100.0,
                      persist=2, cadence_stride=2)
    assert out["tp"] == 1 and out["fn"] == 1 and out["fp"] == 0
    assert out["frames_sent"] == 10 and out["frames_answered"] == 8
    assert out["n_request_errors"] == 2 and out["n_fresh"] == 2 and out["n_cached"] == 6
    assert out["cloud_latency_ms_p50"] == pytest.approx(500)  # fresh calls only
    assert out["cloud_inference_ms_p50"] == pytest.approx(400) and out["net_baseline_ms"] == 100.0
    assert out["billed_prompt_tokens"] == 2000
    fiber, sat = out["profiles"]["fiber"], out["profiles"]["satellite_geo"]
    assert fiber["recall"] == 0.5 and fiber["time_to_first_alert_s_p50"] < sat["time_to_first_alert_s_p50"]
    assert fiber["frames_failed"] == 2 and fiber["bytes_per_1000_frames"] > 0
    assert fiber["usd_per_1000_captured_frames"] is not None
    # temporal: the fire clip has 2 consecutive ALERT frames -> still alerts with persist 2
    assert out["temporal"]["recall"] == 0.5 and out["temporal"]["profiles"]["fiber"]["recall"] == 0.5
    # cadence 2: frames 0 and 2 of each clip; frame 2 of the fire clip is ALERT
    assert out["cadence"]["stride"] == 2 and out["cadence"]["per_frame"]["recall"] == 0.5
    assert out["cadence"]["temporal"]["recall"] == 0.0  # only one positive frame at that cadence
    # per camera frame, the sparser cadence sends half the bytes (10 camera frames either way)
    full, cad = out["profiles"]["fiber"], out["cadence"]["per_frame"]["profiles"]["fiber"]
    assert full["camera_frames"] == cad["camera_frames"] == 10
    assert cad["bytes_per_1000_captured_frames"] == pytest.approx(full["bytes_per_1000_captured_frames"] / 2, rel=0.05)
    assert out["clips"][0]["severities"]["ALERT"] == 2 and out["clips"][2]["severities"]["failed"] == 2
    json.dumps(out, allow_nan=False)  # strict JSON


def test_bench_cloud_skips_cadence_not_divisible_by_stride(tmp_path):
    vlm, _ = cached(tmp_path, {200: FIRE})
    out = bench_cloud("t", [{"path": write_clip(tmp_path, "f", [200] * 3), "label": "alert"}], vlm, fps=2.0,
                      profiles=[FIBER], stride=3, cadence_stride=20)
    assert out["cadence"] is None


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
             "--policy", "fifo", "--settings", "none.json", "--net-baseline-ms", "0"])
    text = (tmp_path / "results" / "bench_cloud_cloud_qwen7b.json").read_text()
    out = json.loads(text)
    assert out["provider_host"] == "api.example.com" and out["recall"] == 1.0
    assert out["outages"] == [[0.0, 0.4]] and out["policy"] == "fifo" and out["prompt"] == "CLOUD_FULL_FRAME_PROMPT"
    assert secret not in text + capsys.readouterr().out
    with pytest.raises(SystemExit):  # results are never silently overwritten
        bc.main(["--clips", "clips.csv", "--settings", "none.json", "--net-baseline-ms", "0"])


class _Proxy:
    """A configured ContextVLM's identity (model, host, format) with a fake's answers."""

    def __init__(self, vlm, fake):
        self._vlm, self._fake = vlm, fake

    def __getattr__(self, name):  # model, host, response_format, fingerprint, parse_retries...
        return getattr(self._vlm, name)

    def classify(self, jpeg):
        return self._fake.classify(jpeg)

    @property
    def last_usage(self):
        return self._fake.last_usage

    @property
    def last_error(self):
        return self._fake.last_error

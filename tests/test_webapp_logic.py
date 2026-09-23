import base64
import json

import cv2
import numpy as np
import pytest

from sentinel.config import Tower
from sentinel.schema import ContextResult, Detection
from sentinel.webapp_logic import (Session, escalate, ground_truth_score, load_benchmarks,
                                   qwen_image_tokens, run_pass)

TOWER = Tower("t1", "Test Lookout", 37.1, -121.9, "", temp_c=30.0)
WILD = ContextResult(source_type="wildland", smoke_color="black", attended="no", near_structures=False,
                     near_road=False, size_estimate="small", description="Dark smoke on a ridge.")
CAMP = ContextResult(source_type="campfire", smoke_color="white", attended="yes", near_structures=False,
                     near_road=False, size_estimate="small", description="Campfire with people.")
FOG = ContextResult(source_type="fog_dust_cloud", smoke_color="white", attended="unclear",
                    near_structures=False, near_road=False, size_estimate="large", description="Fog.")


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        return self.dets


class FakeVLM:
    def __init__(self, ctx, tokens=400, usage=None):
        self.ctx, self.tokens, self.usage = ctx, tokens, usage
        self.jpegs = []

    def classify(self, jpeg):
        self.jpegs.append(jpeg)
        if self.usage is not None:
            self.last_usage = self.usage
        return self.ctx, self.tokens


def ticker(step_s=0.01):
    t = [0.0]

    def timer():
        t[0] += step_s
        return t[0]
    return timer


def frame(w=640, h=480):
    return np.full((h, w, 3), 90, np.uint8)


def jpeg_size(b64_or_bytes):
    data = base64.b64decode(b64_or_bytes) if isinstance(b64_or_bytes, str) else b64_or_bytes
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[1], img.shape[0]


# ---- qwen_image_tokens ----

def test_qwen_tokens_crop_448():
    assert qwen_image_tokens(448, 448) == 256


def test_qwen_tokens_full_hd_frame():
    # 1080 -> 1092 (39 patches), 1920 -> 1932 (69 patches); under max_pixels so no downscale
    assert qwen_image_tokens(1920, 1080) == 39 * 69


def test_qwen_tokens_scales_down_above_max_pixels():
    # 4000x3000 = 12M px < 12845056 -> rounded only; 8000x6000 must be scaled to fit the cap
    assert qwen_image_tokens(4000, 3000) == round(4000 / 28) * round(3000 / 28)
    # beta = sqrt(48e6 / 12845056); floor to 28-multiples: 110 x 147 patches
    assert qwen_image_tokens(8000, 6000) == 110 * 147


def test_qwen_tokens_small_max_pixels_cap():
    assert qwen_image_tokens(1920, 1080, max_pixels=448 * 448) * 784 <= 448 * 448


def test_qwen_tokens_scales_up_below_min_pixels():
    # 20x20 is below min_pixels (3136 = 56*56): upscaled to at least 2x2 patches
    assert qwen_image_tokens(20, 20) == 4
    assert qwen_image_tokens(10, 10) >= 4


# ---- run_pass ----

def test_cascade_skips_vlm_without_strong_detection():
    det = FakeDetector([Detection("smoke", 0.2, (10, 10, 60, 60))])
    vlm = FakeVLM(WILD)
    r = run_pass(frame(), det, vlm, TOWER, now=1_700_000_000.0, timer=ticker())
    assert det.calls == 1 and vlm.jpegs == []
    assert r["vlm_called"] is False and r["vlm_status"] == "skipped_no_detection"
    assert r["severity"] == "IGNORE" and r["tokens"] == 0 and r["crop_image_tokens"] == 0
    assert r["best"] is None and len(r["detections"]) == 1
    assert r["detections"][0]["above_gate"] is False
    assert r["full_frame_image_tokens"] == qwen_image_tokens(640, 480)
    assert r["report"] is None and r["report_text"] is None
    assert r["detect_ms"] == pytest.approx(10.0)
    assert r["width"] == 640 and r["height"] == 480


def test_strong_detection_calls_vlm_on_crop_and_reports():
    det = FakeDetector([Detection("smoke", 0.3, (0, 0, 20, 20)),
                        Detection("smoke", 0.9, (100, 100, 300, 250))])
    vlm = FakeVLM(WILD, tokens=420, usage={"prompt_tokens": 380, "completion_tokens": 40, "calls": 1})
    r = run_pass(frame(1280, 960), det, vlm, TOWER, now=1_700_000_000.0, timer=ticker(), event_id="ev1")
    assert r["vlm_called"] is True and r["vlm_status"] == "ok" and r["vlm_input"] == "crop"
    assert r["best"]["conf"] == 0.9 and r["best"]["box"] == [100, 100, 300, 250]
    w, h = jpeg_size(vlm.jpegs[0])
    assert max(w, h) <= 448
    assert r["crop_image_tokens"] == qwen_image_tokens(w, h)
    assert r["full_frame_image_tokens"] == qwen_image_tokens(1280, 960)
    assert r["tokens"] == 420 and r["tokens_in"] == 380 and r["tokens_out"] == 40
    assert r["context"]["source_type"] == "wildland"
    assert r["severity"] == "ALERT"  # black smoke from a wildland source
    assert r["report"]["event_id"] == "ev1" and r["report"]["severity"] == "ALERT"
    assert r["report_text"].startswith("[ALERT] Wildland")
    assert r["vlm_ms"] == pytest.approx(10.0)
    tw, th = jpeg_size(r["thumbnail_b64"])
    assert max(tw, th) <= 256


def test_force_vlm_sends_full_frame_when_nothing_detected():
    det = FakeDetector([])
    vlm = FakeVLM(FOG, tokens=900)
    r = run_pass(frame(1920, 1080), det, vlm, TOWER, force_vlm=True, now=0.0, timer=ticker())
    assert r["vlm_called"] is True and r["vlm_input"] == "full_frame"
    w, h = jpeg_size(vlm.jpegs[0])
    assert (w, h) == (1280, 720)
    assert r["severity"] == "IGNORE" and r["context"]["source_type"] == "fog_dust_cloud"
    assert r["tokens_in"] is None  # classify gave only a total
    assert r["report_text"].startswith("[IGNORE]")


def test_vlm_failure_falls_back_to_monitor_for_a_detection():
    det = FakeDetector([Detection("fire", 0.8, (10, 10, 60, 60))])
    r = run_pass(frame(), det, FakeVLM(None, tokens=0), TOWER, now=0.0, timer=ticker())
    assert r["vlm_status"] == "failed" and r["severity"] == "MONITOR"
    assert "unavailable" in r["report_text"]


def test_vlm_exception_is_contained():
    class Boom:
        def classify(self, jpeg):
            raise RuntimeError("down")
    det = FakeDetector([Detection("fire", 0.8, (10, 10, 60, 60))])
    r = run_pass(frame(), det, Boom(), TOWER, now=0.0, timer=ticker())
    assert r["vlm_status"] == "failed" and r["severity"] == "MONITOR" and r["tokens"] == 0


def test_missing_vlm_is_reported_as_unavailable():
    det = FakeDetector([Detection("fire", 0.8, (10, 10, 60, 60))])
    r = run_pass(frame(), det, None, TOWER, now=0.0, timer=ticker())
    assert r["vlm_called"] is False and r["vlm_status"] == "unavailable"
    assert r["severity"] == "MONITOR"


def test_benign_context_is_logged():
    det = FakeDetector([Detection("smoke", 0.7, (10, 10, 60, 60))])
    r = run_pass(frame(), det, FakeVLM(CAMP), TOWER, now=0.0, timer=ticker())
    assert r["severity"] == "LOG"


# ---- ground truth ----

@pytest.mark.parametrize("sev,truth,want", [
    ("ALERT", "wildfire", True), ("MONITOR", "wildfire", True), ("LOG", "wildfire", False),
    ("IGNORE", "wildfire", False), ("IGNORE", "no_wildfire", True), ("ALERT", "no_wildfire", False),
    ("ALERT", "unknown", None), ("LOG", "unknown", None),
])
def test_ground_truth_score(sev, truth, want):
    assert ground_truth_score(sev, truth) is want


def test_ground_truth_rejects_unknown_label():
    with pytest.raises(ValueError):
        ground_truth_score("ALERT", "maybe")


# ---- escalation ----

def alert_result():
    det = FakeDetector([Detection("smoke", 0.9, (100, 100, 300, 250))])
    return run_pass(frame(), det, FakeVLM(WILD), TOWER, now=0.0, timer=ticker())


def test_alert_online_is_sent_with_forecast_and_counts_bytes():
    r = alert_result()
    calls = []

    def fc(lat, lon):
        calls.append((lat, lon))
        return {"time": ["t"], "temp_c": [33.0], "wind_mph": [12.0], "wind_dir_deg": [270.0]}
    e = escalate(r, True, forecast_fn=fc)
    assert e["decision"] == "sent" and calls == [(37.1, -121.9)]
    assert e["forecast"]["temp_c"] == [33.0]
    payload = dict(r["report"], forecast=e["forecast"], forecast_status="ok")
    assert e["bytes_up"] == len(json.dumps(payload).encode())
    assert e["bytes_up"] > len(r["report"]["thumbnail_jpeg_b64"])
    assert "wind 12 mph from W" in e["report_text"]


def test_alert_online_forecast_failure_still_sends():
    def fc(lat, lon):
        raise OSError("no route")
    e = escalate(alert_result(), True, forecast_fn=fc)
    assert e["decision"] == "sent" and e["forecast"] is None and e["bytes_up"] > 0
    assert "Forecast pending" in e["report_text"]


def test_alert_offline_is_queued_and_sends_nothing():
    called = []
    e = escalate(alert_result(), False, forecast_fn=lambda a, b: called.append(1))
    assert e["decision"] == "queued" and e["bytes_up"] == 0 and e["forecast"] is None
    assert e["payload_bytes"] > 0 and called == []


@pytest.mark.parametrize("ctx,dets,decision", [
    (CAMP, [Detection("smoke", 0.9, (10, 10, 60, 60))], "logged"),
    (WILD, [], "ignored"),
])
def test_non_alerts_never_leave_the_device(ctx, dets, decision):
    r = run_pass(frame(), FakeDetector(dets), FakeVLM(ctx), TOWER, now=0.0, timer=ticker())
    for online in (True, False):
        e = escalate(r, online, forecast_fn=lambda a, b: {})
        assert e["decision"] == decision and e["bytes_up"] == 0


# ---- session ----

def called_result(image_tokens=2691, prompt=380, completion=40, crop_tokens=256, sev="ALERT"):
    return {"vlm_called": True, "vlm_status": "ok", "tokens": prompt + completion, "tokens_in": prompt,
            "tokens_out": completion, "vlm_calls": 1, "crop_image_tokens": crop_tokens,
            "full_frame_image_tokens": image_tokens, "detect_ms": 4.0, "vlm_ms": 500.0, "severity": sev}


def skipped_result(image_tokens=2691):
    return {"vlm_called": False, "vlm_status": "skipped_no_detection", "tokens": 0, "tokens_in": None,
            "tokens_out": None, "vlm_calls": 0, "crop_image_tokens": 0,
            "full_frame_image_tokens": image_tokens, "detect_ms": 3.0, "vlm_ms": None, "severity": "IGNORE"}


def test_session_counts_and_ground_truth():
    s = Session()
    s.record("after", called_result(), {"decision": "sent", "bytes_up": 5000},
             truth="wildfire", image_bytes=200_000, online=True)
    s.record("after", skipped_result(), {"decision": "ignored", "bytes_up": 0},
             truth="wildfire", image_bytes=100_000, online=False)
    s.record("after", called_result(sev="ALERT"), {"decision": "queued", "bytes_up": 0},
             truth="unknown", image_bytes=100_000, online=False)
    p = s.summary()["pipelines"]["after"]
    assert p["images"] == 3 and p["detector_runs"] == 3 and p["vlm_calls"] == 2
    assert p["calls_avoided"] == 1
    assert p["tokens_actual"] == 840 and p["tokens_in"] == 760 and p["tokens_out"] == 80
    assert p["bytes_up"] == 5000 and p["image_upload_bytes"] == 400_000
    assert p["gt_total"] == 2 and p["gt_correct"] == 1 and p["accuracy"] == 0.5
    assert p["alerts_sent"] == 1 and p["alerts_queued"] == 1
    assert p["online_images"] == 1 and p["offline_images"] == 2
    assert p["latency_ms_p50"] == pytest.approx(504.0)
    assert p["detect_ms_p50"] == pytest.approx(4.0)


def test_session_request_level_network_counts():
    s = Session()
    s.record_request(online=True)
    s.record_request(online=False)
    s.record_request(online=False)
    n = s.summary()["network"]
    assert n == {"requests": 3, "online_requests": 1, "offline_requests": 2,
                 "cloud_only_blind": 2, "alerts_sent": 0, "alerts_queued": 0}


def test_session_full_frame_estimate_uses_observed_overhead():
    s = Session()
    # overhead = 380 - 256 = 124 prompt tokens per call; 40 generated tokens per call
    s.record("after", called_result(), {"decision": "sent", "bytes_up": 0},
             truth="unknown", image_bytes=0, online=True)
    s.record("after", skipped_result(), {"decision": "ignored", "bytes_up": 0},
             truth="unknown", image_bytes=0, online=True)
    p = s.summary()["pipelines"]["after"]
    assert p["full_frame_tokens_in_est"] == 2 * 2691 + 2 * 124
    assert p["full_frame_tokens_out_est"] == 2 * 40
    assert p["full_frame_tokens_est"] == 2 * 2691 + 2 * 124 + 2 * 40


def test_economics_with_zero_prices_is_all_zero_and_safe():
    s = Session()
    e = s.economics({})
    assert e["pipelines"] == {}
    s.record("before", skipped_result(), {"decision": "ignored", "bytes_up": 0},
             truth="unknown", image_bytes=0, online=True)
    e = s.economics({"usd_per_mtok_in": 0, "usd_per_mtok_out": 0, "usd_per_gb": 0})
    b = e["pipelines"]["before"]
    assert b["edge_usd"] == 0 and b["cloud_usd"] == 0 and b["savings_usd"] == 0
    assert b["savings_pct"] is None and b["bytes_savings_pct"] is None
    assert b["token_savings_pct"] == 100.0
    assert e["priced"] is False


def test_economics_with_prices():
    s = Session()
    s.record("after", called_result(), {"decision": "sent", "bytes_up": 10_000},
             truth="unknown", image_bytes=1_000_000, online=True)
    s.record("after", skipped_result(), {"decision": "ignored", "bytes_up": 0},
             truth="unknown", image_bytes=1_000_000, online=True)
    prices = {"usd_per_mtok_in": 2.0, "usd_per_mtok_out": 8.0, "usd_per_gb": 10.0}
    a = s.economics(prices)["pipelines"]["after"]
    cloud_in, cloud_out = 2 * 2691 + 2 * 124, 2 * 40
    want_cloud = cloud_in / 1e6 * 2.0 + cloud_out / 1e6 * 8.0 + 2_000_000 / 1e9 * 10.0
    want_edge = 10_000 / 1e9 * 10.0
    assert a["cloud_usd"] == pytest.approx(want_cloud)
    assert a["edge_usd"] == pytest.approx(want_edge)
    assert a["savings_usd"] == pytest.approx(want_cloud - want_edge)
    assert a["savings_pct"] == pytest.approx(100 * (want_cloud - want_edge) / want_cloud)
    assert a["token_savings_pct"] == pytest.approx(100 * (1 - 420 / (cloud_in + cloud_out)))
    assert a["bytes_savings_pct"] == pytest.approx(100 * (1 - 10_000 / 2_000_000))


def test_economics_single_price_fallback():
    s = Session()
    s.record("after", called_result(), {"decision": "sent", "bytes_up": 0},
             truth="unknown", image_bytes=0, online=True)
    a = s.economics({"usd_per_mtok": 1.0})["pipelines"]["after"]
    assert a["cloud_usd"] == pytest.approx((2691 + 124 + 40) / 1e6)


def test_session_reset():
    s = Session()
    s.record("after", called_result(), {"decision": "sent", "bytes_up": 1},
             truth="wildfire", image_bytes=1, online=True)
    s.record_request(online=True)
    s.reset()
    assert s.summary() == {"pipelines": {}, "network": {"requests": 0, "online_requests": 0,
                                                        "offline_requests": 0, "cloud_only_blind": 0,
                                                        "alerts_sent": 0, "alerts_queued": 0}}


# ---- benchmarks ----

def test_load_benchmarks_reads_present_files_only(tmp_path):
    (tmp_path / "detector_before_yoloworld.json").write_text(json.dumps(
        {"map50": 0.002, "precision": 0.1, "recall": 0.01, "ms_per_image": 3.8, "n_images": 10}))
    (tmp_path / "context_after_lora7b.json").write_text(json.dumps(
        {"source_type_acc": 0.8, "group_acc": 0.9, "latency_ms_p50": 700, "tokens_per_call": 420}))
    (tmp_path / "detector_after_yolo11s.json").write_text("{not json")
    b = load_benchmarks(tmp_path)
    assert b["detector"]["before"] == {"map50": 0.002, "precision": 0.1, "recall": 0.01,
                                       "ms_per_image": 3.8, "n_images": 10}
    assert b["detector"]["after"] is None
    assert b["context"]["before"] is None
    assert b["context"]["after"]["tokens_per_call"] == 420


def test_load_benchmarks_missing_dir(tmp_path):
    b = load_benchmarks(tmp_path / "nope")
    assert b == {"detector": {"before": None, "after": None}, "context": {"before": None, "after": None}}


def test_timed_out_vlm_call_reports_no_token_split():
    det = FakeDetector([Detection("fire", 0.8, (10, 10, 60, 60))])
    vlm = FakeVLM(None, tokens=0, usage={"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
    r = run_pass(frame(), det, vlm, TOWER, now=0.0, timer=ticker())
    assert r["vlm_status"] == "failed" and r["tokens_in"] is None and r["vlm_calls"] == 0
    s = Session()
    s.record("after", r, {"decision": "logged", "bytes_up": 0}, truth="unknown", image_bytes=0, online=True)
    p = s.summary()["pipelines"]["after"]
    assert p["vlm_calls"] == 1 and p["tokens_in"] is None and p["overhead_observed"] is False


def test_full_frame_overhead_is_pooled_across_pipelines():
    # same images, same cloud baseline: BEFORE never called its VLM but still gets the
    # prompt/output overhead observed in AFTER's calls
    s = Session()
    for name, res in (("before", skipped_result()), ("after", called_result())):
        s.record(name, res, {"decision": "ignored", "bytes_up": 0}, truth="unknown", image_bytes=0, online=True)
    p = s.summary()["pipelines"]
    assert p["before"]["full_frame_tokens_est"] == p["after"]["full_frame_tokens_est"] == 2691 + 124 + 40
    assert p["before"]["overhead_observed"] is True

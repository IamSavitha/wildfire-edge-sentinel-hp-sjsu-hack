import cv2
import numpy as np
import pytest

from scripts.bench import NullVLM, bench, read_clips, run_clip, score, summarize
from sentinel.config import Settings
from sentinel.metrics import Metrics
from sentinel.schema import ContextResult, Detection, Severity

SMOKE = Detection("smoke", 0.9, (10, 10, 40, 40))


def ctx(**kw):
    base = dict(source_type="wildland", smoke_color="grey", attended="no", near_structures=True,
                near_road=False, size_estimate="small", description="d")
    base.update(kw)
    return ContextResult(**base)


class FakeVLM:
    def __init__(self, result, tokens=300):
        self.result, self.tokens, self.calls = result, tokens, 0

    def classify(self, jpeg):
        self.calls += 1
        return self.result, self.tokens


def fixed_detector(dets):
    return lambda frame: list(dets)


def write_clip(root, name, n=4, value=40):
    d = root / name
    d.mkdir()
    for i in range(n):
        cv2.imwrite(str(d / f"{i:03d}.jpg"), np.full((64, 64, 3), value, np.uint8))
    return str(d)


SETTINGS = Settings(fps=2.0, min_frames=3, cooldown_s=0, recheck_s=30, max_rechecks=2)


def test_run_clip_alerts_on_dangerous_context(tmp_path):
    m = Metrics()
    sev = run_clip(write_clip(tmp_path, "fire"), SETTINGS, fixed_detector([SMOKE]), FakeVLM(ctx()), m)
    assert sev == [Severity.ALERT]
    assert m.counters["frames"] == 4 and m.counters["vlm_calls"] == 1
    assert m.counters["vlm_tokens"] == 300


def test_run_clip_nothing_detected(tmp_path):
    m = Metrics()
    vlm = FakeVLM(ctx())
    assert run_clip(write_clip(tmp_path, "clear"), SETTINGS, fixed_detector([]), vlm, m) == []
    assert vlm.calls == 0 and m.counters["candidates"] == 0


def test_run_clip_reports_provisional_active_event(tmp_path):
    # small unattended campfire -> MONITOR, still waiting for its trend re-check at clip end
    m = Metrics()
    sev = run_clip(write_clip(tmp_path, "camp"), SETTINGS, fixed_detector([SMOKE]),
                   FakeVLM(ctx(source_type="campfire", near_structures=False)), m)
    assert sev == [Severity.MONITOR]


def test_score_counts_confusion():
    rows = [{"label": "alert"}, {"label": "alert"}, {"label": "no_alert"}, {"label": "no_alert"}]
    out = score(rows, [True, False, True, False])
    assert out == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": 0.5, "recall": 0.5,
                   "false_alarms": 1, "missed": 1}


def test_score_no_predictions_is_zero_not_nan():
    out = score([{"label": "alert"}], [False])
    assert out["precision"] == 0.0 and out["recall"] == 0.0 and out["missed"] == 1


def test_score_rejects_unknown_label_and_length_mismatch():
    with pytest.raises(ValueError):
        score([{"label": "maybe"}], [True])
    with pytest.raises(ValueError):
        score([{"label": "alert"}], [])


def test_bench_end_to_end_with_fakes(tmp_path):
    rows = [{"path": write_clip(tmp_path, "fire"), "label": "alert"},
            {"path": write_clip(tmp_path, "fog"), "label": "no_alert"},
            {"path": write_clip(tmp_path, "clear", value=250), "label": "no_alert"}]

    def detector(frame):  # the bright "clear" clip has no smoke
        return [] if frame.mean() > 200 else [SMOKE]

    class SequenceVLM:  # one call per event, in clip order
        def __init__(self):
            self.results = [ctx(), ctx(source_type="fog_dust_cloud")]

        def classify(self, jpeg):
            return self.results.pop(0), 250

    out = bench("after", rows, SETTINGS, detector, SequenceVLM())
    assert out["name"] == "after" and out["detector_only"] is False
    assert out["precision"] == 1.0 and out["recall"] == 1.0
    assert out["false_alarms"] == 0 and out["missed"] == 0
    assert out["tokens_per_vlm_call"] == 250
    assert out["frames_per_vlm_call"] == 12 / 2  # 3 clips x 4 frames, 2 VLM calls
    assert out["time_to_decision_s_p50"] == 0.0  # both decided immediately at the VLM call
    assert [c["predicted_alert"] for c in out["clips"]] == [True, False, False]
    assert out["clips"][1]["severities"] == ["IGNORE"]


def test_detector_only_counts_any_candidate_as_alert(tmp_path):
    rows = [{"path": write_clip(tmp_path, "fog"), "label": "no_alert"}]
    out = bench("detector_only", rows, SETTINGS, fixed_detector([SMOKE]), NullVLM(), detector_only=True)
    assert out["false_alarms"] == 1 and out["precision"] == 0.0
    assert out["tokens_per_vlm_call"] == 0 and out["frames_per_vlm_call"] is None


def test_summarize_without_vlm_calls():
    out = summarize("x", {"precision": 0.0, "recall": 0.0, "false_alarms": 0, "missed": 0},
                    {"frames": 10}, [], detector_only=False)
    assert out["tokens_per_vlm_call"] == 0 and out["frames_per_vlm_call"] is None
    assert out["time_to_decision_s_p50"] is None and out["time_to_decision_s_p95"] is None


def test_read_clips(tmp_path):
    p = tmp_path / "clips.csv"
    p.write_text("path,label\ndata/demo/a,alert\n\ndata/demo/b, no_alert\n")
    assert read_clips(p) == [{"path": "data/demo/a", "label": "alert"},
                             {"path": "data/demo/b", "label": "no_alert"}]
    p.write_text("path,label\ndata/demo/a,alrt\n")
    with pytest.raises(ValueError):
        read_clips(p)


def test_overrides_before_uses_yoloworld_prompts_and_base_model():
    from scripts.bench import apply_overrides, parse_args
    a = parse_args(["--name", "before", "--detector-weights", "yolov8s-worldv2.pt",
                    "--detector-classes", "smoke, fire", "--model", "Qwen/base"])
    s = apply_overrides(Settings(), a)
    assert s.detector_weights == "yolov8s-worldv2.pt" and s.detector_classes == ["smoke", "fire"]
    assert s.vlm_model == "Qwen/base" and s.full_frame is False


def test_overrides_new_weights_without_classes_drop_prompts():
    from scripts.bench import apply_overrides, parse_args
    base = Settings(detector_weights="yolov8s-worldv2.pt", detector_classes=["smoke", "fire"])
    after = apply_overrides(base, parse_args(["--name", "after", "--detector-weights", "models/smoke_yolo.pt",
                                              "--model", "context", "--full-frame"]))
    assert after.detector_classes is None and after.vlm_model == "context" and after.full_frame
    kept = apply_overrides(base, parse_args(["--name", "x"]))
    assert kept.detector_classes == ["smoke", "fire"] and kept.vlm_model == base.vlm_model


def test_overrides_recheck_s_only_when_given():
    from scripts.bench import apply_overrides, parse_args
    assert apply_overrides(Settings(recheck_s=30), parse_args(["--name", "x", "--recheck-s", "5"])).recheck_s == 5.0
    assert apply_overrides(Settings(recheck_s=30), parse_args(["--name", "x"])).recheck_s == 30


def test_run_clip_reports_first_alert_time_and_payload(tmp_path):
    info = {}
    run_clip(write_clip(tmp_path, "fire"), SETTINGS, fixed_detector([SMOKE]), FakeVLM(ctx()), Metrics(), info)
    assert info["first_alert_s"] == 1.0  # min_frames=3 at 2 fps: the gate opens on frame 2
    assert info["first_alert_compute_ms"] >= 0
    assert info["alert_payload_bytes"] > 100  # JSON report with its thumbnail


def test_run_clip_without_alert_reports_none(tmp_path):
    info = {}
    run_clip(write_clip(tmp_path, "fog"), SETTINGS, fixed_detector([SMOKE]),
             FakeVLM(ctx(source_type="fog_dust_cloud")), Metrics(), info)
    assert info == {"first_alert_s": None, "first_alert_compute_ms": None, "alert_payload_bytes": None}


def test_bench_rows_carry_first_alert_fields(tmp_path):
    rows = [{"path": write_clip(tmp_path, "fire"), "label": "alert"}]
    out = bench("after", rows, SETTINGS, fixed_detector([SMOKE]), FakeVLM(ctx()))
    clip = out["clips"][0]
    assert clip["first_alert_s"] == 1.0 and clip["alert_payload_bytes"] > 0


def test_detector_imgsz_override():
    from scripts.bench import apply_overrides, parse_args
    assert apply_overrides(Settings(), parse_args(["--name", "x"])).detector_imgsz == 640
    s = apply_overrides(Settings(), parse_args(["--name", "x", "--detector-weights", "models/joint_yolo.pt",
                                               "--detector-imgsz", "960"]))
    assert s.detector_imgsz == 960

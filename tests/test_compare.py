import json

from scripts.compare import load_results, render

DET_BEFORE = {"name": "before_yoloworld", "weights": "yolov8s-worldv2.pt", "classes": ["smoke", "fire"],
              "map50": 0.2, "map50_95": 0.08, "precision": 0.31, "recall": 0.25,
              "per_class_map50": {"smoke": 0.2}, "ms_per_image": 30.4}
DET_AFTER = {"name": "after_yolo11s", "weights": "models/smoke_yolo.pt", "classes": None,
             "map50": 0.654, "map50_95": 0.35, "precision": 0.7, "recall": 0.6,
             "per_class_map50": {"smoke": 0.7}, "ms_per_image": 12.6}


def ctx(name, acc, group, fail, p50, tokens):
    return {"name": name, "model": "m", "split": "s", "n": 200, "source_type_acc": acc, "group_acc": group,
            "parse_fail_rate": fail, "latency_ms_p50": p50, "latency_ms_p95": p50 * 2,
            "tokens_per_call": tokens, "per_source": {}}


def bench(name, p, r, fa, tok, fpc):
    return {"name": name, "precision": p, "recall": r, "false_alarms": fa, "missed": 1,
            "tokens_per_vlm_call": tok, "frames_per_vlm_call": fpc}


FULL = {
    "detector_before_yoloworld": DET_BEFORE,
    "detector_after_yolo11s": DET_AFTER,
    "context_linear_probe": {"name": "linear_probe", "n": 200, "source_type_acc": 0.5, "group_acc": 0.7,
                             "latency_ms_per_image": 8.2},
    "context_before_base7b": ctx("before_base7b", 0.41, 0.6, 0.05, 950.0, 812.0),
    "context_after_lora7b": ctx("after_lora7b", 0.82, 0.9, 0.0, 700.4, 790.0),
    "context_ref_teacher32b": ctx("ref_teacher32b", 1.0, 1.0, 0.0, 3000.0, 800.0),
    "bench_before": bench("before", 0.5, 0.75, 6, 812.0, 20.0),
    "bench_after": bench("after", 0.9, 0.8, 1, 790.0, 55.5),
}


def lines_with(md, text):
    return [ln for ln in md.splitlines() if text in ln]


def test_full_render_has_three_tables_and_no_missing_note():
    md = render(FULL)
    assert "## Detector" in md and "## Context VLM" in md and "## End-to-end" in md
    assert "Missing" not in md


def test_detector_table_formats_percent_ms_and_delta():
    md = render(FULL)
    (row,) = lines_with(md, "| mAP50 |")
    assert row == "| mAP50 | 20.0% | 65.4% | +45.4 pp |"
    (row,) = lines_with(md, "| ms/image |")
    assert row == "| ms/image | 30 | 13 | -18 |"


def test_context_rows_in_fixed_order_with_linear_probe_blanks():
    md = render(FULL)
    rows = [ln for ln in md.split("## Context VLM")[1].split("##")[0].splitlines() if ln.startswith("| `")]
    assert [r.split("|")[1].strip() for r in rows] == [
        "`linear_probe`", "`before_base7b`", "`after_lora7b`", "`ref_teacher32b`"]
    assert rows[0].endswith("| 50.0% | 70.0% | — | 8 | — |")
    assert rows[2].endswith("| 82.0% | 90.0% | 0.0% | 700 | 790 |")


def test_end_to_end_table():
    md = render(FULL)
    (row,) = lines_with(md, "| `after` |")
    assert row == "| `after` | 90.0% | 80.0% | 1 | 1 | 790 | 55.5 |"


def test_partial_results_omit_rows_tables_and_note_missing():
    md = render({"detector_after_yolo11s": DET_AFTER, "context_before_base7b": FULL["context_before_base7b"]})
    assert "## End-to-end" not in md
    (row,) = lines_with(md, "| mAP50 |")
    assert row == "| mAP50 | 65.4% |"  # no BEFORE column, no delta
    assert "`after_lora7b`" not in md
    (note,) = lines_with(md, "Missing")
    for f in ("detector_before_yoloworld.json", "context_after_lora7b.json", "context_linear_probe.json",
              "bench_before.json", "bench_after.json"):
        assert f in note
    assert "ref_teacher32b" not in note  # optional rows are not reported as missing


def test_empty_results():
    md = render({})
    assert "## Detector" not in md and "Missing" in md


def test_none_values_render_as_dash():
    md = render({"bench_detector_only": bench("detector_only", 0.3, 0.9, 8, 0, None)})
    (row,) = lines_with(md, "| `detector_only` |")
    assert row.endswith("| 0 | — |")


def test_load_results_reads_known_prefixes(tmp_path):
    (tmp_path / "detector_after_yolo11s.json").write_text(json.dumps(DET_AFTER))
    (tmp_path / "bench_x.json").write_text(json.dumps(bench("x", 1, 1, 0, 1, 1)))
    (tmp_path / "cost_model.jsonl").write_text("{}\n")
    (tmp_path / "notes.json").write_text("{}")
    assert sorted(load_results(tmp_path)) == ["bench_x", "detector_after_yolo11s"]

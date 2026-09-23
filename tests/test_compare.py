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
            "tokens_per_vlm_call": tok, "frames_per_vlm_call": fpc,
            "time_to_decision_s_p50": 5.0, "time_to_decision_s_p95": 12.25}


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
    assert row == "| ms/image | 30.4 | 12.6 | -17.8 |"


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
    assert row == "| `after` | 90.0% | 80.0% | 1 | 1 | 790 | 55.5 | 5.0 | 12.2 |"
    assert "Decision p50 (sim s)" in md and "Decision p95 (sim s)" in md


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
    b = bench("detector_only", 0.3, 0.9, 8, 0, None)
    b["time_to_decision_s_p50"] = b["time_to_decision_s_p95"] = None
    md = render({"bench_detector_only": b})
    (row,) = lines_with(md, "| `detector_only` |")
    assert row.endswith("| 0 | — | — | — |")


def test_load_results_reads_known_prefixes(tmp_path):
    (tmp_path / "detector_after_yolo11s.json").write_text(json.dumps(DET_AFTER))
    (tmp_path / "bench_x.json").write_text(json.dumps(bench("x", 1, 1, 0, 1, 1)))
    (tmp_path / "cost_model.jsonl").write_text("{}\n")
    (tmp_path / "notes.json").write_text("{}")
    assert sorted(load_results(tmp_path)) == ["bench_x", "detector_after_yolo11s"]


# ---------------------------------------------------------------- edge vs cloud-only

CLOUD_CTX = {**ctx("cloud_qwen7b", 0.44, 0.52, 0.02, 1400.0, 1500.0), "cloud": True,
             "provider_host": "api.example.com", "response_format": "json_object",
             "latency_ms_p95": 2600.0, "prompt_tokens_per_call": 1210.0, "completion_tokens_per_call": 58.0}


def edge_bench_with_alerts():
    b = bench("after", 0.9, 0.8, 1, 790.0, 55.5)
    b["clips"] = [{"path": "fire", "label": "alert", "frames": 40, "first_alert_s": 1.5,
                   "first_alert_compute_ms": 12000.0, "alert_payload_bytes": 20_000},
                  {"path": "fog", "label": "no_alert", "frames": 40, "first_alert_s": None,
                   "first_alert_compute_ms": None, "alert_payload_bytes": None}]
    return b


CLOUD_BENCH = {"name": "cloud_qwen7b", "recall": 0.6, "false_alarms": 3, "missed": 6, "frame_stride": 1,
               "outages": [], "prices": {"usd_per_mtok_in": 0.2, "usd_per_mtok_out": 0.6, "usd_per_gb": 0},
               "profiles": {"fiber": {"recall": 0.6, "time_to_first_alert_s_p50": 2.4, "decision_delay_s_p95": 1.9,
                                      "max_backlog_s": 0.0, "usd_per_1000_frames": 0.2766,
                                      "bytes_per_1000_frames": 150e6},
                            "rural_cellular": {"recall": 0.6, "time_to_first_alert_s_p50": 30.1,
                                               "decision_delay_s_p95": 45.0, "max_backlog_s": 38.2,
                                               "usd_per_1000_frames": 0.2766, "bytes_per_1000_frames": 150e6}}}


def test_no_cloud_results_no_section():
    assert "Edge vs cloud-only" not in render(FULL)


def test_bench_cloud_is_not_listed_as_an_edge_bench_row():
    md = render({**FULL, "bench_cloud_cloud_qwen7b": CLOUD_BENCH})
    e2e = md.split("## End-to-end")[1].split("## Edge vs cloud-only")[0]
    assert "`cloud_qwen7b`" not in e2e


def test_edge_vs_cloud_section_with_everything():
    from scripts.outage_scenario import scenario
    from sentinel.netprofile import PROFILES

    edge = edge_bench_with_alerts()
    cloud_clips = {**CLOUD_BENCH, "fps": 2.0, "clips": [
        {"path": "fire", "frames": [{"t": 0.0, "bytes": 150_000, "severity": "ALERT", "latency_ms": 900.0}]},
        {"path": "fog", "frames": [{"t": 0.0, "bytes": 150_000, "severity": "IGNORE", "latency_ms": 900.0}]}]}
    outage = scenario({**edge, "config": {"fps": 2.0}}, cloud_clips, [PROFILES["fiber"]], 10)
    md = render({**FULL, "bench_after": edge, "context_cloud_qwen7b": CLOUD_CTX,
                 "bench_cloud_cloud_qwen7b": CLOUD_BENCH, "outage_after_vs_cloud_10min": outage})
    sec = md.split("## Edge vs cloud-only (measured)")[1]
    assert "Not run yet" not in sec
    (row,) = lines_with(sec, "cloud (api.example.com, json_object)")
    assert "| 44.0% |" in row and "| 1400 | 2600 | 1210 in + 58 out |" in row
    (row,) = [ln for ln in lines_with(sec, "`after_lora7b`")]
    assert row.endswith("| 0 (local) |")
    (row,) = lines_with(sec, "cloud-only, stride 1")
    assert "| 60.0% | 3 | 6 | $0.2766 | 150.00 MB |" in row
    (row,) = lines_with(sec, "| `after` | edge cascade")
    assert row.endswith("| $0 API | 0.25 MB |")  # 20 KB alert over 80 frames
    (row,) = lines_with(sec, "| rural_cellular |")
    assert "| 30.1 | 45.0 | 38.2 | 60.0% |" in row
    edge_rural = 1.5 + 12.0 + (0.15 + 20_000 * 8 / 1e6) / 0.98
    assert row.startswith(f"| rural_cellular | {edge_rural:.1f} |")
    assert "### Outage scenario" in sec and "| fiber |" in sec.split("### Outage scenario")[1]


def test_partial_cloud_results_note_what_is_missing():
    md = render({"context_cloud_qwen7b": {**CLOUD_CTX, "latency_ms_p50": None, "latency_ms_p95": None,
                                          "cached_latency_ms_p50": 1300.0, "cached_latency_ms_p95": 2000.0}})
    sec = md.split("## Edge vs cloud-only (measured)")[1]
    (note,) = lines_with(sec, "Not run yet")
    assert "bench_cloud_" in note and "outage_" in note and "context_cloud_" not in note
    (row,) = lines_with(sec, "latency from cache")
    assert "| 1300 | 2000 |" in row
    assert "Time to dispatch" not in sec


def test_unpriced_cloud_and_old_edge_results():
    old_edge = bench("after", 0.9, 0.8, 1, 790.0, 55.5)  # no clips / first_alert_s
    md = render({"bench_after": old_edge, "bench_cloud_c": {**CLOUD_BENCH, "prices": {}}})
    (row,) = lines_with(md, "cloud-only, stride 1")
    assert "unpriced" in row
    assert "re-run bench_after" in md
    (row,) = lines_with(md, "| fiber |")
    assert row.startswith("| fiber | — | 2.4 |")


def test_load_results_reads_outage_files(tmp_path):
    (tmp_path / "outage_x.json").write_text("{}")
    (tmp_path / "outage_x.md").write_text("")
    assert list(load_results(tmp_path)) == ["outage_x"]

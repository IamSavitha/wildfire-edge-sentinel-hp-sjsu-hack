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


def edge_bench_with_alerts(name="after", first=1.5, recheck=5.0):
    b = bench(name, 0.9, 0.8, 1, 790.0, 55.5)
    b["config"] = {"fps": 2.0, "recheck_s": recheck}
    b["clips"] = [{"path": "data/bench/fire", "label": "alert", "frames": 40, "first_alert_s": first,
                   "first_alert_compute_ms": 12000.0, "alert_payload_bytes": 20_000},
                  {"path": "data/bench/fog", "label": "no_alert", "frames": 40, "first_alert_s": None,
                   "first_alert_compute_ms": None, "alert_payload_bytes": None}]
    return b


def prof(recall, tta, delay, backlog, usd, by_clip):
    return {"recall": recall, "time_to_first_alert_s_p50": tta, "decision_delay_s_p95": delay,
            "max_backlog_s": backlog, "usd_per_1000_frames": usd, "usd_per_1000_captured_frames": usd,
            "bytes_per_1000_frames": 150e6, "bytes_per_1000_captured_frames": 150e6, "first_alert_s_by_clip": by_clip}


CLOUD_BENCH = {
    "name": "cloud_qwen7b", "model": "Qwen/Qwen2.5-VL-7B-Instruct", "recall": 0.6, "false_alarms": 3, "missed": 6,
    "frame_stride": 1, "policy": "latest", "persist": 3, "recheck_s": 5.0, "outages": [], "fps": 2.0,
    "prices": {"usd_per_mtok_in": 0.2, "usd_per_mtok_out": 0.6, "usd_per_gb": 0},
    "clips": [{"path": "data/bench/fire", "label": "alert", "frames": []},
              {"path": "data/bench/fog", "label": "no_alert", "frames": []}],
    "profiles": {"fiber": prof(0.6, 2.4, 1.9, 0.0, 0.2766, [2.4, None]),
                 "rural_cellular": prof(0.6, 30.1, 45.0, 38.2, 0.2766, [30.1, None])},
    "temporal": {"recall": 0.5, "false_alarms": 1, "missed": 7,
                 "profiles": {"fiber": prof(0.5, 4.0, 1.9, 0.0, 0.2766, [4.0, None]),
                              "rural_cellular": prof(0.5, 33.0, 45.0, 38.2, 0.2766, [33.0, None])}},
    "cadence": {"stride": 20, "seconds_between_frames": 10.0,
                "per_frame": {"recall": 0.4, "false_alarms": 0, "missed": 9,
                              "profiles": {"fiber": prof(0.4, 10.9, 1.2, 0.0, 0.0138, [10.9, None])}},
                "temporal": {"recall": 0.0, "false_alarms": 0, "missed": 15, "profiles": {}}},
}


def test_no_cloud_results_no_section():
    assert "Edge vs cloud-only" not in render(FULL)


def test_bench_cloud_is_not_listed_as_an_edge_bench_row():
    md = render({**FULL, "bench_cloud_cloud_qwen7b": CLOUD_BENCH})
    e2e = md.split("## End-to-end")[1].split("## Edge vs cloud-only")[0]
    assert "`cloud_qwen7b`" not in e2e


def _everything():
    from scripts.outage_scenario import scenario
    from sentinel.netprofile import PROFILES

    edge, before = edge_bench_with_alerts(), edge_bench_with_alerts("before", first=3.0)
    ctx = {"source_type": "wildland", "smoke_color": "grey", "attended": "no", "near_structures": True,
           "near_road": False, "size_estimate": "small", "description": "d"}
    cloud_clips = {**CLOUD_BENCH, "clips": [
        {"path": "data/bench/fire", "label": "alert", "frames": [
            {"i": 0, "t": 0.0, "bytes": 150_000, "severity": "ALERT", "ctx": ctx, "latency_ms": 900.0}]},
        {"path": "data/bench/fog", "label": "no_alert", "frames": [
            {"i": 0, "t": 0.0, "bytes": 150_000, "severity": "IGNORE", "ctx": None, "latency_ms": 900.0}]}]}
    outage = scenario(edge, cloud_clips, [PROFILES["fiber"]], 10)
    return render({**FULL, "bench_before": before, "bench_after": edge, "context_cloud_qwen7b": CLOUD_CTX,
                   "bench_cloud_cloud_qwen7b": CLOUD_BENCH, "outage_after_vs_cloud_10min": outage})


def test_edge_vs_cloud_header_names_the_recorded_model():
    sec = _everything().split("## Edge vs cloud-only (measured)")[1]
    assert "Not run yet" not in sec
    assert "`Qwen/Qwen2.5-VL-7B-Instruct` (as recorded by the runs" in sec


def test_edge_vs_cloud_crop_table():
    sec = _everything().split("## Edge vs cloud-only (measured)")[1]
    (row,) = lines_with(sec, "cloud (api.example.com, json_object)")
    assert "| 44.0% |" in row and "| 1400 | 2600 | 1210 in + 58 out |" in row
    (row,) = lines_with(sec, "`after_lora7b`")
    assert row.endswith("| 0 (local) |")


def test_edge_vs_cloud_clip_table_has_every_rule_and_cadence():
    sec = _everything().split("### Tower clips")[1].split("###")[0]
    (row,) = lines_with(sec, "cloud-only, per-frame, every frame")
    assert "| 60.0% | 3 | 6 | $0.2766 | 150.00 MB |" in row
    (row,) = lines_with(sec, "cloud-only, temporal (persist 3, trend over 5 s)")
    assert "| 50.0% | 1 | 7 |" in row
    (row,) = lines_with(sec, "per-frame, one frame per 10 s")
    assert "| 40.0% | 0 | 9 | $0.0138 |" in row
    (row,) = lines_with(sec, "| `after` (recheck 5 s) | edge cascade")
    assert row.endswith("| $0 API | 0.25 MB |")  # 20 KB alert over 80 frames
    assert lines_with(sec, "| `before` (recheck 5 s) | edge cascade")


def test_time_to_dispatch_includes_base_and_after_and_flags_the_deployed_recheck():
    from scripts.outage_scenario import edge_knows_s
    from sentinel.netprofile import PROFILES
    sec = _everything().split("### Time to dispatch")[1].split("### Paired")[0]
    assert "Edge `before` (recheck 5 s)" in sec and "Edge `after` (recheck 5 s)" in sec
    (row,) = lines_with(sec, "| rural_cellular |")
    edge = edge_knows_s(edge_bench_with_alerts()["clips"][0], PROFILES["rural_cellular"], 0.0)
    before = edge_knows_s(edge_bench_with_alerts("before", 3.0)["clips"][0], PROFILES["rural_cellular"], 0.0)
    assert row.startswith(f"| rural_cellular | {before:.1f} | {edge:.1f} | 30.1 | 33.0 | — | — | 45.0 | 38.2 | $0.2766 |")
    assert "recheck_s = 30 s" in sec and "--recheck-s 30" in sec


def test_paired_table_and_per_clip_deltas():
    from scripts.outage_scenario import edge_knows_s
    from sentinel.netprofile import PROFILES
    md = _everything()
    sec = md.split("### Paired")[1].split("### Outage")[0]
    (row,) = lines_with(sec, "| `after` (recheck 5 s) | fiber |")
    e = edge_knows_s(edge_bench_with_alerts()["clips"][0], PROFILES["fiber"], 0.0)
    assert row == f"| `after` (recheck 5 s) | fiber | 1 | {e:.1f} | 2.4 | {2.4 - e:.1f} |"
    (row,) = lines_with(sec, "| `fire` |")
    assert row == f"| `fire` | {e:.1f} | 2.4 | {2.4 - e:+.1f} |"
    assert "### Outage scenario" in md and "Frames analysed during outage" in md


def test_recheck30_run_removes_the_note():
    b30 = edge_bench_with_alerts("after_recheck30", recheck=30.0)
    md = render({"bench_after": edge_bench_with_alerts(), "bench_after_recheck30": b30,
                 "bench_cloud_cloud_qwen7b": CLOUD_BENCH})
    assert "Edge `after_recheck30` (recheck 30 s)" in md and "--recheck-s 30" not in md


def test_partial_cloud_results_note_what_is_missing():
    md = render({"context_cloud_qwen7b": {**CLOUD_CTX, "latency_ms_p50": None, "latency_ms_p95": None,
                                          "cached_latency_ms_p50": 1300.0, "cached_latency_ms_p95": 2000.0}})
    sec = md.split("## Edge vs cloud-only (measured)")[1]
    (note,) = lines_with(sec, "Not run yet")
    assert "bench_cloud_" in note and "outage_" in note and "--cloud" not in note
    (row,) = lines_with(sec, "latency from cache")
    assert "| 1300 | 2000 |" in row
    assert "Time to dispatch" not in sec


def test_unpriced_cloud_and_old_edge_results():
    old_edge = bench("after", 0.9, 0.8, 1, 790.0, 55.5)  # no clips / first_alert_s
    md = render({"bench_after": old_edge, "bench_cloud_c": {**CLOUD_BENCH, "prices": {}}})
    (row,) = lines_with(md, "cloud-only, per-frame, every frame")
    assert "unpriced" in row
    assert "re-run the edge bench" in md
    (row,) = lines_with(md, "| fiber |")
    assert row.startswith("| fiber | 2.4 | 4.0 | 10.9 | — |")


def test_load_results_reads_outage_files(tmp_path):
    (tmp_path / "outage_x.json").write_text("{}")
    (tmp_path / "outage_x.md").write_text("")
    assert list(load_results(tmp_path)) == ["outage_x"]

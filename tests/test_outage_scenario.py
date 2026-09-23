import json

import pytest

from scripts.outage_scenario import (edge_delivery_attempt_s, edge_knows_s, live_after_restore, main, markdown,
                                     scenario, with_backlog)
from sentinel.main import FLUSH_EVERY_S
from sentinel.netprofile import EDGE_POST_RTTS, PROFILES, rtt_s, serialize_s, transfer_s
from sentinel.outbox import MAX_BACKOFF_S

FIBER, SAT = PROFILES["fiber"], PROFILES["satellite_geo"]


def edge_clip(path, label, first_alert_s, frames=40, compute_ms=2000.0, nbytes=20_000):
    return {"path": path, "label": label, "frames": frames, "first_alert_s": first_alert_s,
            "first_alert_compute_ms": compute_ms if first_alert_s is not None else None,
            "alert_payload_bytes": nbytes if first_alert_s is not None else None}


ALERT_CTX = {"source_type": "wildland", "smoke_color": "grey", "attended": "no", "near_structures": True,
             "near_road": False, "size_estimate": "small", "description": "d"}
NONE_CTX = {**ALERT_CTX, "source_type": "fog_dust_cloud", "smoke_color": "none"}


def cloud_frames(sevs, fps=2.0, nbytes=150_000, lat=800.0):
    return [{"i": i, "t": i / fps, "bytes": nbytes, "severity": s, "ctx": ALERT_CTX if s == "ALERT" else NONE_CTX,
             "latency_ms": lat, "prompt_tokens": 1200, "completion_tokens": 60} for i, s in enumerate(sevs)]


EDGE = {"name": "after", "config": {"fps": 2.0, "recheck_s": 5.0},
        "clips": [edge_clip("fire", "alert", 1.0), edge_clip("fog", "no_alert", None)]}
CLOUD = {"name": "cloud_qwen7b", "model": "Qwen/Qwen2.5-VL-7B-Instruct", "fps": 2.0, "alert_on": "alert",
         "net_baseline_ms": 0.0, "persist": 3, "recheck_s": 5.0,
         "clips": [{"path": "fire", "frames": cloud_frames(["IGNORE"] * 2 + ["ALERT"] * 38)},
                   {"path": "fog", "frames": cloud_frames(["IGNORE"] * 40)}]}


def test_constants_come_from_the_shipped_code():
    assert MAX_BACKOFF_S == 10 and FLUSH_EVERY_S == 2.0 and EDGE_POST_RTTS == 3


def test_edge_retry_schedule_follows_the_outbox_backoff():
    assert edge_delivery_attempt_s(3.0, 0) == 3.0            # link up: sent at the decision
    assert edge_delivery_attempt_s(3.0, 10) == 3 + 2 + 4 + 8  # retries at +2, +4, +8
    assert edge_delivery_attempt_s(3.0, 600) == pytest.approx(17 + 10 * 59)  # then every 10 s
    assert edge_delivery_attempt_s(3.0, 600) - 600 < MAX_BACKOFF_S


def test_edge_knows_adds_flush_wait_and_a_new_https_connection():
    clip = edge_clip("f", "alert", 1.0)
    up = transfer_s(20_000, FIBER, rtts=EDGE_POST_RTTS)
    assert edge_knows_s(clip, FIBER, 0) == pytest.approx(1.0 + 2.0 + FLUSH_EVERY_S / 2 + up)
    assert edge_knows_s(clip, FIBER, 600) == pytest.approx(edge_delivery_attempt_s(3.0, 600) + FLUSH_EVERY_S / 2 + up)
    assert edge_knows_s(edge_clip("f", "alert", None), FIBER, 0) is None
    assert edge_knows_s(clip, PROFILES["outage"], 0) is None


def test_live_after_restore_holds_the_clip_tail_when_nothing_was_recorded_after_it():
    frames = cloud_frames(["IGNORE"] * 30 + ["ALERT"] * 10)
    live, modelled = live_after_restore(frames, 2.0, 1, 600, tail=4)
    assert modelled and [f["t"] for f in live] == [600, 600.5, 601, 601.5]
    assert [f["i"] for f in live] == [36, 37, 38, 39] and all(f["modelled"] for f in live)
    real, modelled = live_after_restore(frames, 2.0, 1, 10, tail=4)
    assert not modelled and real[0]["t"] == 10 and len(real) == 20


def test_long_outage_edge_vs_cloud_live_and_buffer():
    res = scenario(EDGE, CLOUD, [FIBER, SAT], outage_min=10)
    fire = res["clips"][0]
    assert fire["edge_frames_analysed_during_outage"] == 40 and fire["cloud_live_modelled"]
    f = fire["profiles"]["fiber"]
    # live after restore: 10 held ALERT frames from t=600; first one decides
    assert f["cloud_live_knows_s"] == pytest.approx(600 + serialize_s(150_000, FIBER) + rtt_s(FIBER) + 0.8)
    assert f["delta_live_minus_edge_s"] == pytest.approx(f["cloud_live_knows_s"] - f["edge_knows_s"])
    assert abs(f["delta_live_minus_edge_s"]) < 15  # after a clean outage on fiber: seconds apart
    up = serialize_s(150_000, FIBER)
    assert f["cloud_buffer_knows_s"] == pytest.approx(600 + 3 * up + rtt_s(FIBER) + 0.8)  # 3rd frame: first ALERT
    assert f["cloud_buffer_queued_frames"] == 1200  # the camera keeps capturing for all 10 min
    assert f["cloud_recorded_drop_knows_s"] is None
    assert f["cloud_live_temporal_knows_s"] > f["cloud_live_knows_s"]  # needs 3 consecutive answers
    s = res["profiles"]["satellite_geo"]
    assert s["edge_known"] == 1 and s["cloud_live_known"] == 1 and s["cloud_recorded_drop_known"] == 0
    assert s["paired_n"] == 1 and s["paired_delta_s_p50"] > res["profiles"]["fiber"]["paired_delta_s_p50"]
    assert s["edge_frames_analysed_during_outage"] == 80 and s["cloud_decisions_during_outage"] == 0
    assert s["edge_bytes_up"] == 20_000 and s["cloud_buffer_bytes_up"] == 2 * 1200 * 150_000
    assert s["cloud_live_bytes_up"] > 0 and res["edge_recheck_s"] == 5.0
    json.dumps(res, allow_nan=False)


def test_backlog_from_before_the_fire_is_uploaded_first():
    res = scenario(EDGE, CLOUD, [FIBER], outage_min=10, lead_min=1)
    f = res["clips"][0]["profiles"]["fiber"]
    up = serialize_s(150_000, FIBER)
    assert f["cloud_buffer_knows_s"] == pytest.approx(600 + 123 * up + rtt_s(FIBER) + 0.8)
    assert f["cloud_buffer_queued_frames"] == 1320
    assert "for 1 min before it" in markdown(res)


def test_with_backlog_shapes_the_queue():
    q = with_backlog(cloud_frames(["ALERT"] * 4), fps=2.0, stride=1, lead_s=1.0, outage_end=4.0)
    assert [f["t"] for f in q] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    assert [f["severity"] for f in q] == [None, None, "ALERT", "ALERT", "ALERT", "ALERT", None, None, None, None]
    assert q[0]["bytes"] == 150_000 and q[0]["latency_ms"] == 800.0 and q[0]["modelled"]


def test_no_outage_compares_links_only():
    res = scenario(EDGE, CLOUD, [FIBER], outage_min=0)
    f = res["clips"][0]["profiles"]["fiber"]
    assert res["outage_window_s"] is None and res["clips"][0]["edge_frames_analysed_during_outage"] == 0
    assert not res["clips"][0]["cloud_live_modelled"]
    assert f["cloud_live_knows_s"] == pytest.approx(1.0 + serialize_s(150_000, FIBER) + rtt_s(FIBER) + 0.8)


def test_net_baseline_is_subtracted_from_cloud_latency():
    res = scenario(EDGE, {**CLOUD, "net_baseline_ms": 300.0}, [FIBER], outage_min=0)
    f = res["clips"][0]["profiles"]["fiber"]
    assert f["cloud_live_knows_s"] == pytest.approx(1.0 + serialize_s(150_000, FIBER) + rtt_s(FIBER) + 0.5)


def test_rejects_old_results_mismatched_fps_and_warns_on_unmatched_clips():
    old = {"name": "before", "clips": [{"path": "fire", "label": "alert", "frames": 40}]}
    with pytest.raises(SystemExit, match="re-run"):
        scenario(old, CLOUD, [FIBER], 10)
    with pytest.raises(SystemExit, match="both"):
        scenario({"name": "x", "clips": [edge_clip("other", "alert", 1.0)]}, CLOUD, [FIBER], 10, warn=lambda m: None)
    with pytest.raises(SystemExit, match="fps"):
        scenario({**EDGE, "config": {"fps": 1.0}}, CLOUD, [FIBER], 10)
    warnings = []
    extra = {**CLOUD, "clips": CLOUD["clips"] + [{"path": "extra", "frames": cloud_frames(["IGNORE"])}]}
    res = scenario(EDGE, extra, [FIBER], 10, warn=warnings.append)
    assert res["unmatched_clips"] == ["extra"] and "1 clip(s)" in warnings[0]


def test_markdown_tells_the_honest_story():
    md = markdown(scenario(EDGE, CLOUD, [FIBER, SAT], 10))
    assert "| fiber |" in md and "| satellite_geo |" in md
    assert "Frames analysed during outage" in md and "Cloud live after restore" in md
    assert "difference in when dispatch knows is seconds" in md and "Footnote" in md
    assert "recheck 5 s" in md


def test_main_writes_json_and_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "bench_after.json").write_text(json.dumps(EDGE))
    (tmp_path / "results" / "bench_cloud_cloud_qwen7b.json").write_text(json.dumps(CLOUD))
    main(["--profiles", "fiber,lte", "--outage-min", "5"])
    out = json.loads((tmp_path / "results" / "outage_after_vs_cloud_qwen7b_5min.json").read_text())
    assert set(out["profiles"]) == {"fiber", "lte"} and out["outage_window_s"] == [0.0, 300.0]
    assert (tmp_path / "results" / "outage_after_vs_cloud_qwen7b_5min.md").exists()
    with pytest.raises(SystemExit):
        main(["--profiles", "fiber", "--outage-min", "5"])

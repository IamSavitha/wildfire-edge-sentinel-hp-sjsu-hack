import json

import pytest

from scripts.outage_scenario import edge_knows_s, main, markdown, scenario
from sentinel.netprofile import PROFILES, transfer_s

FIBER, SAT = PROFILES["fiber"], PROFILES["satellite_geo"]


def edge_clip(path, label, first_alert_s, frames=40, compute_ms=2000.0, nbytes=20_000):
    return {"path": path, "label": label, "frames": frames, "first_alert_s": first_alert_s,
            "first_alert_compute_ms": compute_ms if first_alert_s is not None else None,
            "alert_payload_bytes": nbytes if first_alert_s is not None else None}


def cloud_frames(sevs, fps=2.0, nbytes=150_000, lat=800.0):
    return [{"i": i, "t": i / fps, "bytes": nbytes, "severity": s, "latency_ms": lat,
             "prompt_tokens": 1200, "completion_tokens": 60} for i, s in enumerate(sevs)]


EDGE = {"name": "after", "config": {"fps": 2.0},
        "clips": [edge_clip("fire", "alert", 1.0), edge_clip("fog", "no_alert", None)]}
CLOUD = {"name": "cloud_qwen7b", "fps": 2.0, "alert_on": "alert",
         "clips": [{"path": "fire", "frames": cloud_frames(["IGNORE"] * 2 + ["ALERT"] * 38)},
                   {"path": "fog", "frames": cloud_frames(["IGNORE"] * 40)}]}


def test_edge_knows_after_the_link_returns_plus_alert_upload():
    clip = edge_clip("f", "alert", 1.0)
    assert edge_knows_s(clip, FIBER, 600) == pytest.approx(600 + transfer_s(20_000, FIBER))
    assert edge_knows_s(clip, FIBER, 0) == pytest.approx(1.0 + 2.0 + transfer_s(20_000, FIBER))
    assert edge_knows_s(edge_clip("f", "alert", None), FIBER, 0) is None
    assert edge_knows_s(clip, PROFILES["outage"], 0) is None


def test_long_outage_edge_knows_cloud_drop_blind_cloud_buffer_drains_backlog():
    res = scenario(EDGE, CLOUD, [FIBER, SAT], outage_min=10)
    fire = res["clips"][0]
    assert fire["edge_decisions_during_outage"] == 40
    f = fire["profiles"]["fiber"]
    assert f["edge_knows_s"] == pytest.approx(600 + transfer_s(20_000, FIBER))
    assert f["cloud_drop_knows_s"] is None and f["cloud_drop_frames_dropped"] == 40
    up = transfer_s(150_000, FIBER)
    assert f["cloud_buffer_knows_s"] == pytest.approx(600 + 3 * up + 0.8)  # third frame is the first ALERT
    s = res["profiles"]["satellite_geo"]
    assert s["cloud_buffer_knows_s_p50"] > res["profiles"]["fiber"]["cloud_buffer_knows_s_p50"]
    assert s["edge_known"] == 1 and s["cloud_drop_known"] == 0 and s["cloud_buffer_known"] == 1
    assert s["edge_decisions_during_outage"] == 80 and s["cloud_drop_decisions_during_outage"] == 0
    # the camera keeps capturing for all 10 min at 2 fps: 1200 frames per clip must drain after the outage
    assert f["cloud_buffer_queued_frames"] == 1200
    assert s["edge_bytes_up"] == 20_000 and s["cloud_buffer_bytes_up"] == 2 * 1200 * 150_000
    json.dumps(res, allow_nan=False)


def test_backlog_from_before_the_fire_is_uploaded_first():
    res = scenario(EDGE, CLOUD, [FIBER], outage_min=10, lead_min=1)
    f = res["clips"][0]["profiles"]["fiber"]
    up = transfer_s(150_000, FIBER)
    # 120 modelled frames from the minute before the clip drain ahead of the clip's third frame
    assert f["cloud_buffer_knows_s"] == pytest.approx(600 + 123 * up + 0.8)
    assert f["cloud_buffer_queued_frames"] == 1320
    assert f["edge_knows_s"] == pytest.approx(600 + transfer_s(20_000, FIBER))
    assert "for 1 min before it" in markdown(res)


def test_with_backlog_shapes_the_queue():
    from scripts.outage_scenario import with_backlog
    q = with_backlog(cloud_frames(["ALERT"] * 4), fps=2.0, stride=1, lead_s=1.0, outage_end=4.0)
    assert [f["t"] for f in q] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    assert [f["severity"] for f in q] == [None, None, "ALERT", "ALERT", "ALERT", "ALERT", None, None, None, None]
    assert q[0]["bytes"] == 150_000 and q[0]["latency_ms"] == 800.0 and q[0]["modelled"]


def test_no_outage_compares_links_only():
    res = scenario(EDGE, CLOUD, [FIBER], outage_min=0)
    f = res["clips"][0]["profiles"]["fiber"]
    assert res["outage_window_s"] is None and res["clips"][0]["edge_decisions_during_outage"] == 0
    assert f["cloud_drop_knows_s"] == f["cloud_buffer_knows_s"]
    assert f["cloud_drop_knows_s"] == pytest.approx(1.0 + transfer_s(150_000, FIBER) + 0.8)


def test_older_edge_results_without_first_alert_are_rejected():
    old = {"name": "before", "clips": [{"path": "fire", "label": "alert", "frames": 40}]}
    with pytest.raises(SystemExit, match="re-run"):
        scenario(old, CLOUD, [FIBER], 10)
    with pytest.raises(SystemExit, match="both"):
        scenario({"name": "x", "clips": [edge_clip("other", "alert", 1.0)]}, CLOUD, [FIBER], 10)


def test_markdown_table_has_a_row_per_profile():
    md = markdown(scenario(EDGE, CLOUD, [FIBER, SAT], 10))
    assert "| fiber |" in md and "| satellite_geo |" in md and "no decision [0/1]" in md


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

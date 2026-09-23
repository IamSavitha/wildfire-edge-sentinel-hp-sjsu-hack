from scripts.cost_model import compare

P = {"towers": 1, "fps": 1, "tokens_per_full_frame": 1000, "bytes_per_frame": 1_000_000,
     "usd_per_mtok": 1.0, "usd_per_gb": 10.0, "local_vlm_s": 1.0, "detector_s": 0.01,
     "events_per_day": 10, "vlm_calls_per_event": 2, "alerts_per_day": 1, "bytes_per_alert": 50_000}


def test_cloud_every_frame():
    row = compare(P)[0]
    assert row["vlm_calls"] == 86_400 and row["cloud_tokens"] == 86_400_000
    assert row["usd_per_day"] == 86.4 + 864.0


def test_local_every_frame_is_infeasible_at_one_second_per_call():
    row = compare(P)[1]
    assert row["gpu_seconds"] == 86_400 and row["feasible"] is False


def test_cascade():
    row = compare(P)[2]
    assert row["vlm_calls"] == 20 and row["cloud_tokens"] == 0
    assert row["upstream_gb"] == 0.00005 and row["feasible"] is True


def test_shipped_inputs_still_run_and_carry_split_prices():
    import json
    from pathlib import Path
    params = json.loads((Path(__file__).parents[1] / "config" / "cost_inputs.json").read_text())
    assert {"usd_per_mtok", "usd_per_mtok_in", "usd_per_mtok_out"} <= params.keys()
    assert len(compare(params)) == 3

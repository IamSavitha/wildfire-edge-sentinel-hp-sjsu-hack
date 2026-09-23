import json

import pytest

from sentinel.live_metrics import (discover_models, economics, histogram_quantile, model_snapshot,
                                   parse_prometheus, rates, system_stats)

INF = float("inf")

METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="base7b"} 1.0
vllm:kv_cache_usage_perc{engine="0",model_name="base7b"} 0.25
vllm:prompt_tokens_total{engine="0",model_name="base7b"} 825.0
vllm:generation_tokens_total{engine="0",model_name="base7b"} 259.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="1.0",model_name="base7b"} 1.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="2.0",model_name="base7b"} 3.0
vllm:e2e_request_latency_seconds_bucket{engine="0",le="+Inf",model_name="base7b"} 4.0
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="base7b"} 6.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="base7b"} 4.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="base7b"} 2.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.2",model_name="base7b"} 4.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="base7b"} 4.0
vllm:prompt_tokens_total{engine="0",model_name="other"} 99.0

"""


# ---- parse_prometheus ----

def test_parse_skips_comments_and_blanks():
    samples = parse_prometheus(METRICS)
    assert len(samples) == 13
    assert samples[0] == ("vllm:num_requests_running", {"engine": "0", "model_name": "base7b"}, 1.0)


def test_parse_handles_inf_no_labels_timestamps_and_escapes():
    text = ('up 1\n'
            'x_bucket{le="+Inf"} 3 1700000000000\n'
            'y{path="a\\"b,c}",k="v"} -2.5e1\n')
    assert parse_prometheus(text) == [
        ("up", {}, 1.0),
        ("x_bucket", {"le": "+Inf"}, 3.0),
        ("y", {"path": 'a"b,c}', "k": "v"}, -25.0),
    ]


def test_parse_drops_nan_and_malformed_lines():
    text = 'a NaN\nb{x="1"} 2\nthis is not a sample\nc{unterminated="x 1\nd not_a_number\n'
    assert parse_prometheus(text) == [("b", {"x": "1"}, 2.0)]


# ---- histogram_quantile ----

def test_quantile_interpolates_linearly():
    buckets = [(1.0, 1.0), (2.0, 3.0), (INF, 4.0)]
    assert histogram_quantile(buckets, 0.5) == pytest.approx(1.5)  # rank 2 in (1,2], 1 of 2
    assert histogram_quantile(buckets, 0.25) == pytest.approx(1.0)


def test_quantile_first_bucket_starts_at_zero():
    assert histogram_quantile([(4.0, 2.0), (INF, 2.0)], 0.5) == pytest.approx(2.0)


def test_quantile_in_inf_bucket_returns_highest_finite_bound():
    assert histogram_quantile([(1.0, 1.0), (2.0, 2.0), (INF, 10.0)], 0.95) == 2.0


def test_quantile_unsorted_input_and_empty():
    assert histogram_quantile([(INF, 4.0), (2.0, 3.0), (1.0, 1.0)], 0.5) == pytest.approx(1.5)
    assert histogram_quantile([(1.0, 0.0), (INF, 0.0)], 0.5) is None
    assert histogram_quantile([], 0.5) is None
    assert histogram_quantile([(1.0, 1.0), (INF, 1.0)], 1.5) is None


# ---- model_snapshot ----

def test_model_snapshot_reads_counters_gauges_and_histograms():
    s = model_snapshot(parse_prometheus(METRICS), "base7b")
    assert s["requests"] == 4.0 and s["prompt_tokens"] == 825.0 and s["generation_tokens"] == 259.0
    assert s["running"] == 1.0 and s["kv_cache_pct"] == pytest.approx(25.0)
    assert s["latency_p50_s"] == pytest.approx(1.5)
    assert s["latency_p95_s"] == 2.0          # rank 3.8 lands in +Inf -> highest finite bound
    assert s["ttft_p50_s"] == pytest.approx(0.1)


def test_model_snapshot_missing_metrics_are_none():
    s = model_snapshot(parse_prometheus('vllm:prompt_tokens_total{model_name="m"} 5\n'), "m")
    assert s == {"requests": None, "prompt_tokens": 5.0, "generation_tokens": None, "running": None,
                 "kv_cache_pct": None, "latency_p50_s": None, "latency_p95_s": None, "ttft_p50_s": None}


def test_model_snapshot_none_name_takes_all_and_legacy_names():
    text = ('vllm:gpu_cache_usage_perc{model_name="x"} 0.5\n'
            'vllm:request_success_total{finished_reason="stop",model_name="x"} 3\n'
            'vllm:request_success_total{finished_reason="length",model_name="x"} 1\n')
    s = model_snapshot(parse_prometheus(text), None)
    assert s["kv_cache_pct"] == pytest.approx(50.0) and s["requests"] == 4.0


# ---- rates ----

SNAP0 = {"prompt_tokens": 100.0, "generation_tokens": 10.0, "requests": 1.0}
SNAP1 = {"prompt_tokens": 300.0, "generation_tokens": 50.0, "requests": 3.0}


def test_rates_between_two_snapshots():
    assert rates(SNAP0, SNAP1, 2.0) == {"prompt_tok_per_s": 100.0, "gen_tok_per_s": 20.0,
                                         "req_per_s": 1.0}


def test_rates_first_sample_bad_dt_and_reset_are_none():
    none = {"prompt_tok_per_s": None, "gen_tok_per_s": None, "req_per_s": None}
    assert rates(None, SNAP1, 2.0) == none
    assert rates(SNAP0, SNAP1, 0.0) == none
    r = rates(SNAP1, {"prompt_tokens": 5.0, "generation_tokens": 60.0, "requests": None}, 1.0)
    assert r == {"prompt_tok_per_s": None, "gen_tok_per_s": 10.0, "req_per_s": None}


# ---- economics ----

PRICES = {"usd_per_mtok_in": 2.0, "usd_per_mtok_out": 10.0, "usd_per_gb": 0.5,
          "tokens_per_full_frame": 1000, "bytes_per_frame": 100_000}


def test_economics_per_model_and_total():
    models = {"a": {"prompt_tokens": 1_000_000.0, "generation_tokens": 100_000.0},
              "b": {"prompt_tokens": 500_000.0, "generation_tokens": None}}
    e = economics(models, None, PRICES)
    assert e["models"]["a"] == {"prompt_tokens": 1_000_000.0, "generation_tokens": 100_000.0,
                                "edge_tokens": 1_100_000.0, "cloud_equiv_usd": pytest.approx(3.0)}
    assert e["models"]["b"]["cloud_equiv_usd"] == pytest.approx(1.0)
    assert e["total"]["edge_tokens"] == 1_600_000.0
    assert e["total"]["cloud_equiv_usd"] == pytest.approx(4.0)
    assert e["pipeline"] is None


def test_economics_falls_back_to_single_price():
    e = economics({"a": {"prompt_tokens": 1e6, "generation_tokens": 1e6}}, None, {"usd_per_mtok": 3})
    assert e["total"]["cloud_equiv_usd"] == pytest.approx(6.0)


def test_economics_pipeline_block():
    pm = {"frames": 1000, "vlm_calls": 4, "bytes_up": 2_000, "vlm_tokens": 1200}
    p = economics({}, pm, PRICES)["pipeline"]
    assert p["frames_seen"] == 1000 and p["vlm_calls"] == 4 and p["frames_per_vlm_call"] == 250.0
    assert p["cloud_per_frame_usd"] == pytest.approx(2.0)          # 1e6 tok * $2/M
    assert p["cascade_calls_avoided"] == 996
    assert p["bytes_up"] == 2_000 and p["video_bytes_equiv"] == 100_000_000
    assert p["upstream_saving_ratio"] == pytest.approx(1 - 2_000 / 100_000_000)
    assert p["video_upload_usd"] == pytest.approx(0.05)
    assert p["edge_uplink_usd"] == pytest.approx(0.000001)


def test_economics_zero_prices_and_zero_counts_never_divide():
    e = economics({"a": {"prompt_tokens": 5.0, "generation_tokens": 5.0}}, {}, {})
    assert e["total"]["cloud_equiv_usd"] == 0.0
    p = e["pipeline"]
    assert p["frames_seen"] == 0 and p["frames_per_vlm_call"] is None
    assert p["cloud_per_frame_usd"] == 0.0 and p["upstream_saving_ratio"] is None


# ---- discover_models ----

def test_discover_models(tmp_path):
    (tmp_path / "vllm-base7b.json").write_text(json.dumps({
        "model_uri": "hf:Qwen/Qwen2.5-VL-7B-Instruct@main", "label": "base7b",
        "gpu_memory_fraction": 0.45, "started_at": "2026-09-23T13:01:42+07:00", "type": "vllm"}))
    (tmp_path / "vllm-base7b.sock").write_text("")
    (tmp_path / "vllm-broken.json").write_text("{not json")
    (tmp_path / "other.json").write_text("{}")
    models = discover_models(tmp_path)
    assert [m["label"] for m in models] == ["base7b", "broken"]
    assert models[0]["model_uri"] == "hf:Qwen/Qwen2.5-VL-7B-Instruct@main"
    assert models[0]["gpu_memory_fraction"] == 0.45
    assert models[0]["sock"] == str(tmp_path / "vllm-base7b.sock")
    assert models[1]["model_uri"] is None and models[1]["sock"] is None


def test_discover_models_missing_dir(tmp_path):
    assert discover_models(tmp_path / "nope") == []


# ---- system_stats ----

MEMINFO = "MemTotal:       131072000 kB\nMemFree:  1000 kB\nMemAvailable:    31072000 kB\n"


def test_system_stats_reads_meminfo_and_smi(tmp_path):
    p = tmp_path / "meminfo"
    p.write_text(MEMINFO)
    s = system_stats(p, run=lambda cmd: "37\n")
    assert s["gpu_util_pct"] == 37.0
    assert s["mem_total_gb"] == pytest.approx(131072000 * 1024 / 1e9)
    assert s["mem_used_gb"] == pytest.approx(100000000 * 1024 / 1e9)


def test_system_stats_unavailable(tmp_path):
    def boom(cmd):
        raise FileNotFoundError("nvidia-smi")

    assert system_stats(tmp_path / "none", run=boom) == {
        "gpu_util_pct": None, "mem_used_gb": None, "mem_total_gb": None}
    assert system_stats(tmp_path / "none", run=lambda c: "[N/A]")["gpu_util_pct"] is None

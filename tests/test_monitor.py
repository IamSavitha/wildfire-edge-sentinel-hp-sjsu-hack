import json

import pytest
from fastapi.testclient import TestClient

from sentinel.monitor import create_monitor_app


def prom(model, prompt, gen, reqs, running=0):
    return (f'vllm:num_requests_running{{engine="0",model_name="{model}"}} {running}\n'
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="{model}"}} 0.1\n'
            f'vllm:prompt_tokens_total{{engine="0",model_name="{model}"}} {prompt}\n'
            f'vllm:generation_tokens_total{{engine="0",model_name="{model}"}} {gen}\n'
            f'vllm:e2e_request_latency_seconds_count{{engine="0",model_name="{model}"}} {reqs}\n')


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for label, uri in (("base7b", "hf:Qwen/Qwen2.5-VL-7B-Instruct@main"), ("lora", "hf:x/y@main")):
        (run / f"vllm-{label}.json").write_text(json.dumps(
            {"model_uri": uri, "label": label, "gpu_memory_fraction": 0.3, "started_at": "t0"}))
        (run / f"vllm-{label}.sock").write_text("")
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"usd_per_mtok": 0, "usd_per_mtok_in": 0, "usd_per_mtok_out": 0,
                                  "usd_per_gb": 0, "tokens_per_full_frame": 1000,
                                  "bytes_per_frame": 50_000}))
    texts = {str(run / "vllm-base7b.sock"): prom("base7b", 1000, 100, 2)}

    def fetch(sock):
        if sock not in texts:
            raise ConnectionError("refused")
        return texts[sock]

    clock = Clock()
    app = create_monitor_app(run, prices, fetch=fetch, clock=clock,
                             system=lambda: {"gpu_util_pct": 12.0, "mem_used_gb": 40.0,
                                             "mem_total_gb": 128.0})
    return TestClient(app), texts, clock, run


def test_index_serves_monitor_page(env):
    client, *_ = env
    r = client.get("/")
    assert r.status_code == 200 and "Live Model Economics" in r.text


def test_live_lists_models_and_marks_unreachable(env):
    client, *_ = env
    live = client.get("/api/live").json()
    assert live["ts"] == 1000.0 and live["system"]["gpu_util_pct"] == 12.0
    rows = {m["label"]: m for m in live["models"]}
    ok, bad = rows["base7b"], rows["lora"]
    assert ok["status"] == "ok" and ok["model_uri"].startswith("hf:Qwen/")
    assert ok["prompt_tokens"] == 1000.0 and ok["requests"] == 2.0 and ok["kv_cache_pct"] == 10.0
    assert ok["gpu_memory_fraction"] == 0.3 and ok["prompt_tok_per_s"] is None
    assert bad["status"] == "unreachable" and bad["prompt_tokens"] is None and "refused" in bad["error"]
    assert live["pipeline"] is None and live["economics"]["pipeline"] is None
    assert live["economics"]["total"]["edge_tokens"] == 1100.0
    assert live["economics"]["total"]["cloud_equiv_usd"] == 0.0
    assert live["prices"]["usd_per_mtok_in"] == 0.0


def test_rates_across_polls_with_fake_clock(env):
    client, texts, clock, run = env
    client.get("/api/live")
    clock.t += 2.0
    texts[str(run / "vllm-base7b.sock")] = prom("base7b", 1400, 160, 4)
    row = client.get("/api/live").json()["models"][0]
    assert row["prompt_tok_per_s"] == 200.0 and row["gen_tok_per_s"] == 30.0 and row["req_per_s"] == 1.0
    # a second viewer polling immediately reuses the last rates instead of a ~0 s window
    clock.t += 0.1
    texts[str(run / "vllm-base7b.sock")] = prom("base7b", 1500, 160, 4)
    row = client.get("/api/live").json()["models"][0]
    assert row["prompt_tok_per_s"] == 200.0 and row["prompt_tokens"] == 1500.0


def test_model_name_label_mismatch_falls_back_to_whole_backend(env):
    client, texts, _, run = env
    texts[str(run / "vllm-base7b.sock")] = prom("Qwen/Qwen2.5-VL-7B-Instruct", 7, 3, 1)
    row = client.get("/api/live").json()["models"][0]
    assert row["prompt_tokens"] == 7.0 and row["generation_tokens"] == 3.0


def test_unreachable_model_keeps_last_known_tokens_in_economics(env):
    client, texts, _, run = env
    client.get("/api/live")
    texts.pop(str(run / "vllm-base7b.sock"))
    live = client.get("/api/live").json()
    assert live["models"][0]["status"] == "unreachable"
    assert live["economics"]["total"]["edge_tokens"] == 1100.0


def test_prices_override(env):
    client, *_ = env
    r = client.post("/api/prices", json={"usd_per_mtok_in": 2.0, "usd_per_mtok_out": 10.0})
    assert r.status_code == 200 and r.json()["usd_per_mtok_in"] == 2.0
    econ = client.get("/api/live").json()["economics"]
    assert econ["total"]["cloud_equiv_usd"] == pytest.approx(1000 / 1e6 * 2 + 100 / 1e6 * 10)
    assert client.post("/api/prices", json={"usd_per_gb": -1}).status_code == 422
    assert client.post("/api/prices", json={"usd_per_gb": "abc"}).status_code == 422
    assert client.post("/api/prices", json={"surprise": 1}).status_code == 422


def test_pipeline_metrics_and_pipeline_errors(tmp_path, env):
    _, _, _, run = env
    state = {"metrics": {"frames": 200, "vlm_calls": 2, "bytes_up": 1500, "vlm_tokens": 600}}
    app = create_monitor_app(run, tmp_path / "prices.json", pipeline_url="http://x/api/state",
                             fetch=lambda s: "", fetch_pipeline=lambda url: state,
                             system=lambda: {})
    live = TestClient(app).get("/api/live").json()
    assert live["pipeline"]["frames"] == 200
    p = live["economics"]["pipeline"]
    assert p["frames_per_vlm_call"] == 100.0 and p["cascade_calls_avoided"] == 198
    assert p["video_bytes_equiv"] == 200 * 50_000

    def boom(url):
        raise ConnectionError("down")

    app = create_monitor_app(run, tmp_path / "prices.json", pipeline_url="http://x/api/state",
                             fetch=lambda s: "", fetch_pipeline=boom, system=lambda: {})
    live = TestClient(app).get("/api/live").json()
    assert live["pipeline"] is None and live["economics"]["pipeline"] is None


def test_missing_prices_file_and_run_dir(tmp_path):
    app = create_monitor_app(tmp_path / "nope", tmp_path / "none.json", fetch=lambda s: "",
                             system=lambda: {})
    live = TestClient(app).get("/api/live").json()
    assert live["models"] == [] and live["economics"]["total"]["cloud_equiv_usd"] == 0.0

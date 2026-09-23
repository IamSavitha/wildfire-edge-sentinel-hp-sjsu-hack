import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from sentinel.monitor import create_monitor_app


def prom(model, prompt, gen, reqs, running=0, lat=None):
    """lat: cumulative (le, count) buckets for e2e latency; defaults to all requests <= 1 s."""
    lat = lat if lat is not None else [("1.0", reqs), ("+Inf", reqs)]
    text = (f'vllm:num_requests_running{{engine="0",model_name="{model}"}} {running}\n'
            f'vllm:kv_cache_usage_perc{{engine="0",model_name="{model}"}} 0.1\n'
            f'vllm:prompt_tokens_total{{engine="0",model_name="{model}"}} {prompt}\n'
            f'vllm:generation_tokens_total{{engine="0",model_name="{model}"}} {gen}\n'
            f'vllm:e2e_request_latency_seconds_count{{engine="0",model_name="{model}"}} {reqs}\n')
    for le, c in lat:
        text += f'vllm:e2e_request_latency_seconds_bucket{{engine="0",le="{le}",model_name="{model}"}} {c}\n'
    return text


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


SYSTEM = {"gpu_util_pct": 12.0, "mem_used_gb": 40.0, "mem_total_gb": 128.0}


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
    sock = str(run / "vllm-base7b.sock")
    texts = {sock: prom("base7b", 1000, 100, 2)}
    calls = []

    def fetch(path):
        calls.append(path)
        if path not in texts:
            raise ConnectionError("refused")
        return texts[path]

    clock = Clock()
    app = create_monitor_app(run, prices, fetch=fetch, clock=clock, system=lambda: SYSTEM)
    return TestClient(app), app.state.monitor, texts, clock, sock, calls


def base_row(payload):
    return next(m for m in payload["models"] if m["label"] == "base7b")


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
    assert ok["latency_window"] == "since_start" and ok["latency_p50_s"] == pytest.approx(0.5)
    assert bad["status"] == "unreachable" and bad["prompt_tokens"] is None and "refused" in bad["error"]
    assert live["pipeline"] is None and live["economics"]["pipeline"] is None
    assert live["economics"]["total"]["edge_tokens"] == 1100.0
    assert live["economics"]["total"]["cloud_equiv_usd"] == 0.0
    assert live["prices"]["usd_per_mtok_in"] == 0.0


def test_live_serves_stored_payload_without_refetching(env):
    client, mon, texts, clock, sock, calls = env
    client.get("/api/live")
    n = len(calls)
    for _ in range(3):
        client.get("/api/live")
    assert len(calls) == n
    clock.t += 2.0
    mon.poll_once()
    assert client.get("/api/live").json()["ts"] == 1002.0


def test_rates_across_polls_with_fake_clock(env):
    client, mon, texts, clock, sock, _ = env
    mon.poll_once()
    clock.t += 2.0
    texts[sock] = prom("base7b", 1400, 160, 4)
    mon.poll_once()
    row = base_row(client.get("/api/live").json())
    assert row["prompt_tok_per_s"] == 200.0 and row["gen_tok_per_s"] == 30.0 and row["req_per_s"] == 1.0


def test_latency_is_windowed_then_held_then_survives_restart(env):
    client, mon, texts, clock, sock, _ = env
    texts[sock] = prom("base7b", 100, 10, 10, lat=[("1.0", 10), ("4.0", 10), ("+Inf", 10)])
    mon.poll_once()
    clock.t += 2.0      # two new requests, both in (1, 4]
    texts[sock] = prom("base7b", 200, 20, 12, lat=[("1.0", 10), ("4.0", 12), ("+Inf", 12)])
    row = base_row(mon.poll_once())
    assert row["latency_window"] == "window" and row["latency_p50_s"] == pytest.approx(2.5)
    clock.t += 2.0      # idle: same counters -> keep the last windowed value
    row = base_row(mon.poll_once())
    assert row["latency_window"] == "held" and row["latency_p50_s"] == pytest.approx(2.5)
    clock.t += 2.0      # restart: counters and buckets reset
    texts[sock] = prom("base7b", 50, 5, 1, lat=[("1.0", 1), ("4.0", 1), ("+Inf", 1)])
    live = mon.poll_once()
    row = base_row(live)
    assert row["latency_window"] == "held" and row["latency_p50_s"] == pytest.approx(2.5)
    assert row["prompt_tok_per_s"] is None                       # no rate across a reset
    assert row["prompt_tokens"] == 250.0 and row["requests"] == 13.0   # totals never decrease
    assert live["economics"]["total"]["edge_tokens"] == 250.0 + 25.0


def test_hung_sockets_are_fetched_in_parallel_and_time_out(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for i in range(4):
        (run / f"vllm-m{i}.json").write_text(json.dumps({"label": f"m{i}"}))
        (run / f"vllm-m{i}.sock").write_text("")
    release = threading.Event()

    def fetch(path):
        if path.endswith("m0.sock"):
            return prom("m0", 1, 1, 1)
        release.wait(5)          # three hung backends
        return ""

    app = create_monitor_app(run, tmp_path / "none.json", fetch=fetch, system=lambda: {},
                             fetch_timeout_s=0.3)
    try:
        t0 = time.monotonic()
        live = app.state.monitor.poll_once()
        assert time.monotonic() - t0 < 1.0            # serial fetching would block ~5 s per hung socket
        status = {m["label"]: m["status"] for m in live["models"]}
        assert status == {"m0": "ok", "m1": "unreachable", "m2": "unreachable", "m3": "unreachable"}
        assert "TimeoutError" in live["models"][1]["error"]
    finally:
        release.set()
        app.state.monitor.stop()


def test_background_poller_starts_with_app_and_stops(env):
    run_calls = []

    def fetch(path):
        run_calls.append(path)
        return prom("base7b", 1, 1, 1)

    app = create_monitor_app(env[1].run_dir, "none.json", fetch=fetch, system=lambda: {},
                             interval_s=0.05)
    with TestClient(app) as client:
        deadline = time.monotonic() + 3
        while len(run_calls) < 4 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(run_calls) >= 4
        assert client.get("/api/live").json()["models"]
    thread = app.state.monitor._thread
    assert thread is not None and not thread.is_alive()


def test_model_name_label_mismatch_falls_back_to_whole_backend(env):
    client, _, texts, _, sock, _ = env
    texts[sock] = prom("Qwen/Qwen2.5-VL-7B-Instruct", 7, 3, 1)
    row = base_row(client.get("/api/live").json())
    assert row["prompt_tokens"] == 7.0 and row["generation_tokens"] == 3.0


def test_unreachable_model_keeps_last_known_tokens_in_economics(env):
    _, mon, texts, clock, sock, _ = env
    mon.poll_once()
    texts.pop(sock)
    clock.t += 2.0
    live = mon.poll_once()
    assert base_row(live)["status"] == "unreachable"
    assert live["economics"]["total"]["edge_tokens"] == 1100.0


def test_prices_override(env):
    client, *_ = env
    client.get("/api/live")
    r = client.post("/api/prices", json={"usd_per_mtok_in": 2.0, "usd_per_mtok_out": 10.0})
    assert r.status_code == 200 and r.json()["usd_per_mtok_in"] == 2.0
    econ = client.get("/api/live").json()["economics"]      # stored payload, repriced at once
    assert econ["total"]["cloud_equiv_usd"] == pytest.approx(1000 / 1e6 * 2 + 100 / 1e6 * 10)
    assert client.post("/api/prices", json={"usd_per_gb": -1}).status_code == 422
    assert client.post("/api/prices", json={"usd_per_gb": "abc"}).status_code == 422
    assert client.post("/api/prices", json={"surprise": 1}).status_code == 422


def pipeline_app(run_dir, tmp_path, fetch_pipeline):
    return create_monitor_app(run_dir, tmp_path / "prices.json", pipeline_url="http://x/api/state",
                              fetch=lambda s: "", fetch_pipeline=fetch_pipeline, system=lambda: {})


def test_pipeline_metrics(tmp_path, env):
    run = env[1].run_dir
    state = {"metrics": {"frames": 200, "vlm_calls": 2, "bytes_up": 1500, "vlm_tokens": 600}}
    live = TestClient(pipeline_app(run, tmp_path, lambda url: state)).get("/api/live").json()
    assert live["pipeline"]["frames"] == 200
    p = live["economics"]["pipeline"]
    assert p["frames_per_vlm_call"] == 100.0 and p["cascade_calls_avoided"] == 198
    assert p["video_bytes_equiv"] == 200 * 50_000


@pytest.mark.parametrize("fetch_pipeline", [
    lambda url: (_ for _ in ()).throw(ConnectionError("down")),
    lambda url: {"metrics": ["not", "a", "dict"]},
    lambda url: ["not a dict"],
])
def test_pipeline_errors_and_bad_shapes_are_null(tmp_path, env, fetch_pipeline):
    live = TestClient(pipeline_app(env[1].run_dir, tmp_path, fetch_pipeline)).get("/api/live").json()
    assert live["pipeline"] is None and live["economics"]["pipeline"] is None


def test_missing_prices_file_and_run_dir(tmp_path):
    app = create_monitor_app(tmp_path / "nope", tmp_path / "none.json", fetch=lambda s: "",
                             system=lambda: {})
    live = TestClient(app).get("/api/live").json()
    assert live["models"] == [] and live["economics"]["total"]["cloud_equiv_usd"] == 0.0

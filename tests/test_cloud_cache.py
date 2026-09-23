import json

import pytest

from sentinel.cloud_cache import CachedVLM, FreshCallLimit, ResponseCache, cache_key
from sentinel.schema import ContextResult

CTX = ContextResult(source_type="wildland", smoke_color="grey", attended="no", near_structures=True,
                    near_road=False, size_estimate="small", description="Smoke on a ridge.")


class FakeVLM:
    """Scripted ContextVLM stand-in: outputs are ContextResult, None (parse fail) or an error string."""

    def __init__(self, outputs, model="qwen", response_format="json_object"):
        self.outputs, self.model, self.response_format = list(outputs), model, response_format
        self.calls = 0
        self.last_usage, self.last_error = {}, None

    def classify(self, jpeg):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, str):
            self.last_usage, self.last_error = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}, out
            return None, 0
        self.last_usage = {"prompt_tokens": 500, "completion_tokens": 60, "calls": 1}
        self.last_error = None
        return out, 560


class StepClock:
    """Each pair of reads (before/after a call) spans step_s seconds."""

    def __init__(self, step_s):
        self.t, self.step, self.reads = 0.0, step_s, 0

    def __call__(self):
        self.reads += 1
        if self.reads % 2 == 0:
            self.t += self.step
        return self.t


def test_cache_key_depends_on_model_mode_and_bytes():
    k = cache_key("m", "json_object", b"img")
    assert k == cache_key("m", "json_object", b"img")
    assert len({k, cache_key("m2", "json_object", b"img"), cache_key("m", "none", b"img"),
                cache_key("m", "json_object", b"img2")}) == 4


def test_miss_calls_the_vlm_and_stores_context_usage_and_latency(tmp_path):
    path = tmp_path / "c.jsonl"
    vlm = CachedVLM(FakeVLM([CTX]), ResponseCache(path), clock=StepClock(0.8))
    ctx, tokens = vlm.classify(b"img")
    assert ctx == CTX and tokens == 560
    assert vlm.last_cached is False and vlm.last_latency_ms == pytest.approx(800)
    assert (vlm.n_fresh, vlm.n_cached) == (1, 0)
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["key"] == cache_key("qwen", "json_object", b"img")
    assert rec["context"]["source_type"] == "wildland" and rec["usage"]["prompt_tokens"] == 500
    assert rec["latency_ms"] == pytest.approx(800)
    assert "img" not in path.read_text()  # image bytes are not stored


def test_hit_does_not_call_the_vlm_and_reports_stored_latency_separately(tmp_path):
    path = tmp_path / "c.jsonl"
    CachedVLM(FakeVLM([CTX]), ResponseCache(path), clock=StepClock(0.8)).classify(b"img")
    fake = FakeVLM([])
    vlm = CachedVLM(fake, ResponseCache(path), clock=StepClock(99))  # a fresh run: reloads the file
    ctx, tokens = vlm.classify(b"img")
    assert fake.calls == 0 and ctx == CTX and tokens == 560
    assert vlm.last_cached is True and vlm.last_latency_ms == pytest.approx(800)
    assert vlm.last_usage["completion_tokens"] == 60
    assert (vlm.n_fresh, vlm.n_cached) == (0, 1)


def test_parse_failures_are_cached_but_request_errors_are_not(tmp_path):
    path = tmp_path / "c.jsonl"
    vlm = CachedVLM(FakeVLM([None, "APITimeoutError: slow"]), ResponseCache(path), clock=StepClock(1))
    assert vlm.classify(b"a") == (None, 560)
    assert vlm.classify(b"b") == (None, 0)
    assert vlm.last_error.startswith("APITimeoutError") and vlm.last_cached is False
    assert (vlm.n_fresh, vlm.n_errors) == (1, 1)
    assert len(ResponseCache(path)) == 1
    again = CachedVLM(FakeVLM([CTX]), ResponseCache(path), clock=StepClock(1))
    assert again.classify(b"a") == (None, 560) and again.last_cached
    assert again.classify(b"b")[0] == CTX and not again.last_cached


def test_different_mode_is_a_different_entry(tmp_path):
    path = tmp_path / "c.jsonl"
    CachedVLM(FakeVLM([CTX]), ResponseCache(path)).classify(b"img")
    other = CachedVLM(FakeVLM([CTX], response_format="none"), ResponseCache(path))
    other.classify(b"img")
    assert other.last_cached is False


def test_truncated_last_line_is_ignored(tmp_path):
    path = tmp_path / "c.jsonl"
    CachedVLM(FakeVLM([CTX]), ResponseCache(path)).classify(b"img")
    with path.open("a") as f:
        f.write('{"key": "abc", "cont')
    assert len(ResponseCache(path)) == 1


def test_max_fresh_calls_stops_before_billing_more(tmp_path):
    vlm = CachedVLM(FakeVLM([CTX, CTX]), ResponseCache(tmp_path / "c.jsonl"), max_fresh_calls=1)
    vlm.classify(b"a")
    vlm.classify(b"a")  # cache hit: free
    with pytest.raises(FreshCallLimit):
        vlm.classify(b"b")


def test_consecutive_request_errors_abort(tmp_path):
    vlm = CachedVLM(FakeVLM(["E1", "E2"]), ResponseCache(tmp_path / "c.jsonl"), max_consecutive_errors=2)
    vlm.classify(b"a")
    with pytest.raises(SystemExit, match="2 cloud requests failed"):
        vlm.classify(b"b")

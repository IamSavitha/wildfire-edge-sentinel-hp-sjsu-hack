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


FP = {"host": "api.example.com", "model": "m", "mode": "json_object", "prompt_sha256": "abc", "max_tokens": 160}


def test_cache_key_depends_on_every_fingerprint_field_and_bytes():
    k = cache_key(FP, b"img")
    assert k == cache_key(dict(reversed(list(FP.items()))), b"img")
    variants = [cache_key({**FP, f: "other"}, b"img") for f in FP] + [cache_key(FP, b"img2")]
    assert len({k, *variants}) == len(FP) + 2


def test_key_uses_the_vlm_fingerprint(tmp_path):
    from sentinel.vlm_client import CLOUD_FULL_FRAME_PROMPT, ContextVLM
    path = tmp_path / "c.jsonl"
    a = ContextVLM("m", base_url="https://a.example/v1", client=object(), response_format="none")
    b = ContextVLM("m", base_url="https://a.example/v1", client=object(), response_format="none",
                   system_prompt=CLOUD_FULL_FRAME_PROMPT)
    assert CachedVLM(a, ResponseCache(path)).fingerprint != CachedVLM(b, ResponseCache(path)).fingerprint


def test_miss_calls_the_vlm_and_stores_context_usage_and_latency(tmp_path):
    path = tmp_path / "c.jsonl"
    vlm = CachedVLM(FakeVLM([CTX]), ResponseCache(path), clock=StepClock(0.8))
    ctx, tokens = vlm.classify(b"img")
    assert ctx == CTX and tokens == 560
    assert vlm.last_cached is False and vlm.last_latency_ms == pytest.approx(800)
    assert (vlm.n_fresh, vlm.n_cached) == (1, 0)
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["key"] == cache_key({"model": "qwen", "mode": "json_object"}, b"img")
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


def test_truncated_last_line_is_repaired(tmp_path):
    path = tmp_path / "c.jsonl"
    CachedVLM(FakeVLM([CTX]), ResponseCache(path)).classify(b"img")
    with path.open("a") as f:
        f.write('{"key": "abc", "cont')
    assert len(ResponseCache(path)) == 1
    assert path.read_text().endswith("}\n") and '"abc"' not in path.read_text()
    CachedVLM(FakeVLM([CTX]), ResponseCache(path)).classify(b"img2")  # appends on a clean line
    assert len(ResponseCache(path)) == 2


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


# ---------------------------------------------------------------- transient-error retries

import httpx  # noqa: E402
import openai  # noqa: E402

REQ = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def rate_limited(retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    return openai.RateLimitError("slow down", response=httpx.Response(429, headers=headers, request=REQ), body=None)


class Flaky(FakeVLM):
    """Raises the scripted exceptions (as ContextVLM reports them) before answering."""

    def __init__(self, errors, answer=CTX):
        super().__init__([])
        self.errors, self.answer = list(errors), answer
        self.last_exception = None

    def classify(self, jpeg):
        self.calls += 1
        if self.errors:
            self.last_exception = self.errors.pop(0)
            self.last_usage, self.last_error = {"calls": 0}, type(self.last_exception).__name__
            return None, 0
        self.last_exception, self.last_error = None, None
        self.last_usage = {"prompt_tokens": 500, "completion_tokens": 60, "calls": 1}
        return self.answer, 560


def test_transient_errors_are_retried_with_retry_after_or_backoff_and_untimed(tmp_path):
    from sentinel.cloud_cache import is_transient
    slept = []
    fake = Flaky([rate_limited("3"), openai.APITimeoutError(request=REQ),
                  openai.InternalServerError("boom", response=httpx.Response(503, request=REQ), body=None)])
    clock = StepClock(0.4)
    vlm = CachedVLM(fake, ResponseCache(tmp_path / "c.jsonl"), clock=clock, sleep=slept.append)
    ctx, _ = vlm.classify(b"img")
    assert ctx == CTX and fake.calls == 4 and vlm.n_retries == 3
    assert slept == [3.0, 4.0, 8.0]  # Retry-After, then 2^n
    assert vlm.last_latency_ms == pytest.approx(400)  # the final attempt only
    assert vlm.n_fresh == 1 and vlm.n_errors == 0
    assert is_transient(openai.APIConnectionError(request=REQ)) and not is_transient(ValueError())


def test_retries_give_up_after_three(tmp_path):
    fake = Flaky([rate_limited()] * 5)
    vlm = CachedVLM(fake, ResponseCache(tmp_path / "c.jsonl"), sleep=lambda s: None)
    assert vlm.classify(b"img") == (None, 0)
    assert fake.calls == 4 and vlm.n_errors == 1 and vlm.last_error == "RateLimitError"


def test_non_transient_errors_are_not_retried(tmp_path):
    bad = openai.BadRequestError("no json_schema", response=httpx.Response(400, request=REQ), body=None)
    fake = Flaky([bad])
    vlm = CachedVLM(fake, ResponseCache(tmp_path / "c.jsonl"), sleep=lambda s: pytest.fail("slept"))
    vlm.classify(b"img")
    assert fake.calls == 1 and vlm.n_errors == 1


def test_retry_after_is_capped_at_60_s():
    from sentinel.cloud_cache import retry_delay_s
    assert retry_delay_s(rate_limited("3600"), 1) == 60.0
    assert retry_delay_s(rate_limited("5"), 1) == 5.0 and retry_delay_s(rate_limited(), 2) == 4.0

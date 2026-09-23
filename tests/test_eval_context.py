import pytest

from scripts.eval_context import GROUP, evaluate
from sentinel.schema import ContextResult


def _label(source_type: str) -> dict:
    return {"source_type": source_type, "smoke_color": "grey", "attended": "no",
            "near_structures": False, "near_road": False, "size_estimate": "small",
            "description": "Smoke on a ridge."}


def _row(image: str, source_type: str) -> dict:
    return {"image": image, "label": _label(source_type)}


class FakeClock:
    """Each classify call advances time by the next step (seconds)."""

    def __init__(self, steps):
        self.t = 0.0
        self.steps = list(steps)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls % 2 == 0:  # second read of a pair = after classify
            self.t += self.steps.pop(0)
        return self.t


def _run(rows, answers, steps, tokens=100):
    """answers: image bytes -> predicted source_type, or None for a parse failure."""
    seen = []

    def classify(jpeg):
        seen.append(jpeg)
        st = answers[jpeg]
        return (ContextResult(**_label(st)) if st else None), tokens

    out = evaluate(rows, classify, read_bytes=lambda p: p.encode(), clock=FakeClock(steps))
    return out, seen


def test_accuracy_group_parse_fail_and_per_source():
    rows = [
        _row("a.jpg", "wildland"),       # exact
        _row("b.jpg", "structure"),      # wrong type, same danger group
        _row("c.jpg", "campfire"),       # parse failure
        _row("d.jpg", "fog_dust_cloud"),  # wrong group
    ]
    answers = {b"a.jpg": "wildland", b"b.jpg": "wildland", b"c.jpg": None, b"d.jpg": "campfire"}
    out, seen = _run(rows, answers, steps=[0.1, 0.2, 0.3, 0.4])

    assert seen == [b"a.jpg", b"b.jpg", b"c.jpg", b"d.jpg"]
    assert out["n"] == 4
    assert out["source_type_acc"] == pytest.approx(0.25)
    assert out["group_acc"] == pytest.approx(0.5)
    assert out["parse_fail_rate"] == pytest.approx(0.25)
    assert out["tokens_per_call"] == pytest.approx(100.0)
    assert out["latency_ms_p50"] == pytest.approx(250.0)
    assert out["latency_ms_p95"] == pytest.approx(385.0)
    assert out["per_source"] == {
        "wildland": {"n": 1, "correct": 1},
        "structure": {"n": 1, "correct": 0},
        "campfire": {"n": 1, "correct": 0},
        "fog_dust_cloud": {"n": 1, "correct": 0},
    }


def test_per_source_aggregates_repeated_classes():
    rows = [_row("a.jpg", "campfire"), _row("b.jpg", "campfire"), _row("c.jpg", "wildland")]
    answers = {b"a.jpg": "campfire", b"b.jpg": "bbq_chimney", b"c.jpg": "wildland"}
    out, _ = _run(rows, answers, steps=[0.01] * 3)
    assert out["per_source"]["campfire"] == {"n": 2, "correct": 1}
    assert out["per_source"]["wildland"] == {"n": 1, "correct": 1}
    assert out["group_acc"] == pytest.approx(1.0)  # bbq_chimney and campfire are both benign


def test_empty_split_raises():
    with pytest.raises(ValueError, match="empty split"):
        evaluate([], lambda b: (None, 0), read_bytes=lambda p: b"")


def test_group_covers_every_source_type():
    from typing import get_args

    from sentinel.schema import SourceType
    assert set(GROUP) == set(get_args(SourceType))


def test_output_guard_refuses_existing_file_without_force(tmp_path, capsys):
    from scripts.eval_context import check_output

    out = tmp_path / "context_x.json"
    check_output(out, force=False)  # missing: fine
    out.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        check_output(out, force=False)
    assert exc.value.code == 1
    assert "--force" in capsys.readouterr().err
    check_output(out, force=True)  # explicit overwrite: fine


# ---------------------------------------------------------------- cloud baseline (fakes only)

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from sentinel.cloud_cache import CachedVLM, ResponseCache  # noqa: E402
from sentinel.vlm_client import ContextVLM  # noqa: E402

SECRET = "sk-test-SECRET-eval-0001"


class ScriptedCloud:
    """ContextVLM stand-in: bytes -> (source_type | None | "ERR"), fixed usage per answered call."""

    model, response_format = "qwen7b", "json_object"

    def __init__(self, answers):
        self.answers, self.calls = answers, 0
        self.last_usage, self.last_error = {}, None

    def classify(self, jpeg):
        self.calls += 1
        st = self.answers[jpeg]
        if st == "ERR":
            self.last_usage, self.last_error = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}, "Timeout"
            return None, 0
        self.last_usage, self.last_error = {"prompt_tokens": 400, "completion_tokens": 50, "calls": 1}, None
        return (ContextResult(**_label(st)) if st else None), 450


def test_cloud_evaluate_splits_fresh_cached_errors_and_billed_tokens(tmp_path):
    cache = ResponseCache(tmp_path / "c.jsonl")
    answers = {b"a": "wildland", b"b": None, b"c": "ERR", b"d": "campfire"}
    # pre-fill the cache for d: a previous run measured 900 ms
    CachedVLM(ScriptedCloud(answers), cache, clock=FakeClock([0.9])).classify(b"d")
    fake = ScriptedCloud(answers)
    src = CachedVLM(fake, cache, clock=lambda: 0.0)
    rows = [_row("a", "wildland"), _row("b", "wildland"), _row("c", "wildland"), _row("d", "campfire")]
    out = evaluate(rows, src.classify, read_bytes=lambda p: p.encode(), clock=FakeClock([0.2, 0.4, 5.0, 0.0]),
                   source=src)
    assert fake.calls == 3  # d came from the cache
    assert (out["n_fresh"], out["n_cached"], out["n_request_errors"]) == (2, 1, 1)
    assert out["source_type_acc"] == pytest.approx(0.5)
    assert out["parse_fail_rate"] == pytest.approx(0.5)  # b (bad JSON) and c (request error)
    assert out["request_error_rate"] == pytest.approx(0.25)
    assert out["latency_ms_p50"] == pytest.approx(300)  # fresh only: 200 and 400 ms
    assert out["cached_latency_ms_p50"] == pytest.approx(900)
    assert (out["billed_prompt_tokens"], out["billed_completion_tokens"]) == (800, 100)
    assert (out["prompt_tokens_total"], out["completion_tokens_total"]) == (1200, 150)
    assert out["prompt_tokens_per_call"] == pytest.approx(400)
    # over answered calls only (c had no answer): a and d right out of a, b, d
    assert out["n_answered"] == 3 and out["source_type_acc_answered"] == pytest.approx(2 / 3)
    assert out["group_acc_answered"] == pytest.approx(2 / 3) and out["n_retries"] == 0


def test_local_evaluate_has_no_cloud_fields():
    out, _ = _run([_row("a.jpg", "wildland")], {b"a.jpg": "wildland"}, steps=[0.1])
    assert "n_fresh" not in out and "billed_prompt_tokens" not in out


VALID = json.dumps(_label("wildland"))


def _fake_client(content, calls):
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=SimpleNamespace(prompt_tokens=700, completion_tokens=45))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _cloud_setup(tmp_path, monkeypatch, n=3):
    import scripts.eval_context as ec

    monkeypatch.chdir(tmp_path)
    for i in range(n):
        (tmp_path / f"{i}.jpg").write_bytes(b"\xff\xd8" + bytes([i]))
    (tmp_path / "split.jsonl").write_text(
        "".join(json.dumps(_row(f"{i}.jpg", "wildland")) + "\n" for i in range(n)))
    calls = []
    monkeypatch.setenv("CLOUD_VLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("CLOUD_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setenv("CLOUD_VLM_API_KEY", SECRET)
    real = ec.cloud_vlm_from_env

    def fake_from_env(**kw):  # the real env parsing, with the network client replaced
        vlm = real(**kw)
        vlm.client = _fake_client("```json\n" + VALID + "\n```", calls)
        return vlm
    monkeypatch.setattr(ec, "cloud_vlm_from_env", fake_from_env)
    return ec, calls


def test_cloud_main_writes_result_with_provider_metadata_and_no_key(tmp_path, monkeypatch, capsys):
    ec, calls = _cloud_setup(tmp_path, monkeypatch)
    ec.main(["--cloud", "--split", "split.jsonl", "--response-format", "json_object", "--limit", "2"])
    out = json.loads((tmp_path / "results" / "context_cloud_qwen7b.json").read_text())
    assert out["cloud"] is True and out["n"] == 2 and len(calls) == 2
    assert out["provider_host"] == "api.example.com" and out["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert out["response_format"] == "json_object" and calls[0]["response_format"] == {"type": "json_object"}
    assert out["source_type_acc"] == 1.0 and out["n_fresh"] == 2 and out["n_cached"] == 0
    assert out["billed_prompt_tokens"] == 1400 and out["billed_completion_tokens"] == 90
    text = (tmp_path / "results" / "context_cloud_qwen7b.json").read_text() + capsys.readouterr().out
    text += (tmp_path / "data" / "cloud_cache.jsonl").read_text()
    assert SECRET not in text


def test_cloud_rerun_is_served_from_cache(tmp_path, monkeypatch):
    ec, calls = _cloud_setup(tmp_path, monkeypatch)
    ec.main(["--cloud", "--split", "split.jsonl", "--name", "c1", "--response-format", "none"])
    ec.main(["--cloud", "--split", "split.jsonl", "--name", "c1", "--response-format", "none", "--force"])
    out = json.loads((tmp_path / "results" / "context_c1.json").read_text())
    assert len(calls) == 3  # second run billed nothing
    assert out["n_fresh"] == 0 and out["n_cached"] == 3 and out["latency_ms_p50"] is None
    assert out["cached_latency_ms_p50"] is not None and out["billed_prompt_tokens"] == 0


def test_cloud_main_names_missing_env_var(tmp_path, monkeypatch):
    ec, _ = _cloud_setup(tmp_path, monkeypatch)
    monkeypatch.delenv("CLOUD_VLM_API_KEY")
    with pytest.raises(SystemExit, match="CLOUD_VLM_API_KEY"):
        ec.main(["--cloud", "--split", "split.jsonl"])


def test_local_mode_still_requires_model_and_name():
    from scripts.eval_context import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--name", "x"])
    assert parse_args(["--cloud"]).name == "cloud_qwen7b"

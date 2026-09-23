import json
from types import SimpleNamespace

from sentinel.vlm_client import ContextVLM

VALID = json.dumps({"source_type": "campfire", "smoke_color": "white", "attended": "yes",
                    "near_structures": False, "near_road": False, "size_estimate": "small",
                    "description": "Small campfire in a fire ring with people nearby."})


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=out))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=40))


def vlm(outputs):
    completions = FakeCompletions(outputs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ContextVLM("m", client=client), completions


def test_parses_valid_json():
    v, _ = vlm([VALID])
    ctx, tokens = v.classify(b"\xff\xd8fake")
    assert ctx.source_type == "campfire" and tokens == 340


def test_retries_once_on_bad_json():
    v, _ = vlm(["not json", VALID])
    ctx, tokens = v.classify(b"\xff\xd8fake")
    assert ctx is not None and tokens == 680


def test_gives_up_after_two_bad_outputs():
    v, _ = vlm(["bad", "bad"])
    assert v.classify(b"\xff\xd8fake") == (None, 680)


def test_timeout_returns_none_without_retry():
    v, c = vlm([TimeoutError("slow")])
    assert v.classify(b"\xff\xd8fake") == (None, 0)
    assert len(c.calls) == 1


def test_request_is_token_capped_and_schema_constrained():
    v, c = vlm([VALID])
    v.classify(b"\xff\xd8fake")
    call = c.calls[0]
    assert call["max_tokens"] <= 200
    assert call["temperature"] == 0
    assert call["response_format"]["type"] == "json_schema"


class FakeModels:
    def __init__(self, ids=(), error=None):
        self.ids, self.error = list(ids), error

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self.ids])


def test_ensure_served_passes_when_model_listed():
    from sentinel.vlm_client import ensure_served
    ensure_served(SimpleNamespace(models=FakeModels(["base", "context"])), "context", "http://x/v1")


def test_ensure_served_exits_when_model_missing():
    import pytest

    from sentinel.vlm_client import ensure_served
    with pytest.raises(SystemExit, match=r"'context' not served at http://x/v1; available: \['base'\]"):
        ensure_served(SimpleNamespace(models=FakeModels(["base"])), "context", "http://x/v1")


def test_ensure_served_exits_when_server_unreachable():
    import pytest

    from sentinel.vlm_client import ensure_served
    with pytest.raises(SystemExit, match="http://x/v1"):
        ensure_served(SimpleNamespace(models=FakeModels(error=ConnectionError("refused"))), "m", "http://x/v1")


def test_last_usage_keeps_the_prompt_completion_split():
    v, _ = vlm(["not json", VALID])
    v.classify(b"\xff\xd8fake")
    assert v.last_usage == {"prompt_tokens": 600, "completion_tokens": 80, "calls": 2}
    v2, _ = vlm([TimeoutError("slow")])
    v2.classify(b"\xff\xd8fake")
    assert v2.last_usage == {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def test_unix_socket_base_url_targets_the_socket():
    vlm = ContextVLM("context", base_url="unix:///opt/hp/zrt/run/vllm-base7b.sock")
    assert str(vlm.client.base_url) == "http://localhost/v1/"
    assert vlm.client._client._transport._pool._uds == "/opt/hp/zrt/run/vllm-base7b.sock"


# ---------------------------------------------------------------- cloud provider modes

import logging  # noqa: E402

import pytest  # noqa: E402

from sentinel.vlm_client import (RESPONSE_FORMATS, SYSTEM_PROMPT, cloud_vlm_from_env,  # noqa: E402
                                 extract_json, schema_hint)

SECRET = "sk-test-SECRET-0123456789abcdef"


def cloud(outputs, response_format, usage=True):
    completions = FakeCompletions(outputs)
    if not usage:
        orig = completions.create

        def create(**kw):
            resp = orig(**kw)
            resp.usage = None
            return resp
        completions.create = create
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ContextVLM("qwen", client=client, api_key=SECRET, response_format=response_format), completions


def test_response_formats_are_the_three_modes():
    assert RESPONSE_FORMATS == ("json_schema", "json_object", "none")
    with pytest.raises(ValueError, match="response_format"):
        ContextVLM("m", client=object(), response_format="xml")


def test_json_schema_mode_keeps_the_local_request_unchanged():
    v, c = cloud([VALID], "json_schema")
    v.classify(b"\xff\xd8fake")
    call = c.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["messages"][0]["content"] == SYSTEM_PROMPT


def test_json_object_mode_sends_json_object_and_schema_in_prompt():
    v, c = cloud([VALID], "json_object")
    ctx, _ = v.classify(b"\xff\xd8fake")
    call = c.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    system = call["messages"][0]["content"]
    assert system.startswith(SYSTEM_PROMPT) and schema_hint() in system
    assert ctx.source_type == "campfire"


def test_none_mode_sends_no_response_format():
    v, c = cloud([VALID], "none")
    v.classify(b"\xff\xd8fake")
    assert "response_format" not in c.calls[0]
    assert schema_hint() in c.calls[0]["messages"][0]["content"]


def test_schema_hint_lists_every_field_and_enum_value():
    hint = schema_hint()
    for field in ("source_type", "smoke_color", "attended", "near_structures", "near_road",
                  "size_estimate", "description"):
        assert field in hint
    for value in ("wildland", "bbq_chimney", "fog_dust_cloud", "black", "unclear", "large"):
        assert f'"{value}"' in hint
    assert "true or false" in hint


@pytest.mark.parametrize("text", [
    "```json\n" + VALID + "\n```",
    "Sure! Here is the classification:\n" + VALID + "\nLet me know if you need more.",
    "```\n" + VALID + "```",
])
def test_prosey_or_fenced_json_is_parsed_in_non_schema_modes(text):
    for mode in ("json_object", "none"):
        v, _ = cloud([text], mode)
        ctx, _ = v.classify(b"\xff\xd8fake")
        assert ctx is not None and ctx.source_type == "campfire"


def test_extract_json_skips_braces_that_are_not_json():
    assert extract_json('note {not json} then {"a": {"b": 1}} tail') == {"a": {"b": 1}}
    assert extract_json("no object here") is None
    assert extract_json(None) is None


def test_missing_usage_counts_zero_tokens_and_is_flagged():
    v, _ = cloud([VALID], "json_object", usage=False)
    ctx, tokens = v.classify(b"\xff\xd8fake")
    assert ctx is not None and tokens == 0
    assert v.last_usage["calls"] == 1 and v.last_usage["usage_missing"] == 1


def test_request_error_is_recorded_redacted_and_key_never_logged(caplog):
    v, _ = cloud([RuntimeError(f"Incorrect API key provided: {SECRET}")], "json_object")
    with caplog.at_level(logging.DEBUG):
        assert v.classify(b"\xff\xd8fake") == (None, 0)
    assert v.last_error.startswith("RuntimeError")
    assert SECRET not in v.last_error and SECRET not in caplog.text
    assert "sk-test" not in caplog.text


def test_key_never_in_repr_or_str():
    v, _ = cloud([VALID], "json_object")
    assert SECRET not in repr(v) and SECRET not in str(v)
    assert SECRET not in repr(vars(v))


def test_real_client_gets_the_key_but_repr_hides_it():
    v = ContextVLM("qwen", base_url="https://api.example.com/v1", api_key=SECRET, response_format="none")
    assert v.client.api_key == SECRET
    assert SECRET not in repr(v) and "api.example.com" in repr(v)


ENV = {"CLOUD_VLM_BASE_URL": "https://api.example.com/v1", "CLOUD_VLM_MODEL": "Qwen/Qwen2.5-VL-7B-Instruct",
       "CLOUD_VLM_API_KEY": SECRET}


def test_cloud_vlm_from_env_reads_url_model_key_and_format():
    v = cloud_vlm_from_env(env={**ENV, "CLOUD_VLM_RESPONSE_FORMAT": "json_object"})
    assert v.model == "Qwen/Qwen2.5-VL-7B-Instruct" and v.response_format == "json_object"
    assert v.client.api_key == SECRET and str(v.client.base_url).startswith("https://api.example.com/v1")
    assert cloud_vlm_from_env(env=ENV).response_format == "json_schema"
    assert cloud_vlm_from_env(env=ENV, response_format="none").response_format == "none"


def test_cloud_vlm_from_env_uses_the_process_environment(monkeypatch):
    for k, val in ENV.items():
        monkeypatch.setenv(k.replace("CLOUD_VLM", "ALT"), val)
    assert cloud_vlm_from_env(prefix="ALT").client.api_key == SECRET


@pytest.mark.parametrize("missing", ["CLOUD_VLM_BASE_URL", "CLOUD_VLM_MODEL", "CLOUD_VLM_API_KEY"])
def test_cloud_vlm_from_env_names_the_missing_variable(missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    with pytest.raises(SystemExit, match=missing) as e:
        cloud_vlm_from_env(env=env)
    assert SECRET not in str(e.value)


def test_cloud_vlm_from_env_rejects_bad_format_without_echoing_the_key():
    with pytest.raises(SystemExit, match="CLOUD_VLM_RESPONSE_FORMAT") as e:
        cloud_vlm_from_env(env={**ENV, "CLOUD_VLM_RESPONSE_FORMAT": "yaml"})
    assert SECRET not in str(e.value)


# ---------------------------------------------------------------- prompts, parse retries, fingerprint

def test_cloud_client_does_not_rebill_on_a_parse_failure():
    v = cloud_vlm_from_env(env=ENV, response_format="json_object")
    assert v.parse_retries == 0
    completions = FakeCompletions(["garbage", VALID])
    v.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    assert v.classify(b"\xff\xd8fake") == (None, 340)
    assert len(completions.calls) == 1
    assert cloud_vlm_from_env(env=ENV, parse_retries=1).parse_retries == 1


def test_local_client_still_retries_once():
    v, c = vlm(["bad", VALID])
    assert v.classify(b"\xff\xd8fake")[0] is not None and len(c.calls) == 2


def test_json_schema_mode_falls_back_to_extracting_the_object():
    v, c = vlm(["```json\n" + VALID + "\n```"])
    assert v.classify(b"\xff\xd8fake")[0].source_type == "campfire" and len(c.calls) == 1


def test_full_frame_prompt_is_sent_when_configured():
    from sentinel.vlm_client import CLOUD_FULL_FRAME_PROMPT
    assert "most frames contain no smoke" in CLOUD_FULL_FRAME_PROMPT.lower()
    assert '"none"' in CLOUD_FULL_FRAME_PROMPT and '"fog_dust_cloud"' in CLOUD_FULL_FRAME_PROMPT
    v = cloud_vlm_from_env(env=ENV, response_format="none", system_prompt=CLOUD_FULL_FRAME_PROMPT)
    completions = FakeCompletions([VALID])
    v.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    v.classify(b"\xff\xd8fake")
    assert completions.calls[0]["messages"][0]["content"].startswith(CLOUD_FULL_FRAME_PROMPT)


def test_fingerprint_changes_with_prompt_host_model_mode_and_max_tokens():
    from sentinel.vlm_client import CLOUD_FULL_FRAME_PROMPT
    base = ContextVLM("m", base_url="https://a.example/v1", client=object())
    fp = base.fingerprint()
    assert fp["host"] == "a.example" and fp["model"] == "m" and fp["mode"] == "json_schema" and fp["max_tokens"] == 160
    others = [ContextVLM("m", base_url="https://b.example/v1", client=object()),
              ContextVLM("m2", base_url="https://a.example/v1", client=object()),
              ContextVLM("m", base_url="https://a.example/v1", client=object(), response_format="none"),
              ContextVLM("m", base_url="https://a.example/v1", client=object(), max_tokens=200),
              ContextVLM("m", base_url="https://a.example/v1", client=object(), system_prompt=CLOUD_FULL_FRAME_PROMPT)]
    assert all(o.fingerprint() != fp for o in others)
    assert SECRET not in json.dumps(ContextVLM("m", client=object(), api_key=SECRET).fingerprint())


def test_request_exception_is_kept_for_the_retry_policy():
    v, _ = cloud([TimeoutError("slow")], "json_object")
    v.classify(b"\xff\xd8fake")
    assert isinstance(v.last_exception, TimeoutError)


def test_net_baseline_is_the_fastest_of_cheap_requests():
    from sentinel.vlm_client import measure_net_baseline_ms
    ticks = iter([0, 0.10, 1, 1.30, 2, 2.20, 3, 3.90, 4, 4.25])
    v = ContextVLM("m", client=SimpleNamespace(models=FakeModels(["m"])))
    assert measure_net_baseline_ms(v, n=5, clock=lambda: next(ticks)) == pytest.approx(100)


def test_net_baseline_warns_on_a_large_models_body(caplog):
    from sentinel.vlm_client import measure_net_baseline_ms

    class Raw:
        def list(self):
            return SimpleNamespace(content=b"x" * 60_000)
    models = SimpleNamespace(with_raw_response=Raw(), list=lambda: None)
    v = ContextVLM("m", client=SimpleNamespace(models=models))
    with caplog.at_level(logging.WARNING):
        assert measure_net_baseline_ms(v, n=2) is not None
    assert "60000 bytes" in caplog.text
    broken = ContextVLM("m", client=SimpleNamespace(models=FakeModels(error=RuntimeError(f"bad key {SECRET}"))),
                        api_key=SECRET)
    assert measure_net_baseline_ms(broken) is None

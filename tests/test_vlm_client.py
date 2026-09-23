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

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

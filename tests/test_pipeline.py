import numpy as np

from sentinel.config import Settings, Tower
from sentinel.pipeline import Pipeline
from sentinel.schema import ContextResult, Detection, Severity

FRAME = np.zeros((480, 640, 3), np.uint8)
TOWER = Tower(id="t1", name="Test", lat=37.0, lon=-121.0, source="", temp_c=30.0)
SMALL = Detection("smoke", 0.9, (100, 100, 200, 200))
BIG = Detection("smoke", 0.9, (100, 100, 250, 250))


def ctx(**kw):
    base = dict(source_type="wildland", smoke_color="grey", attended="no", near_structures=False,
                near_road=False, size_estimate="small", description="d")
    base.update(kw)
    return ContextResult(**base)


class FakeVLM:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def classify(self, jpeg):
        self.calls += 1
        return self.result, 300


class RecordingEscalator:
    def __init__(self):
        self.handled = []

    def handle(self, report, severity, now):
        self.handled.append((severity, report))


def run(result, schedule):
    """schedule: list of (time, detections)."""
    vlm, esc = FakeVLM(result), RecordingEscalator()
    dets = iter([d for _, d in schedule])
    p = Pipeline({"t1": TOWER}, lambda frame: next(dets), vlm, esc,
                 Settings(min_frames=3, recheck_s=30, max_rechecks=2, cooldown_s=0))
    for t, _ in schedule:
        p.process("t1", FRAME, now=t)
    return p, vlm, esc


def steady(n=3, det=SMALL):
    return [(t, [det]) for t in range(n)]


def test_clear_danger_alerts_immediately():
    _, vlm, esc = run(ctx(near_structures=True), steady())
    assert [s for s, _ in esc.handled] == [Severity.ALERT] and vlm.calls == 1


def test_fog_is_ignored_immediately():
    _, _, esc = run(ctx(source_type="fog_dust_cloud"), steady())
    assert [s for s, _ in esc.handled] == [Severity.IGNORE]


def test_campfire_waits_for_trend_then_logs():
    p, _, esc = run(ctx(source_type="campfire", attended="yes"), steady() + [(32, [SMALL])])
    assert [s for s, _ in esc.handled] == [Severity.LOG]
    assert p.history[0].trend == "static"


def test_small_wildland_growing_on_recheck_alerts():
    _, _, esc = run(ctx(), steady() + [(32, [BIG])])
    assert [s for s, _ in esc.handled] == [Severity.ALERT]


def test_vlm_failure_falls_back_and_still_alerts_when_growing():
    _, _, esc = run(None, steady() + [(32, [BIG])])
    severity, report = esc.handled[0]
    assert severity == Severity.ALERT and report["source_type"] == "unknown"


def test_one_vlm_call_per_event_not_per_frame():
    _, vlm, _ = run(ctx(source_type="campfire", attended="yes"),
                    steady(n=20) + [(32, [SMALL])])
    assert vlm.calls == 1


def test_monitor_finalizes_after_max_rechecks():
    _, _, esc = run(ctx(), steady() + [(32, [SMALL]), (62, [SMALL])])
    assert [s for s, _ in esc.handled] == [Severity.MONITOR]

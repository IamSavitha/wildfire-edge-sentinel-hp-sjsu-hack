import itertools
import threading
import time
from types import SimpleNamespace

import numpy as np

from sentinel.config import Settings
from sentinel.escalation import Link
from sentinel.main import run_loop, warn_if_not_served

ZERO_FRAME = np.zeros((10, 10, 3), np.uint8)


class FakePipeline:
    def __init__(self):
        self.calls, self.called = [], threading.Event()

    def process(self, tower_id, frame, now):
        self.calls.append((tower_id, now))
        self.called.set()


class FakeEscalator:
    def __init__(self):
        self.calls, self.called = [], threading.Event()

    def flush(self, now):
        self.calls.append(now)
        self.called.set()


def test_run_loop_processes_frames_and_flushes():
    pipe, esc = FakePipeline(), FakeEscalator()
    rt = SimpleNamespace(pipeline=pipe, escalator=esc, link=Link())
    streams = {"t1": itertools.repeat(ZERO_FRAME)}
    stop = threading.Event()
    worker = threading.Thread(target=run_loop, args=(rt, streams, Settings(fps=50), stop), daemon=True)
    worker.start()
    try:
        assert pipe.called.wait(5) and esc.called.wait(5)
    finally:
        stop.set()
        worker.join(5)
    assert not worker.is_alive()
    assert pipe.calls[0][0] == "t1" and len(esc.calls) >= 1


class FlakyPipeline(FakePipeline):
    def process(self, tower_id, frame, now):
        if tower_id == "t1":
            raise RuntimeError("bad tower")
        super().process(tower_id, frame, now)


def test_failing_tower_does_not_stall_others():
    pipe, esc = FlakyPipeline(), FakeEscalator()
    rt = SimpleNamespace(pipeline=pipe, escalator=esc, link=Link())
    streams = {"t1": itertools.repeat(ZERO_FRAME), "t2": itertools.repeat(ZERO_FRAME)}
    stop = threading.Event()
    worker = threading.Thread(target=run_loop, args=(rt, streams, Settings(fps=50), stop), daemon=True)
    worker.start()
    try:
        assert pipe.called.wait(5) and esc.called.wait(5)
    finally:
        stop.set()
        worker.join(5)
    assert not worker.is_alive()
    assert {tid for tid, _ in pipe.calls} == {"t2"} and len(esc.calls) >= 1


def test_ended_stream_is_dropped_and_logged_once(caplog):
    pipe, esc = FakePipeline(), FakeEscalator()
    rt = SimpleNamespace(pipeline=pipe, escalator=esc, link=Link())
    dead = iter([ZERO_FRAME])
    next(dead)  # primed at startup, now exhausted
    streams = {"t1": dead, "t2": itertools.repeat(ZERO_FRAME)}
    stop = threading.Event()
    worker = threading.Thread(target=run_loop, args=(rt, streams, Settings(fps=50), stop), daemon=True)
    with caplog.at_level("ERROR", logger="sentinel.main"):
        worker.start()
        try:
            assert pipe.called.wait(5) and esc.called.wait(5)
            deadline = time.monotonic() + 5
            while len(pipe.calls) < 3:
                assert time.monotonic() < deadline, "timed out waiting for 3 pipeline calls"
                time.sleep(0.01)
        finally:
            stop.set()
            worker.join(5)
    assert not worker.is_alive() and "t1" not in streams
    assert {tid for tid, _ in pipe.calls} == {"t2"}
    ended = [r for r in caplog.records if "stream ended" in r.getMessage()]
    assert len(ended) == 1 and ended[0].exc_info is None


class FakeModels:
    def __init__(self, ids=(), error=None):
        self.ids, self.error = list(ids), error

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self.ids])


def fake_vlm(**kw):
    return SimpleNamespace(client=SimpleNamespace(models=FakeModels(**kw)))


def test_warn_if_not_served_is_quiet_when_served(caplog):
    with caplog.at_level("WARNING", logger="sentinel.main"):
        assert warn_if_not_served(fake_vlm(ids=["base", "context"]), "context", "http://x/v1") is True
    assert not caplog.records


def test_warn_if_not_served_warns_when_model_missing(caplog):
    with caplog.at_level("WARNING", logger="sentinel.main"):
        assert warn_if_not_served(fake_vlm(ids=["base"]), "context", "http://x/v1") is False
    msg = caplog.records[-1].getMessage()
    assert "'context' not served" in msg and "detector-only fallback" in msg


def test_warn_if_not_served_warns_when_server_unreachable(caplog):
    with caplog.at_level("WARNING", logger="sentinel.main"):
        assert warn_if_not_served(fake_vlm(error=ConnectionError("refused")), "m", "http://x/v1") is False
    msg = caplog.records[-1].getMessage()
    assert "cannot list models" in msg and "detector-only fallback" in msg

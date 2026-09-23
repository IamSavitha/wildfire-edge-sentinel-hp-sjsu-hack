import itertools
import threading
from types import SimpleNamespace

import numpy as np

from sentinel.config import Settings
from sentinel.escalation import Link
from sentinel.main import run_loop

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

import threading

import numpy as np
from fastapi.testclient import TestClient

from sentinel.app import Runtime, create_app
from sentinel.config import Settings, Tower
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.schema import ContextResult, Detection

CTX = ContextResult(source_type="wildland", smoke_color="black", attended="no", near_structures=False,
                    near_road=False, size_estimate="small", description="d")


class FakeVLM:
    def classify(self, jpeg):
        return CTX, 300


def runtime(tmp_path):
    metrics, link = Metrics(), Link()
    esc = Escalator(Outbox(":memory:"), link, lambda a, b: {}, lambda p: None, metrics)
    det = [Detection("smoke", 0.9, (10, 10, 50, 50))]
    pipe = Pipeline({"t1": Tower("t1", "Test", 1.0, 2.0, "")}, lambda f: det, FakeVLM(), esc,
                    Settings(min_frames=1), metrics)
    return Runtime(pipe, esc, link, feedback_path=tmp_path / "fb.jsonl")


def test_state_and_link_toggle(tmp_path):
    client = TestClient(create_app(runtime(tmp_path)))
    assert client.get("/api/state").json()["online"] is False
    assert client.post("/api/link", json={"online": True}).json() == {"online": True}
    assert client.get("/api/state").json()["online"] is True


def test_frame_event_and_feedback(tmp_path):
    rt = runtime(tmp_path)
    client = TestClient(create_app(rt))
    assert client.get("/api/frame/t1").status_code == 404
    rt.pipeline.process("t1", np.zeros((100, 100, 3), np.uint8), now=0)
    assert client.get("/api/frame/t1").headers["content-type"] == "image/jpeg"
    state = client.get("/api/state").json()
    assert state["events"][0]["severity"] == "ALERT" and state["outbox"]["pending"] == 1
    event_id = state["events"][0]["id"]
    assert client.post(f"/api/feedback/{event_id}", json={"label": "false_alarm"}).status_code == 200
    assert "false_alarm" in rt.feedback_path.read_text()


def test_index_serves_dashboard(tmp_path):
    assert "Wildfire Edge Sentinel" in TestClient(create_app(runtime(tmp_path))).get("/").text


def test_state_served_while_vlm_classifies(tmp_path):
    started, release = threading.Event(), threading.Event()

    class SlowVLM:
        def classify(self, jpeg):
            started.set()
            release.wait(5)
            return CTX, 300

    rt = runtime(tmp_path)
    rt.pipeline.vlm = SlowVLM()
    client = TestClient(create_app(rt))
    worker = threading.Thread(target=rt.pipeline.process,
                              args=("t1", np.zeros((100, 100, 3), np.uint8), 0))
    worker.start()
    assert started.wait(5)
    state = client.get("/api/state").json()
    assert len(state["active"]) == 1 and state["active"][0]["severity"] is None
    release.set()
    worker.join(5)
    state = client.get("/api/state").json()
    assert state["active"] == [] and state["events"][0]["severity"] == "ALERT"

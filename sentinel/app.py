"""Demo dashboard API: live feeds, events, outbox, link toggle, ranger feedback."""
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from sentinel.escalation import Escalator, Link
from sentinel.imaging import to_jpeg
from sentinel.pipeline import Pipeline

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"


@dataclass
class Runtime:
    pipeline: Pipeline
    escalator: Escalator
    link: Link
    feedback_path: Path = Path("data/feedback.jsonl")
    lock: threading.Lock = field(default_factory=threading.Lock)


class LinkState(BaseModel):
    online: bool


class Feedback(BaseModel):
    label: Literal["correct", "false_alarm"]


def create_app(rt: Runtime) -> FastAPI:
    app = FastAPI(title="Wildfire Edge Sentinel")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD.read_text()

    @app.get("/api/state")
    def state():
        with rt.lock:
            p = rt.pipeline
            return {
                "online": rt.link.online,
                "towers": [{"id": t.id, "name": t.name} for t in p.towers.values()],
                "active": [e.to_dict() for e in p.active.values()],
                "events": [e.to_dict() for e in reversed(p.history[-30:])],
                "outbox": {"pending": rt.escalator.outbox.pending_count(),
                           "sent": rt.escalator.outbox.sent_count()},
                "metrics": p.metrics.summary(),
            }

    @app.post("/api/link")
    def set_link(s: LinkState):
        rt.link.online = s.online
        return {"online": rt.link.online}

    @app.get("/api/frame/{tower_id}")
    def frame(tower_id: str):
        with rt.lock:
            item = rt.pipeline.last_frame.get(tower_id)
        if item is None:
            raise HTTPException(404, "no frame yet")
        img, det = item[0].copy(), item[1]
        if det is not None:
            x1, y1, x2, y2 = map(int, det.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{det.cls} {det.conf:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return Response(to_jpeg(img, 75), media_type="image/jpeg")

    @app.post("/api/feedback/{event_id}")
    def feedback(event_id: str, fb: Feedback):
        with rt.lock:
            ev = next((e for e in rt.pipeline.history if e.id == event_id), None)
        if ev is None:
            raise HTTPException(404, "unknown event")
        rt.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with rt.feedback_path.open("a") as f:
            f.write(json.dumps({"event_id": event_id, "label": fb.label,
                                "context": ev.ctx.model_dump() if ev.ctx else None,
                                "thumbnail_jpeg_b64": ev.thumbnail_b64}) + "\n")
        return {"ok": True}

    return app

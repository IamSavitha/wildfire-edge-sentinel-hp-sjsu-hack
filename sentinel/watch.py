"""Continuous monitoring of one fire as the tower network sees it: the frames of every camera that recorded
it, merged in time order. Every frame goes through the detector; the VLM runs only when a persistent plume
opens (or joins) an incident, then once per batch of frames per camera, with a situation update per batch.

The engine (detector, VLM, incident store) lives in sentinel.incident_map; this module holds the
frame source and the pure per-batch logic so both can be tested without a GPU."""
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from sentinel.trend import classify_trend

_FIGLIB_RE = re.compile(r"^(\d+)_([+-]\d+)\.[A-Za-z]+$")
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
TIMELINE_CAP = 240


@dataclass
class Frame:
    path: Path
    epoch: float | None        # when the camera took it (FIgLib names carry it)
    offset_s: int | None       # seconds from ignition (FIgLib), None if unknown
    camera_id: str | None = None


def list_frames(folder: str | Path, camera_id: str | None = None) -> list[Frame]:
    """Time-ordered frames of a camera folder. FIgLib names are <epoch>_<offset from ignition>.jpg."""
    out = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        m = _FIGLIB_RE.match(p.name)
        out.append(Frame(p, float(m.group(1)) if m else None, int(m.group(2)) if m else None, camera_id))
    return sorted(out, key=lambda f: (f.epoch is None, f.epoch or 0, f.path.name))


def merge_streams(streams: list[list[Frame]]) -> list[Frame]:
    """Frames of several cameras in the order the network would have received them."""
    return sorted((f for s in streams for f in s), key=lambda f: (f.epoch is None, f.epoch or 0, f.camera_id or ""))


def time_steps(frames: list[Frame], step_s: float = 60.0) -> list[int]:
    """Index where each new step_s window starts: replay pauses once per window, not once per camera."""
    starts, last = [], None
    for i, f in enumerate(frames):
        b = None if f.epoch is None else int(f.epoch // step_s)
        if i == 0 or b is None or b != last:
            starts.append(i)
        last = b
    return starts


def summarize_batch(prev: list[dict], cur: list[dict], min_conf: float) -> dict:
    """Compare one batch of frame points with the batch before it.
    A point is {"conf": best conf or 0, "area": best box area as a fraction of the frame (0 if none)}."""
    seen = [p for p in cur if p["conf"] >= min_conf]
    area_now = max((p["area"] for p in cur), default=0.0)
    area_before = max((p["area"] for p in prev), default=0.0)
    if not seen:
        trend = "not_seen"
    else:
        trend = classify_trend(area_before, area_now)
    return {"frames": len(cur), "detected_in": len(seen), "max_conf": max((p["conf"] for p in cur), default=0.0),
            "area_before": area_before, "area_now": area_now,
            "area_ratio": (area_now / area_before) if area_before > 0 else None, "trend": trend}


def update_text(u: dict) -> str:
    """One line for the incident timeline, built from measured values only."""
    parts = [f"smoke in {u['detected_in']}/{u['frames']} frames"]
    if u.get("area_ratio") is not None and u["detected_in"]:
        parts.append(f"plume area ×{u['area_ratio']:.2f}")
    parts.append({"growing": "growing", "static": "steady", "dissipating": "shrinking",
                  "not_seen": "not visible this batch"}.get(u["trend"], u["trend"]))
    if u.get("source_type"):
        parts.append(u["source_type"].replace("_", " "))
    if u.get("severity"):
        parts.append(u["severity"])
    return " · ".join(parts)


@dataclass
class CamState:
    """Per-camera state inside a watch session."""
    camera_id: str
    streak: int = 0
    incident_id: str | None = None
    points: list[dict] = field(default_factory=list)
    batch: list[dict] = field(default_factory=list)
    prev_batch: list[dict] = field(default_factory=list)
    recent: list = field(default_factory=list)      # last few (image, detections, point) before an incident opens
    last_hit: tuple | None = None                   # (image, detection) of the latest frame with smoke
    last_hit_n: int | None = None                   # that frame's index in the incident filmstrip
    latest_thumb: str | None = None
    latest_offset: int | None = None
    frames_done: int = 0


@dataclass
class WatchSession:
    id: str
    label: str                          # e.g. "Junction Fire · 2026-06-29"
    camera_ids: list[str]
    frames: list[Frame]
    interval_s: float
    batch_frames: int
    time_mode: str                      # "live": stamp with the wall clock; "recorded": the frame's own time
    started_at: float
    fire_name: str | None = None
    state: str = "running"              # running | done | stopped | error
    idx: int = 0
    cams: dict[str, CamState] = field(default_factory=dict)
    incident_ids: list[str] = field(default_factory=list)
    frames_done: int = 0
    vlm_calls: int = 0
    vlm_tokens: int = 0
    error: str | None = None
    stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        for cid in self.camera_ids:
            self.cams.setdefault(cid, CamState(cid))

    def view(self) -> dict:
        return {"id": self.id, "label": self.label, "camera_ids": self.camera_ids, "state": self.state,
                "idx": self.idx, "total": len(self.frames), "interval_s": self.interval_s,
                "batch_frames": self.batch_frames, "time_mode": self.time_mode, "incident_ids": self.incident_ids,
                "frames_done": self.frames_done, "vlm_calls": self.vlm_calls, "vlm_tokens": self.vlm_tokens,
                "error": self.error,
                "cams": [{"camera_id": c.camera_id, "incident_id": c.incident_id, "latest_offset": c.latest_offset,
                          "frames_done": c.frames_done, "has_frame": c.latest_thumb is not None,
                          "points": c.points[-120:]} for c in self.cams.values()]}

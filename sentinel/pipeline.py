"""Stage orchestration: detect → gate → context VLM → trend → severity → escalate."""
import base64
import logging
import time
import threading
import uuid
from collections import deque
from dataclasses import dataclass

import numpy as np

from sentinel.config import Settings, Tower
from sentinel.gate import PersistenceGate
from sentinel.imaging import crop_box, to_jpeg
from sentinel.metrics import Metrics
from sentinel.report import build_report, render_text
from sentinel.schema import ContextResult, Detection, Severity
from sentinel.severity import assess, fallback_severity
from sentinel.trend import classify_trend
from sentinel.zones import in_any_zone

AREA_WINDOW = 6  # frames (~3 s at 2 fps): one missed detection must not read as a trend


@dataclass
class Event:
    id: str
    tower_id: str
    opened_at: float
    confidence: float
    last_area: float
    thumbnail_b64: str
    in_zone: bool = False
    ctx: ContextResult | None = None
    trend: str | None = None
    severity: Severity | None = None
    rechecks: int = 0
    next_check_at: float = 0.0
    report: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tower_id": self.tower_id, "opened_at": self.opened_at,
            "severity": self.severity.name if self.severity is not None else None,
            "trend": self.trend, "context": self.ctx.model_dump() if self.ctx else None,
            "report_text": render_text(self.report) if self.report else None,
            "thumbnail_b64": self.thumbnail_b64, "final": self.report is not None,
        }


class Pipeline:
    def __init__(self, towers: dict[str, Tower], detector, vlm, escalator,
                 settings: Settings, metrics: Metrics | None = None):
        self.towers = towers
        self.detector = detector
        self.vlm = vlm
        self.escalator = escalator
        self.s = settings
        self.metrics = metrics or Metrics()
        self.gate = PersistenceGate(settings.min_conf, settings.min_frames, settings.cooldown_s)
        self.active: dict[str, Event] = {}
        self.history: list[Event] = []
        self.latest: dict[str, Detection | None] = {}
        self.last_frame: dict[str, tuple[np.ndarray, Detection | None]] = {}
        self.burn_towers: set[str] = set()
        self.recent_area: dict[str, deque] = {}
        self.latched: dict[str, float] = {}  # tower -> last time smoke was seen after an ALERT
        self.lock = threading.Lock()  # guards state read by the dashboard; released during the VLM call

    # Single writer: only run_loop's thread may call process().
    def process(self, tower_id: str, frame: np.ndarray, now: float) -> None:
        t0 = time.perf_counter()
        dets = self.detector(frame)
        detect_ms = (time.perf_counter() - t0) * 1000
        with self.lock:
            self.metrics.inc("frames")
            self.metrics.time("detect_ms", detect_ms)
            strong = [d for d in dets if d.conf >= self.s.min_conf]
            best = max(strong, key=lambda d: d.conf, default=None)
            self.latest[tower_id] = best
            self.last_frame[tower_id] = (frame, best)
            self.recent_area.setdefault(tower_id, deque(maxlen=AREA_WINDOW)).append(
                best.area if best else 0.0)

            ev = self.active.get(tower_id)
            if ev is None:
                if tower_id in self.latched:  # same fire already alerted: wait for it to clear
                    if now - self.latched[tower_id] >= self.s.cooldown_s:
                        del self.latched[tower_id]  # then fall through to the gate
                    else:
                        if best is not None:
                            self.latched[tower_id] = now
                        return
                candidate = self.gate.update(tower_id, dets, now)
                if candidate is not None:
                    self._open(tower_id, frame, candidate, now)
            elif now >= ev.next_check_at:
                self._recheck(ev, now)

    def _open(self, tower_id: str, frame: np.ndarray, det: Detection, now: float) -> None:
        self.metrics.inc("candidates")
        tower = self.towers[tower_id]
        if self.s.full_frame:
            h, w = frame.shape[:2]
            crop = crop_box(frame, (0, 0, w, h), 0.0, self.s.full_frame_max_side)
        else:
            crop = crop_box(frame, det.box, self.s.crop_pad, self.s.crop_max_side)
        thumb = base64.b64encode(to_jpeg(crop_box(frame, det.box, 0.5, 256), 70)).decode()
        ev = Event(id=uuid.uuid4().hex[:12], tower_id=tower_id, opened_at=now,
                   confidence=det.conf, last_area=max(self.recent_area[tower_id]), thumbnail_b64=thumb,
                   in_zone=in_any_zone(det.box, tower.benign_zones))

        self.active[tower_id] = ev  # visible on the dashboard as "classifying…"
        t0 = time.perf_counter()
        self.lock.release()
        try:
            ctx, tokens = self.vlm.classify(to_jpeg(crop))
        except Exception:
            logging.getLogger(__name__).exception("VLM classify raised")
            ctx, tokens = None, 0
        finally:
            self.lock.acquire()
        ev.ctx = ctx
        self.metrics.time("vlm_ms", (time.perf_counter() - t0) * 1000)
        self.metrics.inc("vlm_calls")
        self.metrics.inc("vlm_tokens", tokens)
        if ev.ctx is None:
            self.metrics.inc("vlm_failures")

        severity = self._assess(ev, trend=None)
        if severity in (Severity.ALERT, Severity.IGNORE):
            del self.active[tower_id]
            self._finalize(ev, severity, now)
            return
        ev.severity = severity  # provisional until the trend re-check
        ev.next_check_at = now + self.s.recheck_s

    def _recheck(self, ev: Event, now: float) -> None:
        area = max(self.recent_area[ev.tower_id])
        ev.trend = classify_trend(ev.last_area, area)
        if area > 0:
            ev.last_area = area
        ev.rechecks += 1
        severity = self._assess(ev, ev.trend)
        if severity == Severity.MONITOR and ev.rechecks < self.s.max_rechecks:
            ev.severity = severity
            ev.next_check_at = now + self.s.recheck_s
            return
        del self.active[ev.tower_id]
        self._finalize(ev, severity, now)

    def _assess(self, ev: Event, trend: str | None) -> Severity:
        if ev.ctx is None:
            return fallback_severity(trend)
        return assess(ev.ctx, trend, ev.in_zone, ev.tower_id in self.burn_towers)

    def _finalize(self, ev: Event, severity: Severity, now: float) -> None:
        ev.severity = severity
        ev.report = build_report(ev.id, self.towers[ev.tower_id], severity, ev.ctx,
                                 ev.trend, ev.confidence, now, ev.thumbnail_b64)
        self.metrics.inc(f"severity_{severity.name}")
        self.metrics.time("decision_s", now - ev.opened_at)
        self.history.append(ev)
        if severity == Severity.ALERT:
            self.latched[ev.tower_id] = now
        self.escalator.handle(ev.report, severity, now)

"""Require a detection to persist across frames before opening an event."""
from collections import defaultdict

from sentinel.schema import Detection


class PersistenceGate:
    def __init__(self, min_conf: float, min_frames: int, cooldown_s: float):
        self.min_conf = min_conf
        self.min_frames = min_frames
        self.cooldown_s = cooldown_s
        self._streak: dict[str, int] = defaultdict(int)
        self._last_fired: dict[str, float] = {}

    def streak(self, tower_id: str) -> int:
        """Consecutive frames with a strong detection so far (for display)."""
        return self._streak.get(tower_id, 0)

    def reset_streak(self, tower_id: str) -> None:
        self._streak.pop(tower_id, None)

    def update(self, tower_id: str, detections: list[Detection], now: float) -> Detection | None:
        strong = [d for d in detections if d.conf >= self.min_conf]
        if not strong:
            self._streak[tower_id] = 0
            return None
        self._streak[tower_id] += 1
        if self._streak[tower_id] < self.min_frames:
            return None
        last = self._last_fired.get(tower_id)
        if last is not None and now - last < self.cooldown_s:
            return None
        self._last_fired[tower_id] = now
        self._streak[tower_id] = 0
        return max(strong, key=lambda d: d.conf)

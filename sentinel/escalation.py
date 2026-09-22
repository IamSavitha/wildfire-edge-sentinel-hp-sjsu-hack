"""The explicit edge→cloud escalation policy.

Rule: only ALERT events leave the device, and the cloud is contacted only for what the
edge cannot produce (forecast) plus delivery to dispatch. Video never leaves the device.
"""
import json
from dataclasses import dataclass
from typing import Callable

from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.schema import Severity


@dataclass
class Link:
    online: bool = False


class Escalator:
    def __init__(self, outbox: Outbox, link: Link,
                 forecast_fn: Callable[[float, float], dict],
                 send_fn: Callable[[dict], None], metrics: Metrics):
        self.outbox = outbox
        self.link = link
        self.forecast_fn = forecast_fn
        self.send_fn = send_fn
        self.metrics = metrics
        self.local_log: list[dict] = []

    def handle(self, report: dict, severity: Severity, now: float) -> None:
        if severity == Severity.ALERT:
            self.outbox.enqueue(report["event_id"], report, now)
        elif severity in (Severity.LOG, Severity.MONITOR):
            self.local_log.append(report)

    def flush(self, now: float) -> int:
        if not self.link.online:
            return 0
        sent = 0
        for event_id, payload, _ in self.outbox.due(now):
            if payload.get("forecast") is None:
                try:
                    payload["forecast"] = self.forecast_fn(payload["lat"], payload["lon"])
                    payload["forecast_status"] = "ok"
                    self.metrics.inc("cloud_forecast_calls")
                except Exception:
                    payload["forecast_status"] = "pending"
            try:
                self.send_fn(payload)
            except Exception:
                self.outbox.mark_failed(event_id, now)
                continue
            self.outbox.mark_sent(event_id, now)
            self.metrics.inc("alerts_sent")
            self.metrics.inc("bytes_up", len(json.dumps(payload).encode()))
            sent += 1
        return sent

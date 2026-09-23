"""The explicit edge→cloud escalation policy.

Rule: only ALERT events leave the device, and the cloud is contacted only for what the
edge cannot produce (forecast) plus delivery to dispatch. Video never leaves the device.
"""
import json
import logging
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
        """Send due ALERTs. The alert goes out first (forecast "pending") and is never held back for
        the forecast; if the forecast then succeeds, a small forecast_update follows (best effort,
        not queued: dispatch already has the alert)."""
        if not self.link.online:
            return 0
        sent = 0
        for event_id, payload, _ in self.outbox.due(now):
            needs_forecast = payload.get("forecast") is None
            if needs_forecast:
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
            if needs_forecast:
                self._send_forecast_update(payload)
        return sent

    def _send_forecast_update(self, alert: dict) -> None:
        try:
            forecast = self.forecast_fn(alert["lat"], alert["lon"])
            self.metrics.inc("cloud_forecast_calls")
        except Exception:
            return
        update = {"type": "forecast_update", "event_id": alert["event_id"],
                  "tower_id": alert.get("tower_id"), "tower_name": alert.get("tower_name"),
                  "severity": alert.get("severity"), "forecast": forecast, "forecast_status": "ok"}
        try:
            self.send_fn(update)
        except Exception:
            logging.getLogger(__name__).warning("forecast update for %s not delivered", alert["event_id"])
            return
        self.metrics.inc("forecast_updates_sent")
        self.metrics.inc("bytes_up", len(json.dumps(update).encode()))

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


FORECAST_UPDATE = "forecast_update"


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
        """Send due items in two passes. Pass 1 sends every due ALERT (forecast "pending"), so no alert
        waits behind another alert's forecast. Pass 2 fetches each forecast and sends a small
        forecast_update; an update that cannot be delivered is queued in the outbox as
        "<event_id>:forecast" and retried like an alert. Returns the number of ALERTs sent."""
        if not self.link.online:
            return 0
        due = self.outbox.due(now)
        alerts = [(e, p) for e, p, _ in due if p.get("type") != FORECAST_UPDATE]
        queued_updates = [(e, p) for e, p, _ in due if p.get("type") == FORECAST_UPDATE]
        sent, need_forecast = 0, []
        for event_id, payload in alerts:
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
                need_forecast.append(payload)
        for payload in need_forecast:
            self._send_forecast_update(payload, now)
        for update_id, update in queued_updates:
            self._deliver_update(update_id, update, now)
        return sent

    def _send_forecast_update(self, alert: dict, now: float) -> None:
        try:
            forecast = self.forecast_fn(alert["lat"], alert["lon"])
            self.metrics.inc("cloud_forecast_calls")
        except Exception:
            return  # dispatch already has the alert, marked "forecast pending"
        update = {"type": FORECAST_UPDATE, "event_id": alert["event_id"],
                  "tower_id": alert.get("tower_id"), "tower_name": alert.get("tower_name"),
                  "severity": alert.get("severity"), "forecast": forecast, "forecast_status": "ok"}
        update_id = f"{alert['event_id']}:forecast"
        if not self._deliver_update(update_id, update, now, queued=False):
            self.outbox.enqueue(update_id, update, now)
            self.outbox.mark_failed(update_id, now)
            logging.getLogger(__name__).warning("forecast update for %s queued for retry", alert["event_id"])

    def _deliver_update(self, update_id: str, update: dict, now: float, queued: bool = True) -> bool:
        try:
            self.send_fn(update)
        except Exception:
            if queued:
                self.outbox.mark_failed(update_id, now)
            return False
        if queued:
            self.outbox.mark_sent(update_id, now)
        self.metrics.inc("forecast_updates_sent")
        self.metrics.inc("bytes_up", len(json.dumps(update).encode()))
        return True

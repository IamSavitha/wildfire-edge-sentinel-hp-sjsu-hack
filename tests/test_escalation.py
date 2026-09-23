from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.schema import Severity

REPORT = {"event_id": "e1", "lat": 37.1, "lon": -121.9, "forecast": None}


def make(online=False, forecast_fn=lambda lat, lon: {"temp_c": [34]}, send_fn=None):
    sent = []
    esc = Escalator(Outbox(":memory:"), Link(online=online), forecast_fn,
                    send_fn or sent.append, Metrics())
    return esc, sent


def test_alert_queues_offline_and_sends_when_online():
    esc, sent = make()
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=1) == 0 and sent == []
    esc.link.online = True
    assert esc.flush(now=2) == 1
    # the alert goes first, never delayed by the forecast; the forecast follows as a small update
    assert sent[0]["event_id"] == "e1" and sent[0]["forecast"] is None and sent[0]["forecast_status"] == "pending"
    assert sent[1]["type"] == "forecast_update" and sent[1]["event_id"] == "e1"
    assert sent[1]["forecast"] == {"temp_c": [34]} and sent[1]["forecast_status"] == "ok"
    assert "thumbnail_jpeg_b64" not in sent[1]
    assert esc.metrics.counters["bytes_up"] > 0 and esc.metrics.counters["forecast_updates_sent"] == 1


def test_alert_is_sent_before_the_forecast_is_requested():
    order = []

    def forecast(lat, lon):
        order.append("forecast")
        return {"temp_c": [30]}
    esc, _ = make(online=True, forecast_fn=forecast, send_fn=lambda p: order.append(p.get("type", "alert")))
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    esc.flush(now=0)
    assert order == ["alert", "forecast", "forecast_update"]


def test_failed_forecast_update_is_queued_and_retried_without_resending_the_alert():
    calls, fail = [], {"on": True}

    def send(p):
        calls.append(p.get("type", "alert"))
        if p.get("type") == "forecast_update" and fail["on"]:
            raise ConnectionError
    esc, _ = make(online=True, send_fn=send)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=0) == 1 and calls == ["alert", "forecast_update"]
    assert esc.outbox.pending_count() == 1  # the update, as "e1:forecast"
    assert esc.outbox.due(now=1) == []      # backoff applies
    (update_id, payload, _), = esc.outbox.due(now=2)
    assert update_id == "e1:forecast" and payload["forecast"] == {"temp_c": [34]}
    fail["on"] = False
    assert esc.flush(now=2) == 0            # no ALERT sent; the update is
    assert calls == ["alert", "forecast_update", "forecast_update"]
    assert esc.outbox.pending_count() == 0 and esc.metrics.counters["forecast_updates_sent"] == 1


def test_all_alerts_go_out_before_any_forecast():
    calls = []

    def forecast(lat, lon):
        calls.append("forecast")
        return {"temp_c": [30]}
    esc, _ = make(online=False, forecast_fn=forecast, send_fn=lambda p: calls.append(p.get("type", "alert")))
    esc.handle({**REPORT, "event_id": "e1"}, Severity.ALERT, now=0)
    esc.handle({**REPORT, "event_id": "e2"}, Severity.ALERT, now=0)
    esc.link.online = True  # link returns with two alerts queued
    assert esc.flush(now=5) == 2
    assert calls == ["alert", "alert", "forecast", "forecast_update", "forecast", "forecast_update"]


def test_non_alerts_never_reach_the_cloud():
    esc, sent = make(online=True)
    for i, sev in enumerate([Severity.IGNORE, Severity.LOG, Severity.MONITOR]):
        esc.handle({**REPORT, "event_id": f"e{i}"}, sev, now=0)
    assert esc.flush(now=1) == 0 and sent == []
    assert len(esc.local_log) == 2


def test_forecast_failure_still_sends_alert():
    def boom(lat, lon):
        raise TimeoutError
    esc, sent = make(online=True, forecast_fn=boom)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=0) == 1 and sent[0]["forecast_status"] == "pending"
    assert len(sent) == 1  # no update without a forecast


def test_send_failure_is_retried_later():
    def down(payload):
        raise ConnectionError
    esc, _ = make(online=True, send_fn=down)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=0) == 0
    assert esc.outbox.pending_count() == 1 and esc.outbox.due(now=1) == []


def test_failed_alert_is_retried_within_seconds_after_the_link_returns():
    from sentinel.outbox import MAX_BACKOFF_S
    assert MAX_BACKOFF_S == 10
    fail = {"on": True}

    def send(p):
        if fail["on"]:
            raise ConnectionError
    esc, _ = make(online=True, send_fn=send)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    t = 0.0
    for _ in range(20):          # a long outage: many failed attempts
        esc.flush(now=t)
        t += 2.0
    fail["on"] = False           # link back
    back = t
    while esc.outbox.pending_count():
        esc.flush(now=t)
        t += 2.0
    assert t - back <= MAX_BACKOFF_S + 2.0

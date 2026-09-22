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
    assert sent[0]["forecast"] == {"temp_c": [34]} and sent[0]["forecast_status"] == "ok"
    assert esc.metrics.counters["bytes_up"] > 0


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


def test_send_failure_is_retried_later():
    def down(payload):
        raise ConnectionError
    esc, _ = make(online=True, send_fn=down)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=0) == 0
    assert esc.outbox.pending_count() == 1 and esc.outbox.due(now=1) == []

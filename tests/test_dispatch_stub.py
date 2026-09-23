import pytest

import scripts.dispatch_stub as stub

ALERT = {"event_id": "e1", "severity": "ALERT", "tower_name": "T", "forecast": None, "forecast_status": "pending"}
UPDATE = {"type": "forecast_update", "event_id": "e1", "severity": "ALERT", "tower_name": "T",
          "forecast": {"temp_c": [30]}, "forecast_status": "ok"}


@pytest.fixture(autouse=True)
def empty():
    stub.REPORTS.clear()
    yield
    stub.REPORTS.clear()


def test_duplicate_alerts_are_stored_once():
    assert stub.receive_report(dict(ALERT)) == "alert"
    assert stub.receive_report(dict(ALERT)) == "duplicate"
    assert len(stub.REPORTS) == 1


def test_forecast_update_fills_in_the_alert():
    stub.receive_report(dict(ALERT))
    assert stub.receive_report(dict(UPDATE)) == "update"
    assert len(stub.REPORTS) == 1
    assert stub.REPORTS[0]["forecast"] == {"temp_c": [30]} and stub.REPORTS[0]["forecast_status"] == "ok"


def test_update_before_its_alert_is_kept():
    stub.receive_report(dict(UPDATE))
    assert len(stub.REPORTS) == 1 and stub.REPORTS[0]["type"] == "forecast_update"
    assert stub.receive_report(dict(ALERT)) == "alert"
    assert len(stub.REPORTS) == 1 and "type" not in stub.REPORTS[0]
    assert stub.REPORTS[0]["forecast_status"] == "ok" and stub.REPORTS[0]["severity"] == "ALERT"

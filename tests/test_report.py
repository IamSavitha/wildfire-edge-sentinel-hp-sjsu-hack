from sentinel.config import Tower
from sentinel.report import build_report, compass, render_text
from sentinel.schema import ContextResult, Severity

TOWER = Tower(id="t1", name="Mt. Umunhum Lookout", lat=37.1606, lon=-121.8983, source="", temp_c=34.0)
CTX = ContextResult(source_type="wildland", smoke_color="grey", attended="no",
                    near_structures=True, near_road=False, size_estimate="medium",
                    description="Grey smoke column rising from a brushy slope above homes.")


def report(ctx=CTX):
    return build_report("e1", TOWER, Severity.ALERT, ctx, "growing", 0.914, now=0.0, thumbnail_b64="")


def test_report_contains_facts():
    r = report()
    assert r["severity"] == "ALERT" and r["lat"] == 37.1606 and r["confidence"] == 0.914
    assert r["detected_at"].startswith("1970-01-01T00:00:00")
    assert r["forecast"] is None


def test_render_text_without_forecast():
    text = render_text(report())
    assert "ALERT" in text and "Wildland" in text and "37.1606" in text and "homes" in text


def test_render_text_with_forecast():
    r = report()
    r["forecast"] = {"temp_c": [34, 36, 38], "wind_mph": [15, 16, 18], "wind_dir_deg": [270, 270, 280]}
    assert "38°C" in render_text(r) and "from W" in render_text(r)


def test_render_text_forecast_pending():
    r = report()
    r["forecast_status"] = "pending"
    assert "Forecast pending" in render_text(r)


def test_report_without_context():
    r = report(ctx=None)
    assert r["source_type"] == "unknown" and "unavailable" in r["description"]


def test_compass():
    assert compass(0) == "N" and compass(270) == "W" and compass(359) == "N" and compass(135) == "SE"

import base64
import logging

import pytest

from sentinel.notify import NotifyError, NtfyNotifier, ascii_header, redact_topic_url

TOPIC = "https://ntfy.example.org/sentinel-test-topic-1234"
JPEG = b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"
REPORT = {
    "event_id": "abc123def456", "tower_id": "live-camera", "tower_name": "Live camera",
    "lat": 37.1606, "lon": -121.8983, "severity": "ALERT", "confidence": 0.91, "trend": "growing",
    "temp_c": 25.0, "source_type": "wildland", "description": "Grey smoke rising — near a ridge…",
    "forecast": None, "forecast_status": "pending",
    "thumbnail_jpeg_b64": base64.b64encode(JPEG).decode(),
}


class Resp:
    def __init__(self, status_code=200, text="{}"):
        self.status_code = status_code
        self.text = text


class FakePost:
    def __init__(self, *statuses, exc=None):
        self.statuses = list(statuses) or [200]
        self.exc = exc
        self.calls = []

    def __call__(self, url, *, content, headers, timeout):
        self.calls.append({"url": url, "content": content, "headers": dict(headers), "timeout": timeout})
        if self.exc is not None:
            raise self.exc
        return Resp(self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0])


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(post=None, clock=None, **kw):
    post = post or FakePost()
    return NtfyNotifier(TOPIC, post=post, clock=clock or Clock(), **kw), post


def test_alert_uploads_snapshot_with_urgent_headers():
    n, post = make()
    assert n.notify_alert(REPORT) == "sent"
    [call] = post.calls
    assert call["url"] == TOPIC
    assert call["content"] == JPEG
    h = call["headers"]
    assert h["Priority"] == "5"
    assert h["Tags"] == "fire,rotating_light"
    assert h["Filename"].endswith(".jpg")
    assert h["Title"] == "WILDFIRE ALERT - Live camera"
    assert "[ALERT] Wildland smoke/fire at Live camera" in h["Message"]
    assert "Authorization" not in h
    assert n.stats()["sent"] == 1


def test_headers_are_ascii_even_for_non_ascii_report_text():
    n, post = make()
    n.notify_alert({**REPORT, "tower_name": "Mirador Peña", "description": "Humo gris — 30°C 🔥"})
    for key, value in post.calls[0]["headers"].items():
        value.encode("ascii")                     # would raise on any non-ASCII byte
        assert "\n" not in value and "\r" not in value, key
    h = post.calls[0]["headers"]
    assert "Pena" in h["Title"] and "Humo gris - 30C" in h["Message"]


def test_no_snapshot_sends_the_full_utf8_text_as_the_body():
    n, post = make()
    n.notify_alert({**REPORT, "thumbnail_jpeg_b64": "", "description": "Humo gris — 30°C"})
    call = post.calls[0]
    assert "Filename" not in call["headers"]
    body = call["content"].decode("utf-8")
    assert "Humo gris — 30°C" in body and "°C" in body
    assert call["headers"]["Content-Type"].startswith("text/plain")


def test_rejected_attachment_falls_back_to_text_only():
    n, post = make(FakePost(400, 200))
    assert n.notify_alert(REPORT) == "sent"
    assert len(post.calls) == 2
    assert "Filename" in post.calls[0]["headers"] and "Filename" not in post.calls[1]["headers"]


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502])
def test_failed_publish_raises_for_the_outbox_to_retry(status):
    n, post = make(FakePost(status))
    with pytest.raises(NotifyError):
        n.notify_alert(REPORT)
    assert n.stats()["failed"] == 1 and n.stats()["sent"] == 0


def test_network_error_raises_a_redacted_error():
    n, _ = make(FakePost(exc=OSError(f"cannot reach {TOPIC}")))
    with pytest.raises(NotifyError) as ei:
        n.notify_alert(REPORT)
    assert "sentinel-test-topic" not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_a_failure_does_not_start_the_rate_limit_window():
    post = FakePost(500, 200)
    n, _ = make(post)
    with pytest.raises(NotifyError):
        n.notify_alert(REPORT)
    assert n.notify_alert(REPORT) == "sent"


def test_alerts_are_never_rate_limited():
    clock = Clock()
    n, post = make(clock=clock, min_interval_s=30)
    for _ in range(5):
        assert n.notify_alert(REPORT) == "sent"
        clock.t += 2
    assert len(post.calls) == 5 and n.stats()["suppressed"] == 0


def test_test_pushes_are_rate_limited_without_raising(caplog):
    clock = Clock()
    n, post = make(clock=clock, min_interval_s=30)
    assert n.send_test() == "sent"
    clock.t += 10
    with caplog.at_level(logging.INFO, logger="sentinel.notify"):
        assert n.send_test() == "suppressed"
    assert "suppressed" in caplog.text and "sentinel-test-topic" not in caplog.text
    assert n.notify_alert(REPORT) == "sent"          # a real ALERT is not held back by a test push
    clock.t += 5
    assert n.send_test() == "suppressed"             # nor does a test push buzz right after an ALERT
    clock.t += 30
    assert n.send_test() == "sent"
    assert len(post.calls) == 3 and n.stats()["suppressed"] == 2


def test_alert_burst_guard_defers_instead_of_dropping():
    clock = Clock()
    n, post = make(clock=clock, burst_max=5, burst_window_s=60)
    for _ in range(5):
        assert n.notify_alert(REPORT) == "sent"
        clock.t += 1
    with pytest.raises(NotifyError, match="burst limit"):
        n.notify_alert(REPORT)                       # raises: the outbox keeps it and retries
    assert len(post.calls) == 5 and n.stats()["deferred"] == 1 and n.stats()["failed"] == 0
    clock.t += 56                                    # the first push leaves the 60 s window
    assert n.notify_alert(REPORT) == "sent"


def test_send_test_is_a_plain_text_push():
    n, post = make()
    assert n.send_test() == "sent"
    call = post.calls[0]
    assert "Filename" not in call["headers"]
    assert "test" in call["headers"]["Title"].lower()
    assert call["content"].decode()


def test_token_goes_in_a_bearer_header():
    n, post = make(token="tk_secret")
    n.send_test()
    assert post.calls[0]["headers"]["Authorization"] == "Bearer tk_secret"
    assert "tk_secret" not in repr(n) and "tk_secret" not in n.describe()


def test_from_env():
    assert NtfyNotifier.from_env({}) is None
    assert NtfyNotifier.from_env({"NTFY_TOPIC_URL": "  "}) is None
    assert NtfyNotifier.from_env({"NTFY_TOPIC_URL": "not a url"}) is None
    assert NtfyNotifier.from_env({"NTFY_TOPIC_URL": "https://ntfy.sh/"}) is None
    n = NtfyNotifier.from_env({"NTFY_TOPIC_URL": TOPIC, "NTFY_TOKEN": "tk_x"}, post=FakePost())
    assert n.topic_url == TOPIC and n.token == "tk_x"


def test_redaction_shows_host_and_four_chars():
    assert redact_topic_url(TOPIC) == "ntfy.example.org/sent…"
    assert redact_topic_url("https://push.example/abcdefgh") == "push.example"   # short topic: host only
    assert redact_topic_url("https://ntfy.sh/ab") == "ntfy.sh"
    n, _ = make()
    assert TOPIC not in n.describe() and "sentinel-test-topic" not in repr(n)
    assert n.host == "ntfy.example.org"
    assert "sent" not in n.stats()["host"] and "target" not in n.stats()


def test_httpx_request_logs_are_silenced():
    assert logging.getLogger("httpx").level >= logging.WARNING      # httpx logs full URLs at INFO


def test_phone_text_does_not_claim_no_connectivity():
    n, post = make()
    n.notify_alert(REPORT)                           # forecast_status "pending", but we are online
    assert "no connectivity" not in post.calls[0]["headers"]["Message"]
    assert "Wildland smoke/fire" in post.calls[0]["headers"]["Message"]


def test_ascii_header_limits_length_and_strips_control_chars():
    s = ascii_header("a\r\nb\tc — d " + "x" * 5000, limit=100)
    assert s.startswith("a b c - d") and len(s) <= 100

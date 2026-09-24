"""Phone push alerts through ntfy (https://ntfy.sh or a self-hosted ntfy server).

One ALERT = one HTTP request = one buzz. The JPEG snapshot is the request body (ntfy's attachment
upload: body + `Filename` header), so the title and the dispatch text must travel in headers. HTTP
headers are latin-1 only and proxies mangle anything else, so both are transliterated to plain ASCII
(`ascii_header`: "—" -> "-", "°C" -> "C", accents dropped, emoji removed); the emoji on the phone
come from the `Tags` header (`fire`, `rotating_light` render as 🔥🚨 in front of the title). With no
snapshot, the full UTF-8 report is sent as the body instead. If the server refuses the attachment
(4xx other than auth/rate-limit, e.g. attachments disabled on a self-hosted server), the alert is
re-sent text-only so it still arrives.

The topic URL is a secret (anyone who knows it can read the alerts): it is read from the
environment only (`NTFY_TOPIC_URL`, optional `NTFY_TOKEN` access token) and logged redacted.
"""
import base64
import logging
import os
import re
import threading
import time
import unicodedata
from typing import Callable, Mapping
from urllib.parse import urlparse

import httpx

from sentinel.report import render_text

log = logging.getLogger(__name__)

ALERT_TAGS = ("fire", "rotating_light")
TEST_TAGS = ("white_check_mark",)
TITLE_LIMIT = 250
MESSAGE_LIMIT = 3500      # ntfy keeps messages up to 4096 bytes
_REPLACE = {"—": "-", "–": "-", "°": "", "…": "...", "’": "'", "‘": "'", "“": '"', "”": '"', "×": "x"}


class NotifyError(RuntimeError):
    """A push that did not go out (network, auth, server error). The outbox retries it."""


def ascii_header(text: str, limit: int = MESSAGE_LIMIT) -> str:
    """Plain printable ASCII on one line, safe for any HTTP header."""
    for k, v in _REPLACE.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text[:limit].rstrip()


def redact_topic_url(url: str) -> str:
    """host + the first 4 characters of the topic, for logs and the UI."""
    p = urlparse(url)
    topic = p.path.strip("/").split("/")[-1] if p.path.strip("/") else ""
    return f"{p.hostname or '?'}/{topic[:4]}…"


def _valid_topic_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https") and bool(p.hostname) and bool(p.path.strip("/"))


class NtfyNotifier:
    def __init__(self, topic_url: str, token: str | None = None, min_interval_s: float = 30.0,
                 clock: Callable[[], float] = time.time, post: Callable = httpx.post,
                 timeout_s: float = 10.0):
        if not _valid_topic_url(topic_url):
            raise ValueError("ntfy topic URL must look like https://<host>/<topic>")
        self.topic_url = topic_url
        self.token = token
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.post = post
        self.timeout_s = timeout_s
        self.target = redact_topic_url(topic_url)
        self._lock = threading.Lock()          # one push at a time; guards the rate-limit window
        self._last_sent: float | None = None
        self._counts = {"sent": 0, "suppressed": 0, "failed": 0}
        self.last_error: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **kw) -> "NtfyNotifier | None":
        env = os.environ if env is None else env
        url = (env.get("NTFY_TOPIC_URL") or "").strip()
        if not url:
            return None
        if not _valid_topic_url(url):
            log.warning("NTFY_TOPIC_URL is set but is not an http(s)://<host>/<topic> URL; phone alerts off")
            return None
        return cls(url, token=(env.get("NTFY_TOKEN") or "").strip() or None, **kw)

    def __repr__(self) -> str:
        return f"NtfyNotifier({self.describe()})"

    def describe(self) -> str:
        return self.target + (" (token)" if self.token else "")

    def stats(self) -> dict:
        with self._lock:
            return {**self._counts, "target": self.target, "min_interval_s": self.min_interval_s,
                    "last_error": self.last_error}

    # ------------------------------------------------------------ public API

    def notify_alert(self, report: dict) -> str:
        """Push an ALERT report. Returns "sent" or "suppressed" (rate limit); raises NotifyError
        when it could not be delivered, so the caller's outbox retries it."""
        title = f"🔥 WILDFIRE ALERT — {report.get('tower_name') or report.get('tower_id') or 'camera'}"
        try:
            snapshot = base64.b64decode(report.get("thumbnail_jpeg_b64") or "", validate=True) or None
        except (ValueError, TypeError):
            snapshot = None
        return self._publish(title, render_text(report), 5, ALERT_TAGS, snapshot,
                             f"sentinel-{report.get('event_id', 'alert')}.jpg")

    def send_test(self) -> str:
        """A rehearsal push (same rate limit as alerts)."""
        return self._publish("Wildfire Edge Sentinel test",
                             "Test alert: phone alerts from the Wildfire Edge Sentinel demo work.",
                             3, TEST_TAGS, None, None)

    # ------------------------------------------------------------ internals

    def _publish(self, title: str, message: str, priority: int, tags, snapshot: bytes | None,
                 filename: str | None) -> str:
        with self._lock:
            now = self.clock()
            if self._last_sent is not None and now - self._last_sent < self.min_interval_s:
                self._counts["suppressed"] += 1
                log.info("push to %s suppressed by the rate limit (%.0f s since the last push, min %.0f s)",
                         self.target, now - self._last_sent, self.min_interval_s)
                return "suppressed"
            headers = {"Title": ascii_header(title, TITLE_LIMIT), "Priority": str(priority),
                       "Tags": ",".join(tags)}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            try:
                if snapshot:
                    status = self._send(snapshot, {**headers, "Message": ascii_header(message),
                                                   "Filename": filename or "snapshot.jpg"})
                    if 400 <= status < 500 and status not in (401, 403, 429):
                        log.warning("ntfy %s refused the snapshot (HTTP %d); sending text only",
                                    self.target, status)
                        status = self._send_text(message, headers)
                else:
                    status = self._send_text(message, headers)
                if status >= 400:
                    raise NotifyError(f"ntfy {self.target} answered HTTP {status}")
            except NotifyError as exc:
                self._fail(str(exc))
                raise
            except Exception as exc:  # noqa: BLE001 - network errors carry the URL: re-raise redacted
                msg = f"ntfy {self.target} unreachable: {type(exc).__name__}"
                self._fail(msg)
                raise NotifyError(msg) from None
            self._last_sent = now
            self._counts["sent"] += 1
            self.last_error = None
            log.info("push sent to %s (%s)", self.target, headers["Title"])
            return "sent"

    def _send_text(self, message: str, headers: dict) -> int:
        body = message[:MESSAGE_LIMIT].encode("utf-8")
        return self._send(body, {**headers, "Content-Type": "text/plain; charset=utf-8"})

    def _send(self, content: bytes, headers: dict) -> int:
        return int(self.post(self.topic_url, content=content, headers=headers, timeout=self.timeout_s).status_code)

    def _fail(self, msg: str) -> None:
        self._counts["failed"] += 1
        self.last_error = msg
        log.warning("push failed: %s", msg)

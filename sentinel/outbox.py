"""Durable SQLite outbox for ALERT reports awaiting connectivity."""
import json
import sqlite3
import threading

MAX_BACKOFF_S = 300


class Outbox:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS outbox ("
                " event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,"
                " attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL,"
                " sent_at REAL)")
            self.db.commit()

    def enqueue(self, event_id: str, payload: dict, now: float) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO outbox (event_id, payload, next_attempt_at) VALUES (?, ?, ?)",
                            (event_id, json.dumps(payload), now))
            self.db.commit()

    def due(self, now: float) -> list[tuple[str, dict, int]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT event_id, payload, attempts FROM outbox"
                " WHERE sent_at IS NULL AND next_attempt_at <= ? ORDER BY next_attempt_at",
                (now,)).fetchall()
        return [(e, json.loads(p), a) for e, p, a in rows]

    def mark_sent(self, event_id: str, now: float) -> None:
        with self.lock:
            self.db.execute("UPDATE outbox SET sent_at = ? WHERE event_id = ?", (now, event_id))
            self.db.commit()

    def mark_failed(self, event_id: str, now: float) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE outbox SET attempts = attempts + 1,"
                " next_attempt_at = ? + min(?, 1 << (attempts + 1)) WHERE event_id = ?",
                (now, MAX_BACKOFF_S, event_id))
            self.db.commit()

    def pending_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent_at IS NULL").fetchone()[0]

    def sent_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent_at IS NOT NULL").fetchone()[0]

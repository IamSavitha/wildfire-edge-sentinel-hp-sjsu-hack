"""Incident store for the map: one row per fire incident, updated as more frames confirm it.

A new report joins an existing active incident when it comes from the same camera, within
MERGE_WINDOW_S of that incident's last update and within MERGE_BEARING_DEG of its bearing (or,
for reports with no bearing, within MERGE_KM of its location). A sighting from a *different* tower
joins when its bearing line crosses one of the incident's bearing lines in front of both cameras
(within CROSS_MAX_KM) and, once the incident is triangulated, within CROSS_AGREE_KM of that fix.
Severity only ever goes up."""
import json
import math
import sqlite3
import threading
import uuid

from sentinel.geo import EARTH_KM, ray_intersection

SEVERITY_RANK = {"IGNORE": 0, "LOG": 1, "MONITOR": 2, "ALERT": 3}
MERGE_WINDOW_S = 2 * 3600
MERGE_BEARING_DEG = 12.0
MERGE_KM = 3.0
HISTORY_CAP = 30
CROSS_WINDOW_S = 30 * 60
CROSS_MAX_KM = 60.0
CROSS_AGREE_KM = 5.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


def _bearing_gap(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


class IncidentStore:
    def __init__(self, path: str):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS incidents (id TEXT PRIMARY KEY, seq INTEGER,"
                            " created_at REAL, updated_at REAL, camera_id TEXT, status TEXT, data TEXT)")
            self.db.commit()

    def _rows(self, where: str = "", args=()) -> list[dict]:
        cur = self.db.execute(f"SELECT data FROM incidents {where} ORDER BY updated_at DESC", args)
        return [json.loads(r[0]) for r in cur.fetchall()]

    def all(self) -> list[dict]:
        with self.lock:
            return self._rows()

    def get(self, incident_id: str) -> dict | None:
        with self.lock:
            rows = self._rows("WHERE id = ?", (incident_id,))
        return rows[0] if rows else None

    def save(self, inc: dict) -> dict:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO incidents VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (inc["id"], inc["seq"], inc["created_at"], inc["updated_at"],
                             inc.get("camera_id"), inc["status"], json.dumps(inc)))
            self.db.commit()
        return inc

    def delete(self, incident_id: str) -> None:
        with self.lock:
            self.db.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
            self.db.commit()

    def clear(self) -> None:
        with self.lock:
            self.db.execute("DELETE FROM incidents")
            self.db.commit()

    def find_match(self, obs: dict, now: float) -> dict | None:
        with self.lock:
            # both sides bounded: a replayed recording (old timestamps) must not join today's incident
            rows = self._rows("WHERE status = 'active' AND updated_at >= ? AND created_at <= ?",
                              (now - MERGE_WINDOW_S, now + MERGE_WINDOW_S))
        sight = obs.get("sighting")
        for inc in rows:
            own = (inc.get("sightings") or {}).get(obs.get("camera_id"))
            if inc.get("camera_id") == obs.get("camera_id") or own:
                ref = (own or {}).get("bearing_deg", inc.get("bearing_deg"))
                if obs.get("bearing_deg") is not None and ref is not None:
                    if _bearing_gap(obs["bearing_deg"], ref) <= MERGE_BEARING_DEG:
                        return inc
                elif haversine_km(obs["lat"], obs["lon"], inc["lat"], inc["lon"]) <= MERGE_KM:
                    return inc
                continue
            if sight and abs(inc["updated_at"] - now) <= CROSS_WINDOW_S and self._crosses(sight, inc):
                return inc
        return None

    @staticmethod
    def _crosses(sight: dict, inc: dict) -> bool:
        """Does a new tower's bearing line cross this incident's bearing line(s) in front of both towers?"""
        for other in (inc.get("sightings") or {}).values():
            if other["camera_id"] == sight["camera_id"]:
                continue
            hit = ray_intersection(sight["lat"], sight["lon"], sight["bearing_deg"],
                                   other["lat"], other["lon"], other["bearing_deg"], CROSS_MAX_KM)
            if hit is None:
                continue
            if len(inc.get("sightings") or {}) < 2 or inc.get("loc_source") != "triangulated":
                return True
            if haversine_km(hit[0], hit[1], inc["lat"], inc["lon"]) <= CROSS_AGREE_KM:
                return True
        return False

    def record(self, obs: dict, now: float) -> tuple[dict, bool]:
        """Create or update an incident from one observation. Returns (incident, created)."""
        inc = self.find_match(obs, now)
        entry = {"at": now, "severity": obs["severity"], "source_type": obs.get("source_type"),
                 "conf": obs.get("conf"), "frame": obs.get("frame_name"), "camera_id": obs.get("camera_id")}
        sight = obs.pop("sighting", None)
        if inc is None:
            with self.lock:
                seq = (self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM incidents").fetchone()[0]) + 1
            inc = {**obs, "id": uuid.uuid4().hex[:12], "seq": seq, "created_at": now, "updated_at": now,
                   "status": "active", "detections": 1, "history": [entry],
                   "sightings": {sight["camera_id"]: sight} if sight else {}}
            if not inc.get("name"):
                inc["name"] = f"{obs.get('site_short') or 'Unplaced'} Incident {seq}"
            return self.save(inc), True
        keep_rank = SEVERITY_RANK.get(inc["severity"], 0) >= SEVERITY_RANK.get(obs["severity"], 0)
        manual_loc = inc.get("loc_source") in ("map_pin", "triangulated")
        new_camera = obs.get("camera_id") != inc.get("camera_id")
        # the first camera stays the incident's primary; another tower only adds its sighting
        skip = {"name", "camera_id", "camera_name", "bearing_deg", "bearing_lo", "bearing_hi", "range_km",
                "site_short"} if new_camera else {"name"}
        updated = {**inc, **{k: v for k, v in obs.items() if v is not None and k not in skip}}
        if sight:
            updated["sightings"] = {**(inc.get("sightings") or {}), sight["camera_id"]: sight}
        if keep_rank:
            updated["severity"] = inc["severity"]
        if manual_loc or new_camera:     # an operator pin or a triangulated fix wins over one camera's guess
            for k in ("lat", "lon", "loc_source", "loc_note", "county"):
                updated[k] = inc.get(k)
        updated.update(updated_at=now, detections=inc.get("detections", 1) + 1,
                       history=(inc.get("history", []) + [entry])[-HISTORY_CAP:])
        return self.save(updated), False

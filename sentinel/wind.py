"""Wind for the incident map, from the tower first and the cloud last.

1. The tower's own weather station (HPWREN sites carry a Vaisala WXT536). On this box there is no
   serial link to a real station, so `SensorEmulator` replays real HPWREN readings captured once
   (`wx_seed.json`) with a small random walk, and every reading it produces is marked emulated.
2. A manual reading entered by the operator.
3. The last forecast cached from the cloud. The cloud is only contacted when an ALERT escalates
   (sentinel.webapp_logic.escalate); this module never makes a network call itself.
"""
import json
import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import Callable

from sentinel.geo import direction_spread, format_wxt, parse_wxt

SENSOR_STALE_S = 15 * 60
log = logging.getLogger(__name__)
MPH_TO_MPS = 0.44704


class SensorEmulator:
    """Stands in for the serial feed from each tower's weather station."""

    def __init__(self, seed_path: str | Path, clock: Callable[[], float] = time.time, rng=None):
        seed = json.loads(Path(seed_path).read_text())
        self.source = seed.get("source")
        self.captured = seed.get("captured_utc")
        self.base = {}
        for k, v in seed["readings"].items():
            try:
                self.base[k] = parse_wxt(v)
            except ValueError as exc:      # a station with no valid mean wind is left out, not fatal
                log.warning("sensor seed %s skipped: %s", k, exc)
        self.clock = clock
        self.rng = rng or random.Random(7)
        self.state = {k: dict(v) for k, v in self.base.items()}
        self.raw: dict[str, str] = {}
        self.lock = threading.Lock()
        self.step()

    def step(self) -> None:
        """One new reading per station: a bounded random walk around the captured reading."""
        now = self.clock()
        with self.lock:
            for k, b in self.base.items():
                s = self.state[k]
                s["dir_from_deg"] = (s["dir_from_deg"] + self.rng.uniform(-6, 6)) % 360
                drift = (s["dir_from_deg"] - b["dir_from_deg"] + 180) % 360 - 180
                if abs(drift) > 30:                      # stay within 30 deg of the real reading
                    s["dir_from_deg"] = (b["dir_from_deg"] + math.copysign(30, drift)) % 360
                s["speed_mps"] = min(max(s["speed_mps"] + self.rng.uniform(-0.3, 0.3),
                                         b["speed_mps"] * 0.6), b["speed_mps"] * 1.4 + 0.5)
                s["speed_min_mps"] = max(0.0, s["speed_mps"] - (b["speed_mps"] - b.get("speed_min_mps", b["speed_mps"])))
                s["gust_mps"] = s["speed_mps"] + (b.get("gust_mps", b["speed_mps"]) - b["speed_mps"])
                half = direction_spread(b.get("dir_min_deg"), b.get("dir_max_deg"))
                s["dir_min_deg"] = (s["dir_from_deg"] - half) % 360
                s["dir_max_deg"] = (s["dir_from_deg"] + half) % 360
                s.setdefault("temp_c", b.get("temp_c", 20.0))
                s.setdefault("rh_pct", b.get("rh_pct", 30.0))
                self.raw[k] = format_wxt(now, s)

    def read(self, station: str) -> str | None:
        with self.lock:
            return self.raw.get(station)

    def run(self, stop: threading.Event, interval_s: float = 10.0) -> None:
        while not stop.wait(interval_s):
            self.step()


class WindService:
    def __init__(self, sensor_read: Callable[[str], str | None] | None, cache_path: str | Path,
                 clock: Callable[[], float] = time.time, sensor_label: str = "on-site sensor"):
        self.sensor_read = sensor_read
        self.sensor_label = sensor_label
        self.cache_path = Path(cache_path)
        self.clock = clock
        self.lock = threading.Lock()
        self.forecasts: dict = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    # -------------------------------------------------- forecast cache (filled only by escalation)

    def remember_forecast(self, key: str, forecast: dict) -> None:
        with self.lock:
            self.forecasts[key] = {"fetched_at": self.clock(), "forecast": forecast}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.forecasts))
            tmp.replace(self.cache_path)

    def _from_forecast(self, key: str) -> dict | None:
        with self.lock:
            entry = self.forecasts.get(key)
        if not entry:
            return None
        fc = entry["forecast"]
        winds, dirs = fc.get("wind_mph") or [], fc.get("wind_dir_deg") or []
        if not winds or not dirs or winds[0] is None or dirs[0] is None:
            return None
        age = self.clock() - entry["fetched_at"]
        # hourly forecast starting at the fetch hour: pick the hour we are in now
        i = min(max(int(age // 3600), 0), min(len(winds), len(dirs)) - 1)
        return {"source": "forecast_cache", "label": "cached cloud forecast",
                "dir_from_deg": float(dirs[i]), "speed_mps": float(winds[i]) * MPH_TO_MPS,
                "gust_mps": None, "spread_deg": 20.0, "observed_at": entry["fetched_at"],
                "age_s": age, "emulated": False, "stale": age > 6 * 3600,
                "outlook": [{"h": h, "dir_from_deg": d, "speed_mps": w * MPH_TO_MPS}
                            for h, (w, d) in enumerate(zip(winds, dirs)) if w is not None and d is not None]}

    # -------------------------------------------------- the one call the app makes

    def sensor(self, station: str | None) -> dict | None:
        if not station or self.sensor_read is None:
            return None
        raw = self.sensor_read(station)
        if not raw:
            return None
        try:
            r = parse_wxt(raw)
        except ValueError:
            return None
        age = self.clock() - r["observed_at"]
        return {"source": "sensor", "label": self.sensor_label, "station": station,
                "dir_from_deg": r["dir_from_deg"], "speed_mps": r["speed_mps"],
                "gust_mps": r.get("gust_mps"), "temp_c": r.get("temp_c"), "rh_pct": r.get("rh_pct"),
                "spread_deg": direction_spread(r.get("dir_min_deg"), r.get("dir_max_deg")),
                "observed_at": r["observed_at"], "age_s": age, "stale": age > SENSOR_STALE_S,
                "emulated": self.sensor_label.startswith("emulated"), "raw": raw}

    def wind_for(self, station: str | None, forecast_key: str | None, manual: dict | None = None) -> dict | None:
        """Best available wind: fresh sensor > manual > cached forecast > stale sensor > None."""
        s = self.sensor(station)
        if s and not s["stale"]:
            return s
        if manual:
            return {"source": "manual", "label": "entered by operator", "spread_deg": 20.0,
                    "gust_mps": None, "emulated": False, "stale": False,
                    "age_s": self.clock() - manual.get("observed_at", self.clock()), **manual}
        return (self._from_forecast(forecast_key) if forecast_key else None) or s

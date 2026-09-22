"""Cloud-side calls, used ONLY by the escalator when the link is up."""
import json
from pathlib import Path

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def parse_open_meteo(data: dict) -> dict:
    h = data["hourly"]
    return {"time": h["time"], "temp_c": h["temperature_2m"],
            "wind_mph": h["wind_speed_10m"], "wind_dir_deg": h["wind_direction_10m"]}


def fetch_forecast(lat: float, lon: float, hours: int = 6, timeout_s: float = 3.0) -> dict:
    params = {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
              "wind_speed_unit": "mph", "forecast_hours": hours}
    r = httpx.get(OPEN_METEO_URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    return parse_open_meteo(r.json())


def send_dispatch(url: str, payload: dict, timeout_s: float = 5.0) -> None:
    httpx.post(url, json=payload, timeout=timeout_s).raise_for_status()


def fetch_burn_schedule(path: str | Path = "config/burn_schedule.json") -> set[str]:
    """Simulated cloud service: tower ids with a prescribed burn scheduled today."""
    p = Path(path)
    return set(json.loads(p.read_text())) if p.exists() else set()

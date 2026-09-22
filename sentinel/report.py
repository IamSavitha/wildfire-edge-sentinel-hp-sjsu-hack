"""Dispatch report: facts templated by code, scene description from the VLM."""
from datetime import datetime, timezone

from sentinel.config import Tower
from sentinel.schema import ContextResult, Severity

FALLBACK_DESCRIPTION = "Context model unavailable; detector-only assessment."
_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(deg: float) -> str:
    return _DIRS[int((deg % 360) / 45 + 0.5) % 8]


def build_report(event_id: str, tower: Tower, severity: Severity, ctx: ContextResult | None,
                 trend: str | None, confidence: float, now: float, thumbnail_b64: str) -> dict:
    return {
        "event_id": event_id,
        "tower_id": tower.id,
        "tower_name": tower.name,
        "lat": tower.lat,
        "lon": tower.lon,
        "detected_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "severity": severity.name,
        "confidence": round(confidence, 3),
        "trend": trend,
        "temp_c": tower.temp_c,
        "source_type": ctx.source_type if ctx else "unknown",
        "context": ctx.model_dump(exclude={"description"}) if ctx else None,
        "description": ctx.description if ctx else FALLBACK_DESCRIPTION,
        "forecast": None,
        "forecast_status": "not_requested",
        "thumbnail_jpeg_b64": thumbnail_b64,
    }


def render_text(r: dict) -> str:
    parts = [
        f"[{r['severity']}] {r['source_type'].replace('_', ' ').title()} smoke/fire at "
        f"{r['tower_name']} ({r['lat']:.4f}, {r['lon']:.4f}).",
        f"Confidence {r['confidence']:.2f}. Trend: {r['trend'] or 'unknown'}. Local temp {r['temp_c']:.0f}°C.",
        r["description"],
    ]
    fc = r.get("forecast")
    if fc:
        parts.append(f"Forecast: up to {max(fc['temp_c']):.0f}°C, wind {fc['wind_mph'][0]:.0f} mph "
                     f"from {compass(fc['wind_dir_deg'][0])}.")
    elif r.get("forecast_status") == "pending":
        parts.append("Forecast pending (no connectivity at detection time).")
    return " ".join(parts)

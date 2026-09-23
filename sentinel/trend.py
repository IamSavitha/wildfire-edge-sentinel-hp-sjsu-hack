"""Growth trend between two observations of the same smoke region."""
from typing import Literal

Trend = Literal["growing", "static", "dissipating"]


def classify_trend(area_before: float, area_after: float,
                   grow: float = 1.3, shrink: float = 0.7) -> Trend:
    if area_before <= 0:
        return "growing" if area_after > 0 else "static"
    ratio = area_after / area_before
    if ratio >= grow:
        return "growing"
    if ratio <= shrink:
        return "dissipating"
    return "static"

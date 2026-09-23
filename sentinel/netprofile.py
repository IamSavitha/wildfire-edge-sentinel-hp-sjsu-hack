"""Deterministic network-link model for the cloud-only baseline and the outage scenarios.

This is a MODEL layered on top of REAL measured cloud inference latency: we do not throttle the
Nano's real network (that would cut the SSH/Tailscale link we operate it through).

    serialize_s = bytes * 8 / uplink_bps / (1 - loss)     how long a payload occupies the uplink
    rtt_s(n)    = n * rtt / (1 - loss)                    n round trips (not occupying the uplink)
    transfer_s  = serialize_s + rtt_s(n)                  one message, delivered

Dividing by (1 - loss) charges the expected retransmissions. This loss model is optimistic, most of
all for GEO satellite: real TCP throughput on a lossy high-latency link is far below the nominal rate.
Round trips per message follow the shipped code: the edge posts each ALERT with a one-shot
`httpx.post` (new HTTPS connection: TCP handshake + TLS + request = EDGE_POST_RTTS), while the cloud
client keeps one HTTPS connection alive, so each cloud request costs CLOUD_REQUEST_RTTS. Responses
are ~1 KB, so the downlink rate is kept for reference only. The `outage` profile never delivers
(math.inf). Link parameters are round, typical figures for each link class, not measurements.
"""
import math
from dataclasses import dataclass

EDGE_POST_RTTS = 3      # new HTTPS connection per alert POST (sentinel/cloud.py send_dispatch)
CLOUD_REQUEST_RTTS = 1  # persistent keep-alive connection (openai client)


@dataclass(frozen=True)
class LinkProfile:
    name: str
    uplink_kbps: float
    downlink_kbps: float
    rtt_ms: float
    loss: float = 0.0

    @property
    def is_down(self) -> bool:
        return self.uplink_kbps <= 0 or self.loss >= 1


PROFILES: dict[str, LinkProfile] = {p.name: p for p in (
    LinkProfile("fiber", 50_000, 100_000, 20, 0.0),
    LinkProfile("lte", 5_000, 20_000, 60, 0.005),
    LinkProfile("rural_cellular", 1_000, 5_000, 150, 0.02),
    LinkProfile("satellite_geo", 512, 2_000, 650, 0.01),
    LinkProfile("outage", 0, 0, math.inf, 1.0),
)}


def get_profile(name: str) -> LinkProfile:
    try:
        return PROFILES[name.strip()]
    except KeyError:
        raise ValueError(f"unknown link profile {name!r}; known: {sorted(PROFILES)}") from None


def parse_profiles(csv: str) -> list[LinkProfile]:
    return [get_profile(n) for n in csv.split(",") if n.strip()]


def serialize_s(nbytes: int | float, profile: LinkProfile) -> float:
    """Seconds `nbytes` occupy the uplink (inflated by expected retransmissions)."""
    if profile.is_down:
        return math.inf
    return nbytes * 8 / (profile.uplink_kbps * 1000) / (1 - profile.loss)


def rtt_s(profile: LinkProfile, n: int = 1) -> float:
    if profile.is_down:
        return math.inf
    return n * profile.rtt_ms / 1000 / (1 - profile.loss)


def transfer_s(nbytes: int | float, profile: LinkProfile, rtts: int = 1) -> float:
    """Seconds to deliver `nbytes` upstream over `profile`: serialization + `rtts` round trips."""
    return serialize_s(nbytes, profile) + rtt_s(profile, rtts)


class Outages:
    """Link-down windows [start_s, end_s) in simulated seconds; overlapping windows are merged."""

    def __init__(self, windows=()):
        merged: list[list[float]] = []
        for start, end in sorted((float(s), float(e)) for s, e in windows):
            if end <= start:
                raise ValueError(f"outage window must end after it starts: ({start}, {end})")
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        self.windows: list[tuple[float, float]] = [(s, e) for s, e in merged]

    def __bool__(self) -> bool:
        return bool(self.windows)

    def is_down(self, t: float) -> bool:
        return any(s <= t < e for s, e in self.windows)

    def next_up(self, t: float) -> float:
        """t itself when the link is up, else the end of the window covering t."""
        for s, e in self.windows:
            if s <= t < e:
                return e
        return t


def parse_outage(text: str) -> tuple[float, float]:
    """"start,end" in simulated seconds -> (start, end)."""
    try:
        start, end = (float(x) for x in text.split(","))
    except ValueError:
        raise ValueError(f"outage must be 'start,end' in seconds, got {text!r}") from None
    if end <= start:
        raise ValueError(f"outage must end after it starts, got {text!r}")
    return start, end

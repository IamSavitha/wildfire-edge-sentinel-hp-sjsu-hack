"""Cloud-only shadow ledger: for every frame the edge node really processes, what a cloud-only design
would have done with the same frame.

Cloud-only (the baseline of scripts/bench_cloud.py): every frame is JPEG-encoded whole at <= 1280 px
and sent to a hosted Qwen2.5-VL-7B, the same model the edge runs. The edge side is measured (bytes it
really uplinked, tokens it really spent, how long it really took); the cloud side is MODELED from the
real frame size (image tokens: `cloud_frame_tokens`; transfer: `netprofile`), with the edge's own
measured VLM time as the cloud model's compute time (same model; datacenter GPUs would be faster,
the network slower). Prices are user inputs. While the uplink is down a cloud-only system cannot
decide at all: those frames are counted as cloud-blind, and the edge's decisions as offline ones.
No I/O here."""
import math
import threading
from collections import defaultdict, deque
from statistics import median

from sentinel.costmodel import DAY_S, compare
from sentinel.live_metrics import effective_prices
from sentinel.netprofile import CLOUD_REQUEST_RTTS, PROFILES, get_profile, transfer_s
from sentinel.webapp_logic import FULL_FRAME_MAX_SIDE, cloud_frame_tokens

# Text tokens around the image in the cloud request: system prompt + schema hint + user turn of
# CLOUD_FULL_FRAME_PROMPT (~150 words of prompt and a ~40-field JSON schema hint). An estimate.
CLOUD_PROMPT_TOKENS = 220
CLOUD_OUTPUT_TOKENS = 80          # one compact context JSON answer
SOURCES = ("tower", "mobile", "field")
SAMPLES = 2000                    # latency samples kept per series

# Fleet projection defaults, used until the ledger has measured values (all shown as assumptions).
FLEET_DEFAULTS = {"tokens_per_full_frame": 1500, "bytes_per_frame": 250_000, "local_vlm_s": 5.6,
                  "detector_s": 0.04, "events_per_tower_day": 2, "vlm_calls_per_event": 3,
                  "alerts_per_tower_day": 0.2, "bytes_per_alert": 30_000}

METHOD = {
    "cloud_bytes": "Every frame sent whole as a JPEG at ≤1280 px (real frame bytes, scaled by area).",
    "cloud_tokens": f"Qwen2.5-VL image tokens of the ≤1280 px frame + {CLOUD_PROMPT_TOKENS} prompt tokens in, "
                    f"{CLOUD_OUTPUT_TOKENS} out, per frame.",
    "cloud_usd": "Cloud tokens and uploaded GB × the prices you set (assumptions, not measurements).",
    "cloud_time": "Edge VLM time (same model) + upload and one round trip over the selected link profile.",
    "edge_bytes": "What the edge really uplinked: ALERT reports only; video never leaves the device.",
    "edge_tokens": "Tokens the edge VLM really spent, only on frames that passed the detector and the gate.",
    "edge_usd": "No per-call API cost on the edge; it pays uplink for ALERT reports only.",
    "edge_time": "Measured on the Nano: detector + VLM, for frames where the VLM classified a plume.",
    "blind": "Frames seen while the uplink was down: a cloud-only system could not decide on them.",
}


def _p50(xs) -> float | None:
    return float(median(xs)) if xs else None


def _side() -> dict:
    return {"frames": 0, "bytes_up": 0, "vlm_calls": 0, "tokens_in": 0, "tokens_out": 0}


class CloudShadow:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.edge = _side()
            self.cloud = _side()
            self.edge_tokens = 0                   # edge VLM reports a total; no in/out split needed
            self.cloud_blind_frames = 0
            self.edge_offline_decisions = 0
            self.by_source: dict[str, dict] = defaultdict(lambda: {"frames": 0, "edge_bytes_up": 0,
                                                                   "cloud_bytes_up": 0, "edge_tokens": 0,
                                                                   "cloud_tokens": 0})
            self.edge_decide_ms = deque(maxlen=SAMPLES)
            self.vlm_ms = deque(maxlen=SAMPLES)
            self.detect_ms = deque(maxlen=SAMPLES)
            self.cloud_frame_bytes = deque(maxlen=SAMPLES)

    def record(self, source: str, width: int, height: int, frame_bytes: int, *, edge_vlm_called: bool,
               edge_tokens: int, edge_bytes_up: int, edge_decide_ms: float | None, online: bool,
               vlm_ms: float | None = None, detect_ms: float | None = None) -> None:
        """One frame the edge processed. `edge_decide_ms` is how long the edge took for this frame."""
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; known: {SOURCES}")
        scale = min(1.0, FULL_FRAME_MAX_SIDE / max(width, height, 1))
        c_bytes = int(round(frame_bytes * scale * scale))
        c_in = cloud_frame_tokens(width, height) + CLOUD_PROMPT_TOKENS
        with self.lock:
            e, c, s = self.edge, self.cloud, self.by_source[source]
            e["frames"] += 1
            e["bytes_up"] += int(edge_bytes_up or 0)
            e["vlm_calls"] += int(bool(edge_vlm_called))
            self.edge_tokens += int(edge_tokens or 0)
            s["frames"] += 1
            s["edge_bytes_up"] += int(edge_bytes_up or 0)
            s["edge_tokens"] += int(edge_tokens or 0)
            if edge_vlm_called and edge_decide_ms is not None:
                self.edge_decide_ms.append(float(edge_decide_ms))
            if vlm_ms is not None:
                self.vlm_ms.append(float(vlm_ms))
            if detect_ms is not None:
                self.detect_ms.append(float(detect_ms))
            if not online:
                self.cloud_blind_frames += 1
                self.edge_offline_decisions += 1
                return
            c["frames"] += 1
            c["bytes_up"] += c_bytes
            c["vlm_calls"] += 1
            c["tokens_in"] += c_in
            c["tokens_out"] += CLOUD_OUTPUT_TOKENS
            s["cloud_bytes_up"] += c_bytes
            s["cloud_tokens"] += c_in + CLOUD_OUTPUT_TOKENS
            self.cloud_frame_bytes.append(c_bytes)

    # ------------------------------------------------------------ views

    @staticmethod
    def profiles() -> list[dict]:
        return [{"name": p.name, "uplink_kbps": p.uplink_kbps, "rtt_ms": None if math.isinf(p.rtt_ms) else p.rtt_ms,
                 "loss": p.loss, "down": p.is_down} for p in PROFILES.values()]

    def summary(self, prices: dict | None = None, profile: str = "lte") -> dict:
        p = effective_prices(prices or {})
        link = get_profile(profile)
        with self.lock:
            e, c = dict(self.edge), dict(self.cloud)
            edge_tokens, blind, offline = self.edge_tokens, self.cloud_blind_frames, self.edge_offline_decisions
            by_source = {k: dict(v) for k, v in self.by_source.items()}
            edge_ms, vlm_ms = _p50(self.edge_decide_ms), _p50(self.vlm_ms)
            frame_bytes = _p50(self.cloud_frame_bytes)
        prices_set = p["usd_per_mtok_in"] > 0 or p["usd_per_mtok_out"] > 0 or p["usd_per_gb"] > 0
        edge_usd = e["bytes_up"] / 1e9 * p["usd_per_gb"]
        cloud_usd = (c["tokens_in"] / 1e6 * p["usd_per_mtok_in"] + c["tokens_out"] / 1e6 * p["usd_per_mtok_out"]
                     + c["bytes_up"] / 1e9 * p["usd_per_gb"])
        cloud_ms = None
        if vlm_ms is not None and frame_bytes is not None and not link.is_down:
            cloud_ms = vlm_ms + transfer_s(frame_bytes, link, CLOUD_REQUEST_RTTS) * 1000
        cloud_tokens = c["tokens_in"] + c["tokens_out"]
        return {
            "modeled": True, "profile": link.name, "link_down": link.is_down, "prices_set": prices_set,
            "edge": {**e, "tokens": edge_tokens, "usd": edge_usd if prices_set else None,
                     "decide_ms_p50": edge_ms},
            "cloud": {**c, "tokens": cloud_tokens, "usd": cloud_usd if prices_set else None,
                      "decide_ms_p50": cloud_ms, "frame_bytes_p50": frame_bytes},
            "cloud_blind_frames": blind, "edge_offline_decisions": offline,
            "savings": {
                "usd": (cloud_usd - edge_usd) if prices_set else None,
                "bytes": c["bytes_up"] - e["bytes_up"],
                "bytes_pct": 100 * (1 - e["bytes_up"] / c["bytes_up"]) if c["bytes_up"] else None,
                "bytes_x": c["bytes_up"] / e["bytes_up"] if e["bytes_up"] else None,
                "tokens_pct": 100 * (1 - edge_tokens / cloud_tokens) if cloud_tokens else None,
                "vlm_calls_avoided": max(0, c["vlm_calls"] - e["vlm_calls"]),
            },
            "by_source": by_source,
            "method": METHOD,
        }

    def fleet(self, towers: int, fps: float, days: int, prices: dict | None = None) -> dict:
        """towers × fps × days through sentinel.costmodel.compare, with this ledger's measured averages
        where it has them and FLEET_DEFAULTS otherwise; `measured` names what came from the ledger."""
        p = effective_prices(prices or {})
        with self.lock:
            c = dict(self.cloud)
            vlm_ms, det_ms = _p50(self.vlm_ms), _p50(self.detect_ms)
        d, measured = dict(FLEET_DEFAULTS), []
        if c["frames"]:
            d["tokens_per_full_frame"] = (c["tokens_in"]) / c["frames"]
            d["bytes_per_frame"] = c["bytes_up"] / c["frames"]
            measured += ["tokens_per_full_frame", "bytes_per_frame"]
        if vlm_ms is not None:
            d["local_vlm_s"] = vlm_ms / 1000
            measured.append("local_vlm_s")
        if det_ms is not None:
            d["detector_s"] = det_ms / 1000
            measured.append("detector_s")
        inputs = {"towers": towers, "fps": fps, "usd_per_mtok": p["usd_per_mtok_in"], "usd_per_gb": p["usd_per_gb"],
                  "tokens_per_full_frame": d["tokens_per_full_frame"], "bytes_per_frame": d["bytes_per_frame"],
                  "local_vlm_s": d["local_vlm_s"], "detector_s": d["detector_s"],
                  "events_per_day": towers * d["events_per_tower_day"],
                  "vlm_calls_per_event": d["vlm_calls_per_event"],
                  "alerts_per_day": towers * d["alerts_per_tower_day"], "bytes_per_alert": d["bytes_per_alert"]}
        rows = compare(inputs)
        for r in rows:          # compare() is per day; scale the additive columns to the period
            for k in ("vlm_calls", "cloud_tokens", "upstream_gb", "gpu_seconds", "usd_per_day"):
                r[k + "_total"] = r[k] * days
        return {"towers": towers, "fps": fps, "days": days, "rows": rows, "inputs": inputs,
                "measured": measured, "day_s": DAY_S}

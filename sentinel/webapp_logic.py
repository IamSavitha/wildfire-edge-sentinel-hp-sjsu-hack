"""Pure logic behind the demo web app (`sentinel.webapp`): one cascade pass over a single image,
the escalation decision for it, and session usage/economics. No network or GPU I/O here; the
detector, VLM, forecast and clocks are injected so everything is testable with fakes."""
import base64
import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from statistics import median
from typing import Callable

import cv2
import numpy as np

from sentinel.config import Tower
from sentinel.imaging import crop_box, to_jpeg
from sentinel.live_metrics import effective_prices
from sentinel.report import build_report, render_text
from sentinel.schema import Severity
from sentinel.severity import assess, fallback_severity

log = logging.getLogger(__name__)

# vLLM's default max_pixels for Qwen2.5-VL (16384 * 28 * 28): what a full frame costs if a cloud
# deployment of the same model received it unresized. An estimate, not a measurement.
QWEN_MAX_PIXELS = 12845056
QWEN_MIN_PIXELS = 3136
FULL_FRAME_MAX_SIDE = 1280   # what the edge VLM sees under force_vlm (Settings.full_frame_max_side)
THUMB_MAX_SIDE = 256
TRUTHS = ("unknown", "wildfire", "no_wildfire")
WILDFIRE_SEVERITIES = {"ALERT", "MONITOR"}
LATENCY_SAMPLES = 2000


# ---------------------------------------------------------------- image tokens

def qwen_image_tokens(w: int, h: int, max_pixels: int = QWEN_MAX_PIXELS,
                      min_pixels: int = QWEN_MIN_PIXELS, factor: int = 28) -> int:
    """Image tokens Qwen2.5-VL spends on a w x h image (its `smart_resize`: both sides rounded to
    multiples of `factor`, then scaled down to fit max_pixels or up to reach min_pixels; one token
    per factor x factor patch after the 2x2 merge)."""
    if w <= 0 or h <= 0:
        return 0
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return (h_bar // factor) * (w_bar // factor)


# ---------------------------------------------------------------- one pass

def _det_dict(d, min_conf: float) -> dict:
    return {"cls": d.cls, "conf": round(float(d.conf), 4),
            "box": [round(float(v), 1) for v in d.box], "above_gate": d.conf >= min_conf}


def _thumbnail(image: np.ndarray, dets, min_conf: float) -> str:
    """Whole frame with detector boxes drawn, scaled to <= 256 px, as base64 JPEG."""
    img = image.copy()
    h, w = img.shape[:2]
    thick = max(2, round(max(h, w) / 256))
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        color = (36, 72, 229) if d.conf >= min_conf else (160, 160, 160)   # BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    return base64.b64encode(to_jpeg(crop_box(img, (0, 0, w, h), 0.0, THUMB_MAX_SIDE), 75)).decode()


def run_pass(image_bgr: np.ndarray, detector, vlm, tower: Tower, *, min_conf: float = 0.4,
             crop_pad: float = 0.5, crop_max_side: int = 448, force_vlm: bool = False,
             now: float | None = None, timer: Callable[[], float] = time.perf_counter,
             event_id: str | None = None) -> dict:
    """Run the cascade on one image: detector -> (gate) -> context VLM -> severity -> report.

    No detection >= min_conf: the VLM is not called and severity is IGNORE, unless force_vlm, in
    which case the VLM sees the whole frame. `vlm=None` means the VLM is not served."""
    now = time.time() if now is None else now
    h, w = image_bgr.shape[:2]
    t0 = timer()
    dets = list(detector(image_bgr))
    detect_ms = (timer() - t0) * 1000
    strong = [d for d in dets if d.conf >= min_conf]
    best = max(strong, key=lambda d: d.conf, default=None)

    out = {
        "width": w, "height": h,
        "detections": [_det_dict(d, min_conf) for d in sorted(dets, key=lambda d: -d.conf)],
        "best": _det_dict(best, min_conf) if best else None,
        "min_conf": min_conf, "detect_ms": detect_ms,
        "vlm_called": False, "vlm_status": None, "vlm_input": None, "vlm_ms": None, "vlm_calls": 0,
        "tokens": 0, "tokens_in": None, "tokens_out": None,
        "crop_image_tokens": 0, "sent_size": None,
        "full_frame_image_tokens": qwen_image_tokens(w, h),
        "context": None, "severity": None, "report": None, "report_text": None,
        "thumbnail_b64": _thumbnail(image_bgr, dets, min_conf),
    }

    if best is None and not force_vlm:
        out["vlm_status"] = "skipped_no_detection"
        out["severity"] = Severity.IGNORE.name
        return out

    if best is not None:
        sent = crop_box(image_bgr, best.box, crop_pad, crop_max_side)
        out["vlm_input"] = "crop"
    else:
        sent = crop_box(image_bgr, (0, 0, w, h), 0.0, FULL_FRAME_MAX_SIDE)
        out["vlm_input"] = "full_frame"
    sh, sw = sent.shape[:2]
    out["sent_size"] = [sw, sh]

    ctx = None
    if vlm is None:
        out["vlm_status"] = "unavailable"
    else:
        t0 = timer()
        try:
            ctx, tokens = vlm.classify(to_jpeg(sent))
        except Exception:  # noqa: BLE001 - a VLM failure must never lose the detection
            log.exception("VLM classify raised")
            ctx, tokens = None, 0
        out["vlm_ms"] = (timer() - t0) * 1000
        out["vlm_called"] = True
        out["vlm_status"] = "ok" if ctx is not None else "failed"
        out["tokens"] = int(tokens)
        usage = getattr(vlm, "last_usage", None)
        calls = 1 if tokens else 0
        if isinstance(usage, dict) and "prompt_tokens" in usage:
            calls = int(usage.get("calls", 1))
            if calls > 0:   # a request that never completed has no split to report
                out["tokens_in"] = int(usage["prompt_tokens"])
                out["tokens_out"] = int(usage.get("completion_tokens") or 0)
        out["vlm_calls"] = calls
        out["crop_image_tokens"] = qwen_image_tokens(sw, sh)

    if ctx is not None:
        severity = assess(ctx, None)
    elif best is not None:
        severity = fallback_severity(None)    # never silently ignore a detection
    else:
        severity = Severity.IGNORE
    out["context"] = ctx.model_dump() if ctx is not None else None
    out["severity"] = severity.name
    report = build_report(event_id or uuid.uuid4().hex[:12], tower, severity, ctx, None,
                          best.conf if best else 0.0, now, out["thumbnail_b64"])
    out["report"] = report
    out["report_text"] = render_text(report)
    return out


# ---------------------------------------------------------------- ground truth & escalation

def ground_truth_score(severity, truth: str) -> bool | None:
    """Was the pass right? A pass predicts "wildfire" iff severity is ALERT or MONITOR.
    None when the truth is unknown (not scored)."""
    if truth not in TRUTHS:
        raise ValueError(f"truth must be one of {TRUTHS}")
    if truth == "unknown":
        return None
    name = severity.name if isinstance(severity, Severity) else str(severity)
    return (name in WILDFIRE_SEVERITIES) == (truth == "wildfire")


def escalate(result: dict, online: bool, *, forecast_fn: Callable[[float, float], dict]) -> dict:
    """The edge->cloud rule for one pass: only ALERT leaves the device. Online, the report (with
    one thumbnail) is sent after fetching a forecast; offline it waits in the outbox."""
    sev = result.get("severity")
    report = result.get("report")
    if sev != "ALERT" or report is None:
        if sev in ("LOG", "MONITOR"):
            return {"decision": "logged", "bytes_up": 0, "payload_bytes": 0, "forecast": None,
                    "report_text": result.get("report_text"),
                    "note": "Kept on the device (local log); nothing sent upstream."}
        return {"decision": "ignored", "bytes_up": 0, "payload_bytes": 0, "forecast": None,
                "report_text": result.get("report_text"),
                "note": "Counted locally only; nothing sent upstream."}
    payload = dict(report)
    if not online:
        payload["forecast_status"] = "pending"
        return {"decision": "queued", "bytes_up": 0,
                "payload_bytes": len(json.dumps(payload).encode()), "forecast": None,
                "report_text": render_text(payload),
                "note": "Link down: alert held in the outbox, sent when the link returns."}
    try:
        payload["forecast"] = forecast_fn(payload["lat"], payload["lon"])
        payload["forecast_status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - the alert goes out even without a forecast
        log.warning("forecast failed: %s", exc)
        payload["forecast"] = None
        payload["forecast_status"] = "pending"
    size = len(json.dumps(payload).encode())
    return {"decision": "sent", "bytes_up": size, "payload_bytes": size,
            "forecast": payload["forecast"], "report_text": render_text(payload),
            "note": "Alert report + one thumbnail sent to dispatch (simulated)"
                    + ("" if payload["forecast"] else "; forecast pending") + "."}


# ---------------------------------------------------------------- session usage & economics

def _p50(xs) -> float | None:
    return float(median(xs)) if xs else None


class _Stats:
    def __init__(self):
        self.images = self.detector_runs = self.vlm_calls = self.vlm_requests = self.calls_avoided = 0
        self.tokens_actual = 0
        self.tokens_in = self.tokens_out = 0
        self.split_calls = 0              # VLM passes that reported an in/out split
        self.split_crop_tokens = 0        # image tokens inside those passes' prompt tokens
        self.split_requests = 0
        self.full_frame_image_tokens = 0
        self.bytes_up = self.image_upload_bytes = 0
        self.gt_correct = self.gt_total = 0
        self.alerts_sent = self.alerts_queued = 0
        self.online_images = self.offline_images = 0
        self.latency = deque(maxlen=LATENCY_SAMPLES)
        self.detect = deque(maxlen=LATENCY_SAMPLES)
        self.vlm = deque(maxlen=LATENCY_SAMPLES)

    def summary(self, overhead: float, out_per: float, observed: bool) -> dict:
        ff_in = self.full_frame_image_tokens + round(self.images * overhead)
        ff_out = round(self.images * out_per)
        return {
            "images": self.images, "detector_runs": self.detector_runs, "vlm_calls": self.vlm_calls,
            "calls_avoided": self.calls_avoided, "tokens_actual": self.tokens_actual,
            "tokens_in": self.tokens_in if self.split_calls else None,
            "tokens_out": self.tokens_out if self.split_calls else None,
            "full_frame_image_tokens": self.full_frame_image_tokens,
            "full_frame_tokens_in_est": ff_in, "full_frame_tokens_out_est": ff_out,
            "full_frame_tokens_est": ff_in + ff_out,
            "overhead_observed": observed,
            "bytes_up": self.bytes_up, "image_upload_bytes": self.image_upload_bytes,
            "gt_correct": self.gt_correct, "gt_total": self.gt_total,
            "accuracy": self.gt_correct / self.gt_total if self.gt_total else None,
            "alerts_sent": self.alerts_sent, "alerts_queued": self.alerts_queued,
            "online_images": self.online_images, "offline_images": self.offline_images,
            "latency_ms_p50": _p50(self.latency), "detect_ms_p50": _p50(self.detect),
            "vlm_ms_p50": _p50(self.vlm),
        }


class Session:
    """Usage accumulated since the page (or the last reset), per pipeline name."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.pipelines: dict[str, _Stats] = {}
            self.requests = self.online_requests = self.offline_requests = 0

    def record_request(self, online: bool) -> None:
        with self.lock:
            self.requests += 1
            if online:
                self.online_requests += 1
            else:
                self.offline_requests += 1

    def record(self, pipeline: str, result: dict, escalation: dict, *, truth: str,
               image_bytes: int, online: bool) -> bool | None:
        correct = ground_truth_score(result["severity"], truth)
        with self.lock:
            s = self.pipelines.setdefault(pipeline, _Stats())
            s.images += 1
            s.detector_runs += 1
            if result.get("vlm_status") == "skipped_no_detection":
                s.calls_avoided += 1
            if result.get("vlm_called"):
                s.vlm_calls += 1
                requests = max(1, int(result.get("vlm_calls") or 1))
                if result.get("tokens_in") is not None:
                    s.split_calls += 1
                    s.split_requests += requests
                    s.tokens_in += int(result["tokens_in"])
                    s.tokens_out += int(result.get("tokens_out") or 0)
                    s.split_crop_tokens += int(result.get("crop_image_tokens") or 0) * requests
            s.tokens_actual += int(result.get("tokens") or 0)
            s.full_frame_image_tokens += int(result.get("full_frame_image_tokens") or 0)
            s.bytes_up += int(escalation.get("bytes_up") or 0)
            s.image_upload_bytes += int(image_bytes or 0)
            if correct is not None:
                s.gt_total += 1
                s.gt_correct += int(correct)
            decision = escalation.get("decision")
            s.alerts_sent += decision == "sent"
            s.alerts_queued += decision == "queued"
            if online:
                s.online_images += 1
            else:
                s.offline_images += 1
            det_ms = float(result.get("detect_ms") or 0.0)
            vlm_ms = result.get("vlm_ms")
            s.detect.append(det_ms)
            if vlm_ms is not None:
                s.vlm.append(float(vlm_ms))
            s.latency.append(det_ms + float(vlm_ms or 0.0))
        return correct

    def summary(self) -> dict:
        with self.lock:
            # prompt-text overhead and output length per VLM request, pooled over every pipeline
            # (the cloud baseline is the same images whichever cascade ran them)
            stats = list(self.pipelines.values())
            reqs = sum(s.split_requests for s in stats)
            if reqs:
                overhead = max(0.0, sum(s.tokens_in - s.split_crop_tokens for s in stats) / reqs)
                out_per = sum(s.tokens_out for s in stats) / reqs
            else:
                overhead = out_per = 0.0
            pipes = {name: s.summary(overhead, out_per, reqs > 0) for name, s in self.pipelines.items()}
            return {"pipelines": pipes, "network": {
                "requests": self.requests, "online_requests": self.online_requests,
                "offline_requests": self.offline_requests,
                # a cloud-only system has no model to run while the link is down
                "cloud_only_blind": self.offline_requests,
                "alerts_sent": sum(p["alerts_sent"] for p in pipes.values()),
                "alerts_queued": sum(p["alerts_queued"] for p in pipes.values()),
            }}

    def economics(self, prices: dict) -> dict:
        """Edge cascade vs "cloud: every image full-frame to a VLM", per pipeline. Edge inference
        is $0 marginal; the edge pays only uplink for what it sent. Prices are user-supplied."""
        p = effective_prices(prices or {})
        out = {}
        for name, s in self.summary()["pipelines"].items():
            edge_usd = s["bytes_up"] / 1e9 * p["usd_per_gb"]
            cloud_usd = (s["full_frame_tokens_in_est"] / 1e6 * p["usd_per_mtok_in"]
                         + s["full_frame_tokens_out_est"] / 1e6 * p["usd_per_mtok_out"]
                         + s["image_upload_bytes"] / 1e9 * p["usd_per_gb"])
            cloud_tokens = s["full_frame_tokens_est"]
            out[name] = {
                "edge_usd": edge_usd, "cloud_usd": cloud_usd, "savings_usd": cloud_usd - edge_usd,
                "savings_pct": 100 * (cloud_usd - edge_usd) / cloud_usd if cloud_usd > 0 else None,
                "edge_tokens": s["tokens_actual"], "cloud_tokens_est": cloud_tokens,
                "token_savings_pct": (100 * (1 - s["tokens_actual"] / cloud_tokens)
                                      if cloud_tokens > 0 else None),
                "edge_bytes_up": s["bytes_up"], "cloud_bytes_up": s["image_upload_bytes"],
                "bytes_savings_pct": (100 * (1 - s["bytes_up"] / s["image_upload_bytes"])
                                      if s["image_upload_bytes"] > 0 else None),
            }
        priced = p["usd_per_mtok_in"] > 0 or p["usd_per_mtok_out"] > 0 or p["usd_per_gb"] > 0
        return {"pipelines": out, "priced": priced, "prices": p}


# ---------------------------------------------------------------- benchmarks

BENCHMARKS = {
    "detector": {"before": "detector_before_yoloworld.json", "after": "detector_after_yolo11s.json"},
    "context": {"before": "context_before_base7b.json", "after": "context_after_lora7b.json"},
}
BENCH_KEYS = {
    "detector": ("map50", "precision", "recall", "ms_per_image", "n_images"),
    "context": ("source_type_acc", "group_acc", "latency_ms_p50", "tokens_per_call", "n"),
}


def load_benchmarks(results_dir) -> dict:
    """Before/after benchmark numbers from results/*.json; None for files that are absent/bad."""
    out: dict = {}
    for stage, files in BENCHMARKS.items():
        out[stage] = {}
        for phase, fname in files.items():
            try:
                data = json.loads((Path(results_dir) / fname).read_text())
                row = {k: data[k] for k in BENCH_KEYS[stage] if k in data} if isinstance(data, dict) else None
            except (OSError, ValueError):
                row = None
            out[stage][phase] = row or None
    return out

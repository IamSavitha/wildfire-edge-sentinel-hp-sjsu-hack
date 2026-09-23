"""Cloud-only end-to-end baseline on the same labeled clips as scripts/bench.py.
Writes results/bench_cloud_<name>.json.

Architecture being measured: no edge model at all. EVERY frame (or every k-th with --frame-stride)
is JPEG-encoded whole at <= 1280 px (the same cap as the edge full-frame path) and sent to the
hosted Qwen2.5-VL-7B-Instruct with the same prompt as the edge. Severity is assess(ctx, trend=None):
cloud-only has no local detector, persistence gate or area trend, so each frame is judged on its
own. A frame whose request fails (no answer) makes no decision. A clip is predicted "alert" if any
frame yields ALERT (--alert-on monitor also counts MONITOR).

Measured vs modelled:
  MEASURED  cloud answers, billed tokens (provider usage), and the cloud inference latency: the real
            wall time of each HTTPS request from the machine running this script (for a cache hit,
            the latency measured when the call was originally made; reported separately). It already
            includes one upload over this machine's own (fast) link, so adding the modelled tower
            uplink on top slightly overstates cloud delay on fiber (by tens of ms), not on slow links.
  MODELLED  the tower's uplink (sentinel/netprofile.py): per frame, decision time =
            capture time + queueing on a serial uplink + transfer_s(frame bytes) + measured latency.
            Uploads share one uplink (a frame waits while the previous one is still uploading);
            inference itself does not block the next upload. Outage windows (--outage) either drop
            the frames captured while the link is down, or buffer them and upload them serially
            once it returns (--outage-policy buffer), so the backlog delay shows up. Windows are
            in clip time; for outages longer than a clip use scripts/outage_scenario.py, which also
            queues the frames the camera captures outside the recorded clip.

Cost: every uncached frame is a paid API call (25 clips x 40 frames = 1000 calls). Use --limit and
--frame-stride for a dry run and --max-fresh-calls as a hard cap; answers are cached in
data/cloud_cache.jsonl so re-running (e.g. with other --profiles or --outage) is not re-billed.

    python scripts/bench_cloud.py --limit 2 --frame-stride 10       # dry run: ~8 calls
    python scripts/bench_cloud.py --profiles fiber,lte,rural_cellular,satellite_geo
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from sentinel.cloud_cache import CachedVLM, ResponseCache
from sentinel.config import load_settings
from sentinel.imaging import crop_box, to_jpeg
from sentinel.live_metrics import effective_prices
from sentinel.netprofile import LinkProfile, Outages, parse_outage, parse_profiles, transfer_s
from sentinel.replayer import frames
from sentinel.severity import assess
from sentinel.vlm_client import RESPONSE_FORMATS, cloud_vlm_from_env

try:
    from scripts.bench import read_clips, score
    from scripts.eval_context import check_output
except ModuleNotFoundError as e:  # run as `python scripts/bench_cloud.py`
    if e.name != "scripts":
        raise
    from bench import read_clips, score
    from eval_context import check_output

DEFAULT_PROFILES = "fiber,lte,rural_cellular,satellite_geo"
POLICIES = ("drop", "buffer")
ALERT_ON = {"alert": {"ALERT"}, "monitor": {"ALERT", "MONITOR"}}


def _pct(xs, q) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def encode_frame(frame: np.ndarray, max_side: int) -> bytes:
    """The whole frame at <= max_side px on its longest side, as the edge full-frame path sends it."""
    h, w = frame.shape[:2]
    return to_jpeg(crop_box(frame, (0, 0, w, h), 0.0, max_side))


def classify_clip(path: str, vlm, fps: float, stride: int = 1, max_side: int = 1280) -> list[dict]:
    """Send every stride-th frame of a clip to the cloud VLM. `vlm` is a CachedVLM."""
    out = []
    for i, frame in enumerate(frames(path, fps, loop=False)):
        if i % stride:
            continue
        jpeg = encode_frame(frame, max_side)
        ctx, _ = vlm.classify(jpeg)
        usage = vlm.last_usage or {}
        answered = vlm.last_error is None
        out.append({
            "i": i, "t": i / fps, "bytes": len(jpeg),
            "severity": assess(ctx, None).name if ctx is not None else None,
            "source_type": ctx.source_type if ctx is not None else None,
            "latency_ms": vlm.last_latency_ms if answered else None,
            "cached": bool(vlm.last_cached), "error": vlm.last_error,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        })
    return out


def simulate(frame_rows: list[dict], profile: LinkProfile, outages: Outages | None = None,
             policy: str = "drop", alert_on: str = "alert") -> dict:
    """Replay one clip's measured cloud answers over a modelled link. Times are sim seconds from
    clip start. Returns when each decision lands and what it cost to get there."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    outages = outages or Outages()
    positive = ALERT_ON[alert_on]
    res = {"frames": len(frame_rows), "decided": 0, "dropped": 0, "failed": 0, "bytes_up": 0,
           "prompt_tokens": 0, "completion_tokens": 0, "first_alert_s": None,
           "decisions_during_outage": 0, "captured_during_outage": 0,
           "delays_s": [], "max_backlog_s": 0.0}
    link_free = 0.0
    for f in sorted(frame_rows, key=lambda r: r["t"]):
        down_at_capture = outages.is_down(f["t"])
        res["captured_during_outage"] += down_at_capture
        if profile.is_down or (down_at_capture and policy == "drop"):
            res["dropped"] += 1
            continue
        start = max(f["t"], link_free)
        while outages.is_down(start):
            start = outages.next_up(start)
        link_free = start + transfer_s(f["bytes"], profile)
        res["bytes_up"] += f["bytes"]
        res["max_backlog_s"] = max(res["max_backlog_s"], start - f["t"])
        if f.get("latency_ms") is None:  # the request failed: uploaded, but no answer
            res["failed"] += 1
            continue
        res["prompt_tokens"] += f.get("prompt_tokens", 0)
        res["completion_tokens"] += f.get("completion_tokens", 0)
        decided_at = link_free + f["latency_ms"] / 1000
        res["decided"] += 1
        res["delays_s"].append(decided_at - f["t"])
        res["decisions_during_outage"] += outages.is_down(decided_at)
        if f.get("severity") in positive and (res["first_alert_s"] is None or decided_at < res["first_alert_s"]):
            res["first_alert_s"] = decided_at
    return res


def usd(prompt_tokens: float, completion_tokens: float, nbytes: float, prices: dict) -> float:
    p = effective_prices(prices or {})
    return (prompt_tokens / 1e6 * p["usd_per_mtok_in"] + completion_tokens / 1e6 * p["usd_per_mtok_out"]
            + nbytes / 1e9 * p["usd_per_gb"])


def profile_summary(rows: list[dict], sims: list[dict], prices: dict) -> dict:
    predictions = [s["first_alert_s"] is not None for s in sims]
    tta = [s["first_alert_s"] for r, s in zip(rows, sims) if r["label"] == "alert" and s["first_alert_s"] is not None]
    delays = [d for s in sims for d in s["delays_s"]]
    tot = {k: sum(s[k] for s in sims) for k in ("frames", "decided", "dropped", "failed", "bytes_up",
                                                 "prompt_tokens", "completion_tokens",
                                                 "decisions_during_outage", "captured_during_outage")}
    cost = usd(tot["prompt_tokens"], tot["completion_tokens"], tot["bytes_up"], prices)
    sent = tot["decided"] + tot["failed"]
    return {
        **score(rows, predictions),
        "time_to_first_alert_s_p50": _pct(tta, 50), "time_to_first_alert_s_p95": _pct(tta, 95),
        "decision_delay_s_p50": _pct(delays, 50), "decision_delay_s_p95": _pct(delays, 95),
        "max_backlog_s": max((s["max_backlog_s"] for s in sims), default=0.0),
        **{f"frames_{k}" if k in ("decided", "dropped", "failed") else k: v for k, v in tot.items()},
        "usd": cost,
        "usd_per_1000_frames": cost / sent * 1000 if sent else None,
        "bytes_per_1000_frames": tot["bytes_up"] / sent * 1000 if sent else None,
        "tokens_per_decision": ((tot["prompt_tokens"] + tot["completion_tokens"]) / tot["decided"]
                                if tot["decided"] else None),
    }


def bench_cloud(name: str, rows: list[dict], vlm, *, fps: float, profiles: list[LinkProfile],
                outages: Outages | None = None, policy: str = "drop", alert_on: str = "alert",
                stride: int = 1, max_side: int = 1280, prices: dict | None = None,
                log=lambda msg: None) -> dict:
    outages = outages or Outages()
    clips = []
    for n, row in enumerate(rows, 1):
        fr = classify_clip(row["path"], vlm, fps, stride, max_side)
        log(f"[{n}/{len(rows)}] {row['path']}: {len(fr)} frames "
            f"({vlm.n_fresh} fresh / {vlm.n_cached} cached calls so far)")
        clips.append({"path": row["path"], "label": row["label"], "frames": fr})

    frames_all = [f for c in clips for f in c["frames"]]
    positive = ALERT_ON[alert_on]
    ideal = [any(f["severity"] in positive for f in c["frames"]) for c in clips]
    fresh = [f["latency_ms"] for f in frames_all if not f["cached"] and f["latency_ms"] is not None]
    cached = [f["latency_ms"] for f in frames_all if f["cached"] and f["latency_ms"] is not None]
    billed = [f for f in frames_all if not f["cached"]]
    answered = [f for f in frames_all if f["latency_ms"] is not None]

    per_profile = {}
    for prof in profiles:
        sims = [simulate(c["frames"], prof, outages, policy, alert_on) for c in clips]
        per_profile[prof.name] = profile_summary(rows, sims, prices or {})
        for c, s in zip(clips, sims):
            c.setdefault("per_profile", {})[prof.name] = {
                k: s[k] for k in ("first_alert_s", "decided", "dropped", "failed", "bytes_up",
                                  "decisions_during_outage", "max_backlog_s")}
    for c, pred in zip(clips, ideal):
        c["predicted_alert"] = pred
        c["severities"] = {s: sum(f["severity"] == s for f in c["frames"])
                           for s in ("ALERT", "MONITOR", "LOG", "IGNORE", None)}
        c["severities"]["failed"] = c["severities"].pop(None)

    return {
        "name": name, "architecture": "cloud_only", "n_clips": len(rows),
        "model": vlm.model, "response_format": vlm.mode,
        "fps": fps, "frame_stride": stride, "max_side": max_side, "alert_on": alert_on,
        "outage_policy": policy, "outages": outages.windows,
        # no network model: every answered frame decides (accuracy of the cloud-only design itself)
        **score(rows, ideal),
        "frames_sent": len(frames_all), "frames_answered": len(answered),
        "n_fresh": vlm.n_fresh, "n_cached": vlm.n_cached, "n_request_errors": vlm.n_errors,
        "parse_fail_rate": (sum(f["severity"] is None and f["latency_ms"] is not None for f in frames_all)
                            / len(frames_all) if frames_all else None),
        "cloud_latency_ms_p50": _pct(fresh, 50), "cloud_latency_ms_p95": _pct(fresh, 95),
        "cached_latency_ms_p50": _pct(cached, 50), "cached_latency_ms_p95": _pct(cached, 95),
        "billed_prompt_tokens": sum(f["prompt_tokens"] for f in billed),
        "billed_completion_tokens": sum(f["completion_tokens"] for f in billed),
        "tokens_per_frame": (sum(f["prompt_tokens"] + f["completion_tokens"] for f in answered) / len(answered)
                             if answered else None),
        "frame_bytes_p50": _pct([f["bytes"] for f in frames_all], 50),
        "prices": effective_prices(prices or {}),
        "profiles": per_profile,
        "clips": clips,
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="cloud_qwen7b", help="result tag -> results/bench_cloud_<name>.json")
    ap.add_argument("--clips", default="data/bench/clips.csv")
    ap.add_argument("--settings", default="config/settings.json", help="fps and full_frame_max_side")
    ap.add_argument("--profiles", default=DEFAULT_PROFILES, help="comma-separated link profiles")
    ap.add_argument("--frame-stride", type=int, default=1, help="send every k-th frame (dry runs, cost)")
    ap.add_argument("--limit", type=int, help="first N clips only (dry runs)")
    ap.add_argument("--alert-on", choices=sorted(ALERT_ON), default="alert",
                    help="clip is an alert if any frame is ALERT (or ALERT/MONITOR with 'monitor')")
    ap.add_argument("--outage", action="append", default=[], metavar="START,END",
                    help="link down from START to END sim seconds after clip start (repeatable)")
    ap.add_argument("--outage-policy", choices=POLICIES, default="drop",
                    help="drop: frames captured in an outage are never decided; buffer: sent after it")
    ap.add_argument("--response-format", choices=RESPONSE_FORMATS,
                    help="overrides CLOUD_VLM_RESPONSE_FORMAT (default json_schema)")
    ap.add_argument("--cost-inputs", default="config/cost_inputs.json",
                    help="usd_per_mtok_in/out, usd_per_gb (zeros allowed)")
    ap.add_argument("--cache", default="data/cloud_cache.jsonl")
    ap.add_argument("--max-fresh-calls", type=int, help="hard cap on billed calls in this run")
    ap.add_argument("--timeout", type=float, default=60, help="per-request timeout, seconds")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args(argv)
    if a.frame_stride < 1:
        ap.error("--frame-stride must be >= 1")
    try:
        a.profile_list = parse_profiles(a.profiles)
        a.outage_windows = Outages([parse_outage(o) for o in a.outage])
    except ValueError as e:
        ap.error(str(e))
    if not a.profile_list:
        ap.error("--profiles is empty")
    return a


def main(argv=None) -> None:
    a = parse_args(argv)
    out = Path("results") / f"bench_cloud_{a.name}.json"
    check_output(out, a.force)
    s = load_settings(a.settings)
    rows = read_clips(a.clips)[:a.limit] if a.limit else read_clips(a.clips)
    if not rows:
        raise SystemExit(f"no clips in {a.clips}")
    prices = json.loads(Path(a.cost_inputs).read_text()) if Path(a.cost_inputs).exists() else {}

    vlm = cloud_vlm_from_env(timeout_s=a.timeout, response_format=a.response_format)
    cached = CachedVLM(vlm, ResponseCache(a.cache), max_fresh_calls=a.max_fresh_calls)
    print(f"cloud-only: {len(rows)} clips, stride {a.frame_stride}, model {vlm.model} at {vlm.host} "
          f"({vlm.response_format}); {len(cached.cache)} cached answers on file", file=sys.stderr)
    result = bench_cloud(a.name, rows, cached, fps=s.fps, profiles=a.profile_list, outages=a.outage_windows,
                         policy=a.outage_policy, alert_on=a.alert_on, stride=a.frame_stride,
                         max_side=s.full_frame_max_side, prices=prices,
                         log=lambda m: print(m, file=sys.stderr))
    result.update(provider_host=vlm.host, clips_file=a.clips, limit=a.limit)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "name", "precision", "recall", "false_alarms", "missed", "frames_sent", "n_fresh", "n_cached",
        "cloud_latency_ms_p50", "cloud_latency_ms_p95", "billed_prompt_tokens", "billed_completion_tokens")},
        indent=2))
    def sec(x):
        return "—" if x is None else f"{x:.1f} s"
    for pname, p in result["profiles"].items():
        print(f"{pname:>15}: recall {p['recall']:.2f}, false alarms {p['false_alarms']}, first alert p50 "
              f"{sec(p['time_to_first_alert_s_p50'])}, decision delay p95 {sec(p['decision_delay_s_p95'])}, "
              f"max backlog {sec(p['max_backlog_s'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

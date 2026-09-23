"""Cloud-only end-to-end baseline on the same labeled clips as scripts/bench.py.
Writes results/bench_cloud_<name>.json.

Architecture being measured: no edge model at all. EVERY frame (or every k-th with --frame-stride)
is JPEG-encoded whole at <= 1280 px (the same cap as the edge full-frame path) and sent to the
hosted Qwen2.5-VL-7B-Instruct. The prompt is the full-frame variant of the edge prompt
(CLOUD_FULL_FRAME_PROMPT: most frames have no smoke; "no smoke" is answered as smoke_color "none",
which maps to IGNORE). A frame whose request fails makes no decision.

Two decision rules are reported, so the comparison with the edge's gate + trend re-check is fair:
  per-frame      severity = assess(ctx, trend=None); a clip alerts if any frame is ALERT
                 (--alert-on monitor also counts MONITOR).
  temporal       a cloud trend from the size_estimate rank vs the answered frame --recheck-s earlier
                 is passed into assess(), and ALERT needs --persist K consecutive positive answered frames.
A cheaper realistic cadence (every --cadence-stride-th frame, default 20 = one frame per 10 s at
2 fps) is derived from the same answers: no extra calls. Its temporal rule keeps the same persistence
in seconds: persist = max(1, ceil(K * frame_stride / cadence_stride)), so it can still alert in a
~20 s clip. The consecutive run continues across failed requests (only answered frames count, and
a request that got no answer does not reset it), which is lenient to the cloud.

Measured vs modelled:
  MEASURED  cloud answers, billed tokens (provider usage), cloud latency (real wall time of each HTTPS
            request from this machine; for a cache hit, the latency measured when the call was made),
            and net_baseline_ms: the fastest of 5 cheap authenticated requests (GET /models), an
            approximation of the network + HTTPS overhead already inside every measured latency.
  MODELLED  the tower's uplink (sentinel/netprofile.py). Per sent frame:
              upload_end = start + serialize_s(bytes)     (the uplink is busy only while serializing)
              decided_at = upload_end + rtt_s(1) + max(0, latency - net_baseline_ms)
            so the eval machine's own network is not counted twice. Policies (--policy):
              latest         (default) when the uplink frees, send the newest captured frame; older
                             unsent frames are stale and skipped (a live system would do this)
              fifo           every frame, oldest first; a slow link builds a backlog; outage frames
                             wait and drain after it
              drop           fifo, but frames captured while the link is down are discarded
              buffer_newest  every frame, newest first
            Outage windows (--outage) are in clip time; for outages longer than a clip use
            scripts/outage_scenario.py, which also models frames captured outside the recorded clip.

Cost: every uncached frame is a paid API call (25 clips x 40 frames = 1000 calls). Use --limit and
--frame-stride for a dry run and --max-fresh-calls as a hard cap; answers are cached in
data/cloud_cache.jsonl so re-running (e.g. with other --profiles, --policy or --outage) is not re-billed.

    python scripts/bench_cloud.py --limit 2 --frame-stride 10       # dry run: ~8 calls
    python scripts/bench_cloud.py --profiles fiber,lte,rural_cellular,satellite_geo
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from sentinel.cloud_cache import CachedVLM, ResponseCache
from sentinel.config import load_settings
from sentinel.imaging import crop_box, to_jpeg
from sentinel.live_metrics import effective_prices
from sentinel.netprofile import (CLOUD_REQUEST_RTTS, LinkProfile, Outages, parse_outage, parse_profiles,
                                 rtt_s, serialize_s)
from sentinel.replayer import frames
from sentinel.schema import ContextResult
from sentinel.severity import assess
from sentinel.vlm_client import CLOUD_FULL_FRAME_PROMPT, RESPONSE_FORMATS, cloud_vlm_from_env, measure_net_baseline_ms

try:
    from scripts.bench import read_clips, score
    from scripts.eval_context import check_output
except ModuleNotFoundError as e:  # run as `python scripts/bench_cloud.py`
    if e.name != "scripts":
        raise
    from bench import read_clips, score
    from eval_context import check_output

DEFAULT_PROFILES = "fiber,lte,rural_cellular,satellite_geo"
POLICIES = ("latest", "fifo", "drop", "buffer_newest")
ALERT_ON = {"alert": {"ALERT"}, "monitor": {"ALERT", "MONITOR"}}
SIZE_RANK = {"small": 0, "medium": 1, "large": 2}


def _pct(xs, q) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def encode_frame(frame: np.ndarray, max_side: int) -> bytes:
    """The whole frame at <= max_side px on its longest side, as the edge full-frame path sends it."""
    h, w = frame.shape[:2]
    return to_jpeg(crop_box(frame, (0, 0, w, h), 0.0, max_side))


def frame_severity(ctx: dict | None, trend: str | None = None) -> str | None:
    """Severity of one full-frame answer; "no smoke" (smoke_color none) is IGNORE."""
    if ctx is None:
        return None
    if ctx.get("smoke_color") == "none":
        return "IGNORE"
    return assess(ContextResult.model_validate(ctx), trend).name


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
        c = ctx.model_dump() if ctx is not None else None
        out.append({
            "i": i, "t": i / fps, "bytes": len(jpeg), "ctx": c, "severity": frame_severity(c),
            "latency_ms": vlm.last_latency_ms if answered else None,
            "cached": bool(vlm.last_cached), "error": vlm.last_error,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
        })
    return out


def _trend(prev: dict | None, cur: dict | None) -> str | None:
    if not prev or not cur or prev.get("smoke_color") == "none" or cur.get("smoke_color") == "none":
        return None
    d = SIZE_RANK[cur["size_estimate"]] - SIZE_RANK[prev["size_estimate"]]
    return "growing" if d > 0 else "dissipating" if d < 0 else "static"


def first_alert(decided: list[dict], alert_on: str = "alert", temporal: dict | None = None) -> float | None:
    """When the first alert lands. decided: answered frames with t, decided_at, ctx, severity.
    temporal = {"persist": K, "recheck_s": R}: trend from the answered frame R s earlier, and ALERT
    only after K consecutive positive frames (the alert lands when the K-th answer arrives).
    Only answered frames are passed in, so a failed request does not break the run (lenient to the
    cloud); an answered but unparseable or non-positive frame does."""
    positive = ALERT_ON[alert_on]
    if temporal is None:
        times = [d["decided_at"] for d in decided if d.get("severity") in positive]
        return min(times) if times else None
    k, recheck = max(1, int(temporal["persist"])), float(temporal["recheck_s"])
    seq = sorted(decided, key=lambda d: d["t"])
    run: list[float] = []
    best, j = None, -1  # seq[j]: the latest answered frame at least `recheck` s before the current one
    for d in seq:
        while j + 1 < len(seq) and seq[j + 1]["t"] <= d["t"] - recheck:
            j += 1
        sev = frame_severity(d.get("ctx"), _trend(seq[j].get("ctx") if j >= 0 else None, d.get("ctx")))
        if sev in positive:
            run.append(d["decided_at"])
            if len(run) >= k:
                at = max(run[-k:])
                best = at if best is None else min(best, at)
        else:
            run = []
    return best


def simulate(frame_rows: list[dict], profile: LinkProfile, outages: Outages | None = None,
             policy: str = "latest", alert_on: str = "alert", net_baseline_ms: float | None = None,
             temporal: dict | None = None) -> dict:
    """Replay one clip's measured cloud answers over a modelled uplink. Times are sim seconds from
    clip start. See the module docstring for the policies."""
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    outages = outages or Outages()
    base_ms = net_baseline_ms or 0.0
    res = {"frames": len(frame_rows), "sent": 0, "decided": 0, "dropped": 0, "skipped": 0, "failed": 0,
           "bytes_up": 0, "prompt_tokens": 0, "completion_tokens": 0, "first_alert_s": None,
           "decisions_during_outage": 0, "captured_during_outage": 0, "delays_s": [], "max_backlog_s": 0.0}
    rows = sorted(frame_rows, key=lambda r: r["t"])
    res["captured_during_outage"] = sum(outages.is_down(f["t"]) for f in rows)
    if profile.is_down:
        res["dropped"] = len(rows)
        return res
    decided, pending, idx, link_free = [], [], 0, 0.0
    while idx < len(rows) or pending:
        now = link_free if pending else max(link_free, rows[idx]["t"])
        while idx < len(rows) and rows[idx]["t"] <= now:
            f = rows[idx]
            idx += 1
            if policy == "drop" and outages.is_down(f["t"]):
                res["dropped"] += 1
            else:
                pending.append(f)
        if not pending:
            link_free = now
            continue
        if outages.is_down(now):
            link_free = outages.next_up(now)
            continue
        if policy == "latest":
            f, res["skipped"] = pending[-1], res["skipped"] + len(pending) - 1
            pending = []
        elif policy == "buffer_newest":
            f = pending.pop()
        else:
            f = pending.pop(0)
        link_free = now + serialize_s(f["bytes"], profile)
        res["sent"] += 1
        res["bytes_up"] += f["bytes"]
        res["max_backlog_s"] = max(res["max_backlog_s"], now - f["t"])
        if f.get("latency_ms") is None:  # uploaded, but the request failed: no answer
            res["failed"] += 1
            continue
        res["prompt_tokens"] += f.get("prompt_tokens", 0)
        res["completion_tokens"] += f.get("completion_tokens", 0)
        at = link_free + rtt_s(profile, CLOUD_REQUEST_RTTS) + max(0.0, f["latency_ms"] - base_ms) / 1000
        res["decided"] += 1
        res["delays_s"].append(at - f["t"])
        res["decisions_during_outage"] += outages.is_down(at)
        decided.append({**f, "decided_at": at})
    res["first_alert_s"] = first_alert(decided, alert_on, temporal)
    return res


def usd(prompt_tokens: float, completion_tokens: float, nbytes: float, prices: dict) -> float:
    p = effective_prices(prices or {})
    return (prompt_tokens / 1e6 * p["usd_per_mtok_in"] + completion_tokens / 1e6 * p["usd_per_mtok_out"]
            + nbytes / 1e9 * p["usd_per_gb"])


def profile_summary(rows: list[dict], sims: list[dict], prices: dict, camera_frames: int | None = None) -> dict:
    """camera_frames: every frame the camera captured (default: the frames offered to the link); the
    "per 1,000 captured frames" figures use it, so a sparser cadence reads as cheaper per camera frame."""
    predictions = [s["first_alert_s"] is not None for s in sims]
    tta = [s["first_alert_s"] for r, s in zip(rows, sims) if r["label"] == "alert" and s["first_alert_s"] is not None]
    delays = [d for s in sims for d in s["delays_s"]]
    keys = ("frames", "sent", "decided", "dropped", "skipped", "failed", "bytes_up", "prompt_tokens",
            "completion_tokens", "decisions_during_outage", "captured_during_outage")
    tot = {k: sum(s[k] for s in sims) for k in keys}
    cost = usd(tot["prompt_tokens"], tot["completion_tokens"], tot["bytes_up"], prices)
    captured = camera_frames or tot["frames"]
    return {
        **score(rows, predictions),
        "first_alert_s_by_clip": [s["first_alert_s"] for s in sims],
        "time_to_first_alert_s_p50": _pct(tta, 50), "time_to_first_alert_s_p95": _pct(tta, 95),
        "decision_delay_s_p50": _pct(delays, 50), "decision_delay_s_p95": _pct(delays, 95),
        "max_backlog_s": max((s["max_backlog_s"] for s in sims), default=0.0),
        "frames": tot["frames"], "frames_sent": tot["sent"], "frames_decided": tot["decided"],
        "frames_dropped": tot["dropped"], "frames_skipped": tot["skipped"], "frames_failed": tot["failed"],
        "bytes_up": tot["bytes_up"], "prompt_tokens": tot["prompt_tokens"],
        "completion_tokens": tot["completion_tokens"], "decisions_during_outage": tot["decisions_during_outage"],
        "captured_during_outage": tot["captured_during_outage"],
        "usd": cost,
        "usd_per_1000_frames": cost / tot["sent"] * 1000 if tot["sent"] else None,   # per frame sent
        "camera_frames": captured,
        "usd_per_1000_captured_frames": cost / captured * 1000 if captured else None,
        "bytes_per_1000_frames": tot["bytes_up"] / tot["sent"] * 1000 if tot["sent"] else None,
        "bytes_per_1000_captured_frames": tot["bytes_up"] / captured * 1000 if captured else None,
        "tokens_per_decision": ((tot["prompt_tokens"] + tot["completion_tokens"]) / tot["decided"]
                                if tot["decided"] else None),
    }


def _ideal(clips: list[dict], alert_on: str, temporal: dict | None) -> list[bool]:
    """No network model: every answered frame decides, at its capture time."""
    out = []
    for c in clips:
        dec = [{**f, "decided_at": f["t"]} for f in c["frames"] if f["latency_ms"] is not None]
        out.append(first_alert(dec, alert_on, temporal) is not None)
    return out


def evaluate_rule(rows: list[dict], clips: list[dict], profiles: list[LinkProfile], *, outages: Outages,
                  policy: str, alert_on: str, net_baseline_ms: float | None, temporal: dict | None,
                  prices: dict, stride_filter: int = 1) -> dict:
    """Score one decision rule (per-frame or temporal) at one cadence over every link profile."""
    subset = [{**c, "frames": [f for f in c["frames"] if f["i"] % stride_filter == 0]} for c in clips]
    camera = sum(max((f["i"] for f in c["frames"]), default=-1) + 1 for c in clips)
    out = {**score(rows, _ideal(subset, alert_on, temporal)), "profiles": {}}
    for prof in profiles:
        sims = [simulate(c["frames"], prof, outages, policy, alert_on, net_baseline_ms, temporal) for c in subset]
        out["profiles"][prof.name] = profile_summary(rows, sims, prices, camera)
    return out


def bench_cloud(name: str, rows: list[dict], vlm, *, fps: float, profiles: list[LinkProfile],
                outages: Outages | None = None, policy: str = "latest", alert_on: str = "alert",
                stride: int = 1, max_side: int = 1280, prices: dict | None = None,
                net_baseline_ms: float | None = None, persist: int = 3, recheck_s: float = 5.0,
                cadence_stride: int = 20, log=lambda msg: None) -> dict:
    outages = outages or Outages()
    clips = []
    for n, row in enumerate(rows, 1):
        fr = classify_clip(row["path"], vlm, fps, stride, max_side)
        log(f"[{n}/{len(rows)}] {row['path']}: {len(fr)} frames "
            f"({vlm.n_fresh} fresh / {vlm.n_cached} cached calls so far)")
        clips.append({"path": row["path"], "label": row["label"], "frames": fr})

    frames_all = [f for c in clips for f in c["frames"]]
    fresh = [f["latency_ms"] for f in frames_all if not f["cached"] and f["latency_ms"] is not None]
    cached = [f["latency_ms"] for f in frames_all if f["cached"] and f["latency_ms"] is not None]
    billed = [f for f in frames_all if not f["cached"]]
    answered = [f for f in frames_all if f["latency_ms"] is not None]
    temporal = {"persist": persist, "recheck_s": recheck_s}
    common = dict(outages=outages, policy=policy, alert_on=alert_on, net_baseline_ms=net_baseline_ms,
                  prices=prices or {})
    per_frame = evaluate_rule(rows, clips, profiles, temporal=None, **common)
    with_logic = evaluate_rule(rows, clips, profiles, temporal=temporal, **common)
    cadence = None
    if cadence_stride and cadence_stride % stride == 0 and cadence_stride > stride:
        # same persistence in seconds: K frames at the run's stride span K*stride frames of camera time
        cad_persist = max(1, math.ceil(persist * stride / cadence_stride))
        cadence = {"stride": cadence_stride, "seconds_between_frames": cadence_stride / fps,
                   "temporal_persist": cad_persist,
                   "per_frame": evaluate_rule(rows, clips, profiles, temporal=None, stride_filter=cadence_stride, **common),
                   "temporal": evaluate_rule(rows, clips, profiles, stride_filter=cadence_stride,
                                             temporal={**temporal, "persist": cad_persist}, **common)}
    for c in clips:
        c["predicted_alert"] = any(f["severity"] in ALERT_ON[alert_on] for f in c["frames"])
        c["severities"] = {s: sum(f["severity"] == s for f in c["frames"]) for s in ("ALERT", "MONITOR", "LOG", "IGNORE")}
        c["severities"]["failed"] = sum(f["severity"] is None for f in c["frames"])
    base = net_baseline_ms or 0.0
    return {
        "name": name, "architecture": "cloud_only", "n_clips": len(rows),
        "model": vlm.model, "response_format": vlm.mode, "prompt": "CLOUD_FULL_FRAME_PROMPT",
        "fps": fps, "frame_stride": stride, "max_side": max_side, "alert_on": alert_on,
        "policy": policy, "outages": outages.windows, "persist": persist, "recheck_s": recheck_s,
        # per-frame rule, no network model (the accuracy of the cloud-only design itself)
        **{k: per_frame[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "false_alarms", "missed")},
        "profiles": per_frame["profiles"],
        "temporal": with_logic,
        "cadence": cadence,
        "frames_sent": len(frames_all), "frames_answered": len(answered),
        "n_fresh": vlm.n_fresh, "n_cached": vlm.n_cached, "n_request_errors": vlm.n_errors,
        "n_retries": getattr(vlm, "n_retries", 0),
        "parse_fail_rate": (sum(f["ctx"] is None and f["latency_ms"] is not None for f in frames_all)
                            / len(frames_all) if frames_all else None),
        "net_baseline_ms": net_baseline_ms,
        "cloud_latency_ms_p50": _pct(fresh, 50), "cloud_latency_ms_p95": _pct(fresh, 95),
        "cached_latency_ms_p50": _pct(cached, 50), "cached_latency_ms_p95": _pct(cached, 95),
        "cloud_inference_ms_p50": _pct([max(0.0, x - base) for x in fresh + cached], 50),
        "billed_prompt_tokens": sum(f["prompt_tokens"] for f in billed),
        "billed_completion_tokens": sum(f["completion_tokens"] for f in billed),
        "tokens_per_frame": (sum(f["prompt_tokens"] + f["completion_tokens"] for f in answered) / len(answered)
                             if answered else None),
        "frame_bytes_p50": _pct([f["bytes"] for f in frames_all], 50),
        "prices": effective_prices(prices or {}),
        "clips": clips,
    }


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="cloud_qwen7b", help="result tag -> results/bench_cloud_<name>.json")
    ap.add_argument("--clips", default="data/bench/clips.csv")
    ap.add_argument("--settings", default="config/settings.json", help="fps and full_frame_max_side")
    ap.add_argument("--profiles", default=DEFAULT_PROFILES, help="comma-separated link profiles")
    ap.add_argument("--policy", choices=POLICIES, default="latest", help="uplink policy (see above)")
    ap.add_argument("--frame-stride", type=int, default=1, help="send every k-th frame (dry runs, cost)")
    ap.add_argument("--cadence-stride", type=int, default=20,
                    help="also report a cheaper cadence from the same answers (20 = one frame per 10 s)")
    ap.add_argument("--persist", type=int, default=3, help="temporal rule: K consecutive positive frames")
    ap.add_argument("--recheck-s", type=float, default=5.0,
                    help="temporal rule: trend vs the answer this many s earlier (edge bench uses 5)")
    ap.add_argument("--limit", type=int, help="first N clips only (dry runs)")
    ap.add_argument("--alert-on", choices=sorted(ALERT_ON), default="alert",
                    help="clip is an alert if any frame is ALERT (or ALERT/MONITOR with 'monitor')")
    ap.add_argument("--outage", action="append", default=[], metavar="START,END",
                    help="link down from START to END sim seconds after clip start (repeatable)")
    ap.add_argument("--response-format", choices=RESPONSE_FORMATS,
                    help="overrides CLOUD_VLM_RESPONSE_FORMAT (default json_schema)")
    ap.add_argument("--parse-retries", type=int, default=0,
                    help="extra paid requests after an unparseable reply (default 0)")
    ap.add_argument("--net-baseline-ms", type=float,
                    help="skip measuring the provider's network baseline and use this value")
    ap.add_argument("--cost-inputs", default="config/cost_inputs.json",
                    help="usd_per_mtok_in/out, usd_per_gb (zeros allowed)")
    ap.add_argument("--cache", default="data/cloud_cache.jsonl")
    ap.add_argument("--max-fresh-calls", type=int, help="hard cap on billed calls in this run")
    ap.add_argument("--timeout", type=float, default=60, help="per-request timeout, seconds")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args(argv)
    if a.frame_stride < 1 or a.persist < 1:
        ap.error("--frame-stride and --persist must be >= 1")
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

    vlm = cloud_vlm_from_env(timeout_s=a.timeout, response_format=a.response_format,
                             system_prompt=CLOUD_FULL_FRAME_PROMPT, parse_retries=a.parse_retries)
    baseline = a.net_baseline_ms if a.net_baseline_ms is not None else measure_net_baseline_ms(vlm)
    cached = CachedVLM(vlm, ResponseCache(a.cache), max_fresh_calls=a.max_fresh_calls)
    print(f"cloud-only: {len(rows)} clips, stride {a.frame_stride}, model {vlm.model} at {vlm.host} "
          f"({vlm.response_format}); network baseline {baseline} ms; {len(cached.cache)} cached answers on file",
          file=sys.stderr)
    result = bench_cloud(a.name, rows, cached, fps=s.fps, profiles=a.profile_list, outages=a.outage_windows,
                         policy=a.policy, alert_on=a.alert_on, stride=a.frame_stride,
                         max_side=s.full_frame_max_side, prices=prices, net_baseline_ms=baseline,
                         persist=a.persist, recheck_s=a.recheck_s, cadence_stride=a.cadence_stride,
                         log=lambda m: print(m, file=sys.stderr))
    result.update(provider_host=vlm.host, clips_file=a.clips, limit=a.limit, parse_retries=vlm.parse_retries)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "name", "precision", "recall", "false_alarms", "missed", "frames_sent", "n_fresh", "n_cached",
        "net_baseline_ms", "cloud_latency_ms_p50", "cloud_latency_ms_p95", "billed_prompt_tokens",
        "billed_completion_tokens")}, indent=2))

    def sec(x):
        return "—" if x is None else f"{x:.1f} s"
    for label, block in (("per-frame", result), ("temporal", result["temporal"])):
        for pname, p in block["profiles"].items():
            print(f"{label:>9} {pname:>15}: recall {p['recall']:.2f}, false alarms {p['false_alarms']}, "
                  f"first alert p50 {sec(p['time_to_first_alert_s_p50'])}, decision delay p95 "
                  f"{sec(p['decision_delay_s_p95'])}, max backlog {sec(p['max_backlog_s'])}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Edge vs cloud-only when the tower's link is down: when does dispatch know about the fire?

Reads results/bench_<edge>.json (scripts/bench.py, per-clip first_alert_s, first_alert_compute_ms,
alert_payload_bytes) and results/bench_cloud_<cloud>.json (scripts/bench_cloud.py, per-frame measured
cloud answers and latency). For every link profile the link is down from clip start for
--outage-min minutes of simulated time (optionally also --outage-lead-min before it), then returns.

  edge          decides locally during the outage: every frame is analysed on the device, so local
                action is possible at once, and the ALERT waits in the outbox. Delivery follows the
                shipped code: the first send is attempted at the decision, failed sends are retried
                after 2, 4, 8 s then every MAX_BACKOFF_S (sentinel/outbox.py); the first attempt after
                the link returns goes out on the next outbox flush (+ FLUSH_EVERY_S / 2 on average,
                sentinel/main.py) as one POST on a new HTTPS connection: transfer_s(actual alert
                payload bytes incl. thumbnail, EDGE_POST_RTTS round trips).
  cloud live    after the link returns the camera streams again (policy "latest": newest frame
                whenever the uplink frees). The recorded clips are ~20 s, so when the outage outlasts
                the clip the live frames are MODELLED by holding the clip's last --live-tail answered
                frames, re-timed from outage_end (assumes the fire is still visible then).
  cloud buffer  every frame captured during the outage is queued and uploaded oldest first once the
                link returns, including modelled frames for the part of the outage outside the clip
                (median size / latency / tokens, never an alert), so its bytes cover the whole outage.
  recorded-window drop (footnote): only recorded frames after the link returns, none for a long
                outage, so it is usually "no decision"; kept for transparency.

Cloud decisions use serialization-only uplink occupancy, one round trip per request on a kept-alive
connection, and the measured latency minus the run's network baseline (see scripts/bench_cloud.py).
Paired statistics are over clips where both sides alerted. Writes results/outage_<name>.json and .md.

    python scripts/outage_scenario.py --edge after --cloud cloud_qwen7b --outage-min 10
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from sentinel.main import FLUSH_EVERY_S
from sentinel.netprofile import EDGE_POST_RTTS, LinkProfile, Outages, parse_profiles, transfer_s
from sentinel.outbox import MAX_BACKOFF_S

try:
    from scripts.bench_cloud import DEFAULT_PROFILES, simulate
    from scripts.eval_context import check_output
except ModuleNotFoundError as e:  # run as `python scripts/outage_scenario.py`
    if e.name != "scripts":
        raise
    from bench_cloud import DEFAULT_PROFILES, simulate
    from eval_context import check_output

LIVE_TAIL = 10  # held frames after the link returns (5 s at 2 fps)


def _finite(x):
    return x if x is not None and math.isfinite(x) else None


def _p50(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 50)) if xs else None


def edge_delivery_attempt_s(decided_s: float, outage_end: float) -> float:
    """Time of the first send attempt at or after outage_end, following the outbox retry schedule
    (attempt at the decision, then +2, +4, +8 s, then every MAX_BACKOFF_S)."""
    attempt, n = decided_s, 0
    while attempt < outage_end:
        n += 1
        attempt += min(MAX_BACKOFF_S, 2 ** n)
    return attempt


def edge_knows_s(clip: dict, profile: LinkProfile, outage_end: float) -> float | None:
    """Sim seconds from clip start until dispatch has the edge's first ALERT, or None."""
    if clip.get("first_alert_s") is None:
        return None
    decided = clip["first_alert_s"] + (clip.get("first_alert_compute_ms") or 0.0) / 1000
    attempt = edge_delivery_attempt_s(decided, outage_end)
    return _finite(attempt + FLUSH_EVERY_S / 2
                   + transfer_s(clip.get("alert_payload_bytes") or 0, profile, rtts=EDGE_POST_RTTS))


def with_backlog(frames: list[dict], fps: float, stride: int, lead_s: float, outage_end: float) -> list[dict]:
    """The clip's frames shifted by lead_s, plus modelled frames captured while the link is down
    outside the recorded clip (lead before it; after it until outage_end). Times are relative to
    the start of the outage, i.e. the outage window becomes [0, lead_s + outage_end)."""
    if not frames:
        return []
    dt = stride / fps

    def median(key):
        xs = [f[key] for f in frames if f.get(key) is not None]
        return float(np.median(xs)) if xs else None
    template = {"bytes": int(median("bytes")), "severity": None, "ctx": None, "latency_ms": median("latency_ms"),
                "prompt_tokens": int(median("prompt_tokens") or 0),
                "completion_tokens": int(median("completion_tokens") or 0), "modelled": True}
    shifted = [{**f, "t": f["t"] + lead_s} for f in frames]
    clip_end = max(f["t"] for f in shifted) + dt
    before = [{**template, "t": k * dt} for k in range(int(round(lead_s / dt)))]
    after = [{**template, "t": clip_end + k * dt}
             for k in range(max(0, math.ceil((lead_s + outage_end - clip_end) / dt)))]
    return before + shifted + after


def live_after_restore(frames: list[dict], fps: float, stride: int, outage_end: float,
                       tail: int = LIVE_TAIL) -> tuple[list[dict], bool]:
    """Frames the camera streams once the link is back: the recorded ones captured after
    outage_end, or (none recorded) the clip's last `tail` answered frames held and re-timed from
    outage_end. Returns (frames, modelled)."""
    real = [f for f in frames if f["t"] >= outage_end]
    if real:
        return real, False
    answered = [f for f in sorted(frames, key=lambda f: f["t"]) if f.get("latency_ms") is not None][-tail:]
    dt = stride / fps
    return [{**f, "t": outage_end + j * dt, "modelled": True} for j, f in enumerate(answered)], True


def scenario(edge: dict, cloud: dict, profiles: list[LinkProfile], outage_min: float,
             lead_min: float = 0.0, live_tail: int = LIVE_TAIL, warn=lambda m: print(m, file=sys.stderr)) -> dict:
    missing = [c["path"] for c in edge.get("clips", []) if "first_alert_s" not in c]
    if missing:
        raise SystemExit(f"{edge.get('name')}: clips lack first_alert_s (older result); re-run "
                         f"scripts/bench.py to record it (e.g. {missing[0]})")
    edge_fps, cloud_fps = (edge.get("config") or {}).get("fps"), cloud.get("fps")
    if edge_fps and cloud_fps and float(edge_fps) != float(cloud_fps):
        raise SystemExit(f"fps differs: edge {edge_fps} vs cloud {cloud_fps}; the clip clocks must match")
    fps = float(edge_fps or cloud_fps or 2.0)
    outage_end = outage_min * 60.0
    outages = Outages([(0.0, outage_end)]) if outage_end > 0 else Outages()
    lead_s = lead_min * 60.0 if outage_end > 0 else 0.0
    backlog_outages = Outages([(0.0, lead_s + outage_end)]) if outage_end > 0 else Outages()
    stride = int(cloud.get("frame_stride") or 1)
    alert_on = cloud.get("alert_on", "alert")
    base_ms = cloud.get("net_baseline_ms")
    temporal = {"persist": cloud.get("persist", 3), "recheck_s": cloud.get("recheck_s", 5.0)}
    cloud_by_path = {c["path"]: c for c in cloud.get("clips", [])}
    edge_paths = {e["path"] for e in edge["clips"]}
    unmatched = sorted((edge_paths ^ set(cloud_by_path)))
    if unmatched:
        warn(f"warning: {len(unmatched)} clip(s) appear in only one result file and are skipped: {unmatched[:3]}")
    pairs = [(e, cloud_by_path[e["path"]]) for e in edge["clips"] if e["path"] in cloud_by_path]
    if not pairs:
        raise SystemExit("no clip appears in both result files (compare the clips.csv paths)")

    clips = []
    for e, c in pairs:
        live, modelled = live_after_restore(c["frames"], fps, stride, outage_end, live_tail)
        row = {"path": e["path"], "label": e["label"], "edge_first_alert_s": e["first_alert_s"],
               "edge_frames_analysed_during_outage": sum(
                   1 for i in range(int(e.get("frames") or 0)) if outages.is_down(i / fps)),
               "cloud_live_modelled": modelled, "profiles": {}}
        for prof in profiles:
            r = {"edge_knows_s": edge_knows_s(e, prof, outage_end),
                 "edge_bytes_up": (e.get("alert_payload_bytes") or 0) if e.get("first_alert_s") is not None else 0}
            sim = simulate(live, prof, outages, "latest", alert_on, base_ms)
            r["cloud_live_knows_s"] = _finite(sim["first_alert_s"])
            r["cloud_live_bytes_up"] = sim["bytes_up"]
            r["cloud_live_decisions_during_outage"] = sim["decisions_during_outage"]
            r["cloud_live_temporal_knows_s"] = _finite(
                simulate(live, prof, outages, "latest", alert_on, base_ms, temporal)["first_alert_s"])
            if outage_end > 0:
                queued = with_backlog(c["frames"], fps, stride, lead_s, outage_end)
                sim = simulate(queued, prof, backlog_outages, "fifo", alert_on, base_ms)
                if sim["first_alert_s"] is not None:
                    sim["first_alert_s"] -= lead_s  # back to clip time
            else:
                sim = simulate(c["frames"], prof, outages, "fifo", alert_on, base_ms)
            r["cloud_buffer_knows_s"] = _finite(sim["first_alert_s"])
            r["cloud_buffer_bytes_up"] = sim["bytes_up"]
            r["cloud_buffer_queued_frames"] = sim["captured_during_outage"]
            rec = simulate(c["frames"], prof, outages, "drop", alert_on, base_ms)
            r["cloud_recorded_drop_knows_s"] = _finite(rec["first_alert_s"])
            both = r["edge_knows_s"] is not None and r["cloud_live_knows_s"] is not None
            r["delta_live_minus_edge_s"] = r["cloud_live_knows_s"] - r["edge_knows_s"] if both else None
            row["profiles"][prof.name] = r
        clips.append(row)

    summary = {}
    fires = [c for c in clips if c["label"] == "alert"]
    for prof in profiles:
        rs = [c["profiles"][prof.name] for c in fires]
        allr = [c["profiles"][prof.name] for c in clips]
        s = {"alert_clips": len(fires)}
        for side in ("edge", "cloud_live", "cloud_live_temporal", "cloud_buffer", "cloud_recorded_drop"):
            s[f"{side}_knows_s_p50"] = _p50([r[f"{side}_knows_s"] for r in rs])
            s[f"{side}_known"] = sum(r[f"{side}_knows_s"] is not None for r in rs)
        deltas = [r["delta_live_minus_edge_s"] for r in rs if r["delta_live_minus_edge_s"] is not None]
        paired = [r for r in rs if r["delta_live_minus_edge_s"] is not None]
        s.update({
            "paired_n": len(paired),
            "paired_edge_knows_s_p50": _p50([r["edge_knows_s"] for r in paired]),
            "paired_cloud_live_knows_s_p50": _p50([r["cloud_live_knows_s"] for r in paired]),
            "paired_delta_s_p50": _p50(deltas),
            "edge_frames_analysed_during_outage": sum(c["edge_frames_analysed_during_outage"] for c in clips),
            "cloud_decisions_during_outage": sum(r["cloud_live_decisions_during_outage"] for r in allr),
            "edge_bytes_up": sum(r["edge_bytes_up"] for r in allr),
            "cloud_live_bytes_up": sum(r["cloud_live_bytes_up"] for r in allr),
            "cloud_buffer_bytes_up": sum(r["cloud_buffer_bytes_up"] for r in allr),
        })
        summary[prof.name] = s

    return {"edge": edge.get("name"), "cloud": cloud.get("name"), "cloud_model": cloud.get("model"),
            "edge_recheck_s": (edge.get("config") or {}).get("recheck_s"),
            "outage_min": outage_min, "outage_lead_min": lead_min if outage_end > 0 else 0.0,
            "outage_window_s": [0.0, outage_end] if outage_end > 0 else None, "fps": fps,
            "live_tail": live_tail, "live_modelled_clips": sum(c["cloud_live_modelled"] for c in clips),
            "max_backoff_s": MAX_BACKOFF_S, "flush_every_s": FLUSH_EVERY_S, "edge_post_rtts": EDGE_POST_RTTS,
            "net_baseline_ms": base_ms, "n_clips": len(clips), "unmatched_clips": unmatched,
            "profiles": summary, "clips": clips}


def _s(x) -> str:
    return "no decision" if x is None else f"{x:.1f}"


def _d(x) -> str:
    return "—" if x is None else f"{x:+.1f}"


def _bytes(x) -> str:
    return f"{x / 1e6:.2f} MB" if x >= 1e5 else f"{x / 1e3:.1f} KB"


def markdown(res: dict) -> str:
    w = res["outage_window_s"]
    lead = res.get("outage_lead_min") or 0
    head = (f"Link down for the first {res['outage_min']:g} min of every clip (sim seconds {w[0]:.0f}–{w[1]:.0f})"
            + (f", and for {lead:g} min before it" if lead else "") if w else "No outage (link up throughout)")
    first = next(iter(res["profiles"].values()))
    recheck = res.get("edge_recheck_s")
    lines = [f"### Outage scenario: `{res['edge']}` (edge{f', recheck {recheck:g} s' if recheck else ''}) "
             f"vs `{res['cloud']}` (cloud-only)", "",
             f"{head}; {res['n_clips']} clips, {first['alert_clips']} labelled alert. Times are when dispatch "
             "knows (p50 over labelled-fire clips that alerted, sim seconds from clip start); brackets: fire "
             "clips known / labelled. Δ = cloud live − edge, paired p50 over clips where both alerted.", "",
             "| Link | Edge: dispatch knows | Cloud live after restore | Cloud live, temporal rule "
             "| Cloud buffer (FIFO backlog) | Δ paired (n) | Frames analysed during outage (edge / cloud) "
             "| Bytes up (edge / cloud live / cloud buffer) |",
             "|---|---|---|---|---|---|---|---|"]
    for name, s in res["profiles"].items():
        n = s["alert_clips"]
        lines.append(
            f"| {name} | {_s(s['edge_knows_s_p50'])} [{s['edge_known']}/{n}] "
            f"| {_s(s['cloud_live_knows_s_p50'])} [{s['cloud_live_known']}/{n}] "
            f"| {_s(s['cloud_live_temporal_knows_s_p50'])} [{s['cloud_live_temporal_known']}/{n}] "
            f"| {_s(s['cloud_buffer_knows_s_p50'])} [{s['cloud_buffer_known']}/{n}] "
            f"| {_d(s['paired_delta_s_p50'])} ({s['paired_n']}) "
            f"| {s['edge_frames_analysed_during_outage']} / {s['cloud_decisions_during_outage']} "
            f"| {_bytes(s['edge_bytes_up'])} / {_bytes(s['cloud_live_bytes_up'])} / {_bytes(s['cloud_buffer_bytes_up'])} |")
    rec = ", ".join(f"{k} {v['cloud_recorded_drop_known']}/{v['alert_clips']}" for k, v in res["profiles"].items())
    modelled = res.get("live_modelled_clips", 0)
    lines += ["",
              "What this shows: the edge analyses every frame during the outage (local action is possible at "
              "once), sends a few KB when the link returns instead of a backlog of MB, and does not need the "
              "fire to still be visible then; it wins clearly on slow links. After a clean outage on a good "
              "link, the difference in when dispatch knows is seconds.", "",
              f"Edge delivery follows the shipped outbox: retries after 2, 4, 8 s then every {res['max_backoff_s']:g} s, "
              f"next flush (+{res['flush_every_s'] / 2:g} s on average), one POST with the thumbnail on a new HTTPS "
              f"connection ({res['edge_post_rtts']} round trips). Cloud live = policy \"latest\" once the link is "
              f"back; for {modelled} clip(s) no recorded frame follows the outage, so the clip's last "
              f"{res['live_tail']} answered frames are held and re-timed from the restore (modelled: assumes the "
              "fire is still visible). Cloud buffer also queues modelled frames for the part of the outage outside "
              "the clip. Link rates and the outage are a deterministic model (`sentinel/netprofile.py`) on top of "
              "measured latencies.", "",
              f"Footnote: counting only recorded frames after the restore (drop policy), the cloud decides for "
              f"{rec} fire clips; the clips are ~20 s long, so a long outage leaves nothing recorded after it.", ""]
    return "\n".join(lines)


def load(path_or_name: str, prefix: str) -> dict:
    p = Path(path_or_name)
    if not p.suffix:
        p = Path("results") / f"{prefix}{path_or_name}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found")
    return json.loads(p.read_text())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge", default="after", help="bench name (results/bench_<edge>.json) or a path")
    ap.add_argument("--cloud", default="cloud_qwen7b", help="bench_cloud name or a path")
    ap.add_argument("--profiles", default=DEFAULT_PROFILES)
    ap.add_argument("--outage-min", type=float, default=10.0,
                    help="link down from clip start for this many minutes of sim time (0 = no outage)")
    ap.add_argument("--outage-lead-min", type=float, default=0.0,
                    help="the link was already down this many minutes before the clip started "
                         "(buffer policy: that backlog is uploaded ahead of the clip)")
    ap.add_argument("--live-tail", type=int, default=LIVE_TAIL,
                    help="answered frames held after the restore when no recorded frame follows the outage")
    ap.add_argument("--name", help="output tag (default <edge>_vs_<cloud>_<N>min)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args(argv)
    if a.outage_min < 0 or a.outage_lead_min < 0 or a.live_tail < 1:
        ap.error("--outage-min and --outage-lead-min must be >= 0, --live-tail >= 1")
    try:
        profiles = parse_profiles(a.profiles)
    except ValueError as e:
        ap.error(str(e))
    edge, cloud = load(a.edge, "bench_"), load(a.cloud, "bench_cloud_")
    name = a.name or (f"{edge.get('name')}_vs_{cloud.get('name')}_{a.outage_min:g}min"
                      + (f"_lead{a.outage_lead_min:g}" if a.outage_lead_min else ""))
    out = Path("results") / f"outage_{name}.json"
    check_output(out, a.force)

    res = {"name": name, **scenario(edge, cloud, profiles, a.outage_min, a.outage_lead_min, a.live_tail)}
    md = markdown(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

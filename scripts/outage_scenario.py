"""Edge vs cloud-only when the tower's link is down: when does dispatch know about the fire?

Reads results/bench_<edge>.json (scripts/bench.py, which records per-clip first_alert_s,
first_alert_compute_ms and alert_payload_bytes) and results/bench_cloud_<cloud>.json
(scripts/bench_cloud.py, per-frame measured cloud answers and latency). For every link profile the
link is down from clip start for --outage-min minutes of simulated time, then comes back:

  edge        decides locally during the outage (every frame is analysed on the device) and holds
              the ALERT in its outbox; dispatch knows at
              max(first_alert_s + measured edge compute, outage_end) + transfer_s(alert bytes)
  cloud drop  frames captured while the link is down are never decided; dispatch knows at the first
              ALERT among frames captured after outage_end, plus its upload and measured latency
  cloud buffer frames are queued and uploaded serially (oldest first) once the link returns. The
              camera keeps capturing for the whole outage, not just the recorded clip, so the queue
              also holds modelled frames (the clip's median size, latency and tokens; never an alert)
              for the rest of the outage, and for --outage-lead-min minutes before the clip if the
              link went down before the fire appeared: those are uploaded ahead of the clip's frames.

The clips are ~20 s long, so with a multi-minute outage the cloud (drop) usually has NO frame after
the link returns inside the recorded clip: that is reported as "no decision in the recorded window"
(null), not as a time. Link profiles and outages are a deterministic model (sentinel/netprofile.py)
on top of the real measured latencies. Writes results/outage_<name>.json and results/outage_<name>.md.

    python scripts/outage_scenario.py --edge after --cloud cloud_qwen7b --outage-min 10
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from sentinel.netprofile import LinkProfile, Outages, parse_profiles, transfer_s

try:
    from scripts.bench_cloud import DEFAULT_PROFILES, simulate
    from scripts.eval_context import check_output
except ModuleNotFoundError as e:  # run as `python scripts/outage_scenario.py`
    if e.name != "scripts":
        raise
    from bench_cloud import DEFAULT_PROFILES, simulate
    from eval_context import check_output

POLICIES = ("drop", "buffer")


def _finite(x):
    return x if x is not None and math.isfinite(x) else None


def _p50(xs):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 50)) if xs else None


def edge_knows_s(clip: dict, profile: LinkProfile, outage_end: float) -> float | None:
    """Sim seconds from clip start until dispatch has the edge's first ALERT, or None."""
    if clip.get("first_alert_s") is None:
        return None
    decided = clip["first_alert_s"] + (clip.get("first_alert_compute_ms") or 0.0) / 1000
    return _finite(max(decided, outage_end) + transfer_s(clip.get("alert_payload_bytes") or 0, profile))


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
    template = {"bytes": int(median("bytes")), "severity": None, "latency_ms": median("latency_ms"),
                "prompt_tokens": int(median("prompt_tokens") or 0),
                "completion_tokens": int(median("completion_tokens") or 0), "modelled": True}
    shifted = [{**f, "t": f["t"] + lead_s} for f in frames]
    clip_end = max(f["t"] for f in shifted) + dt
    before = [{**template, "t": k * dt} for k in range(int(round(lead_s / dt)))]
    after = [{**template, "t": clip_end + k * dt}
             for k in range(max(0, math.ceil((lead_s + outage_end - clip_end) / dt)))]
    return before + shifted + after


def scenario(edge: dict, cloud: dict, profiles: list[LinkProfile], outage_min: float,
             lead_min: float = 0.0) -> dict:
    missing = [c["path"] for c in edge.get("clips", []) if "first_alert_s" not in c]
    if missing:
        raise SystemExit(f"{edge.get('name')}: clips lack first_alert_s (older result); re-run "
                         f"scripts/bench.py with --force to record it (e.g. {missing[0]})")
    fps = edge.get("config", {}).get("fps") or cloud.get("fps") or 2.0
    outage_end = outage_min * 60.0
    outages = Outages([(0.0, outage_end)]) if outage_end > 0 else Outages()
    lead_s = lead_min * 60.0 if outage_end > 0 else 0.0
    backlog_outages = Outages([(0.0, lead_s + outage_end)]) if outage_end > 0 else Outages()
    stride = int(cloud.get("frame_stride") or 1)
    cloud_by_path = {c["path"]: c for c in cloud.get("clips", [])}
    pairs = [(e, cloud_by_path[e["path"]]) for e in edge["clips"] if e["path"] in cloud_by_path]
    if not pairs:
        raise SystemExit("no clip appears in both result files (compare the clips.csv paths)")

    clips = []
    for e, c in pairs:
        captured_down = sum(1 for i in range(int(e.get("frames") or 0)) if outages.is_down(i / fps))
        row = {"path": e["path"], "label": e["label"], "edge_first_alert_s": e["first_alert_s"],
               "edge_decisions_during_outage": captured_down, "profiles": {}}
        for prof in profiles:
            r = {"edge_knows_s": edge_knows_s(e, prof, outage_end),
                 "edge_bytes_up": (e.get("alert_payload_bytes") or 0) if e.get("first_alert_s") is not None else 0}
            for pol in POLICIES:
                if pol == "buffer" and outage_end > 0:
                    queued = with_backlog(c["frames"], fps, stride, lead_s, outage_end)
                    sim = simulate(queued, prof, backlog_outages, pol, cloud.get("alert_on", "alert"))
                    if sim["first_alert_s"] is not None:
                        sim["first_alert_s"] -= lead_s  # back to clip time
                    r["cloud_buffer_queued_frames"] = sim["captured_during_outage"]
                else:
                    sim = simulate(c["frames"], prof, outages, pol, cloud.get("alert_on", "alert"))
                r[f"cloud_{pol}_knows_s"] = _finite(sim["first_alert_s"])
                r[f"cloud_{pol}_bytes_up"] = sim["bytes_up"]
                r[f"cloud_{pol}_decisions_during_outage"] = sim["decisions_during_outage"]
                r[f"cloud_{pol}_frames_dropped"] = sim["dropped"]
            row["profiles"][prof.name] = r
        clips.append(row)

    summary = {}
    fires = [c for c in clips if c["label"] == "alert"]
    for prof in profiles:
        rs = [c["profiles"][prof.name] for c in fires]
        allr = [c["profiles"][prof.name] for c in clips]
        s = {"alert_clips": len(fires),
             "edge_knows_s_p50": _p50([r["edge_knows_s"] for r in rs]),
             "edge_known": sum(r["edge_knows_s"] is not None for r in rs),
             "edge_decisions_during_outage": sum(c["edge_decisions_during_outage"] for c in clips),
             "edge_bytes_up": sum(r["edge_bytes_up"] for r in allr)}
        for pol in POLICIES:
            s[f"cloud_{pol}_knows_s_p50"] = _p50([r[f"cloud_{pol}_knows_s"] for r in rs])
            s[f"cloud_{pol}_known"] = sum(r[f"cloud_{pol}_knows_s"] is not None for r in rs)
            s[f"cloud_{pol}_decisions_during_outage"] = sum(r[f"cloud_{pol}_decisions_during_outage"] for r in allr)
            s[f"cloud_{pol}_bytes_up"] = sum(r[f"cloud_{pol}_bytes_up"] for r in allr)
        summary[prof.name] = s

    return {"edge": edge.get("name"), "cloud": cloud.get("name"), "outage_min": outage_min,
            "outage_lead_min": lead_min if outage_end > 0 else 0.0,
            "outage_window_s": [0.0, outage_end] if outage_end > 0 else None, "fps": fps,
            "n_clips": len(clips), "profiles": summary, "clips": clips}


def _s(x) -> str:
    return "no decision" if x is None else f"{x:.1f}"


def _bytes(x) -> str:
    return f"{x / 1e6:.2f} MB" if x >= 1e5 else f"{x / 1e3:.1f} KB"


def markdown(res: dict) -> str:
    w = res["outage_window_s"]
    lead = res.get("outage_lead_min") or 0
    head = (f"Link down for the first {res['outage_min']:g} min of every clip (sim seconds {w[0]:.0f}–{w[1]:.0f})"
            + (f", and for {lead:g} min before it" if lead else "") if w else "No outage (link up throughout)")
    lines = [f"### Outage scenario: `{res['edge']}` (edge) vs `{res['cloud']}` (cloud-only)", "",
             f"{head}; {res['n_clips']} clips, {next(iter(res['profiles'].values()))['alert_clips']} labelled alert. "
             "Time is when dispatch knows (p50 over labelled-fire clips that produced an alert, sim seconds "
             "from clip start); counts in brackets are fire clips known / labelled.", "",
             "| Link | Edge: dispatch knows | Cloud (drop) | Cloud (buffer) | Decisions during outage (edge / cloud) "
             "| Bytes up (edge / cloud buffer) |",
             "|---|---|---|---|---|---|"]
    for name, s in res["profiles"].items():
        n = s["alert_clips"]
        lines.append(
            f"| {name} | {_s(s['edge_knows_s_p50'])} [{s['edge_known']}/{n}] "
            f"| {_s(s['cloud_drop_knows_s_p50'])} [{s['cloud_drop_known']}/{n}] "
            f"| {_s(s['cloud_buffer_knows_s_p50'])} [{s['cloud_buffer_known']}/{n}] "
            f"| {s['edge_decisions_during_outage']} / {s['cloud_drop_decisions_during_outage']} "
            f"| {_bytes(s['edge_bytes_up'])} / {_bytes(s['cloud_buffer_bytes_up'])} |")
    lines += ["", "Edge = measured local decision (sim time + measured on-device compute), held in the outbox and "
              "sent when the link returns. Cloud = measured per-frame cloud answers and latency. Link rates and "
              "the outage are a deterministic model (`sentinel/netprofile.py`); \"no decision\" means no ALERT "
              "reached dispatch from the recorded clip (for cloud-drop: no frame after the link returned). "
              "Cloud (buffer) also queues the modelled frames the camera captures during the rest of the "
              "outage, so its bytes are for the whole outage.", ""]
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
    ap.add_argument("--name", help="output tag (default <edge>_vs_<cloud>_<N>min)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args(argv)
    if a.outage_min < 0 or a.outage_lead_min < 0:
        ap.error("--outage-min and --outage-lead-min must be >= 0")
    try:
        profiles = parse_profiles(a.profiles)
    except ValueError as e:
        ap.error(str(e))
    edge, cloud = load(a.edge, "bench_"), load(a.cloud, "bench_cloud_")
    name = a.name or (f"{edge.get('name')}_vs_{cloud.get('name')}_{a.outage_min:g}min"
                      + (f"_lead{a.outage_lead_min:g}" if a.outage_lead_min else ""))
    out = Path("results") / f"outage_{name}.json"
    check_output(out, a.force)

    res = {"name": name, **scenario(edge, cloud, profiles, a.outage_min, a.outage_lead_min)}
    md = markdown(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + "\n")
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

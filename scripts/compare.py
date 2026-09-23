"""Build results/before_after.md from whichever result files exist in results/.

Reads detector_*.json (scripts/eval_detector.py), context_*.json (scripts/eval_context.py,
scripts/linear_probe.py), bench_*.json (scripts/bench.py) and, for the "Edge vs cloud-only"
section, context_cloud_*.json (eval_context.py --cloud), bench_cloud_*.json (scripts/bench_cloud.py)
and outage_*.json (scripts/outage_scenario.py). Missing files are skipped and listed in a one-line
note, so this can run at any point in the Nano schedule.

    python scripts/compare.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sentinel.netprofile import PROFILES

try:
    from scripts.outage_scenario import edge_knows_s
    from scripts.outage_scenario import markdown as outage_markdown
except ModuleNotFoundError as e:  # run as `python scripts/compare.py`
    if e.name != "scripts":
        raise
    from outage_scenario import edge_knows_s
    from outage_scenario import markdown as outage_markdown

PREFIXES = ("detector_", "context_", "bench_", "outage_")

DETECTOR_RUNS = [("detector_before_yoloworld", "Before: YOLO-World zero-shot"),
                 ("detector_after_yolo11s", "After: YOLO11s fine-tuned")]
DETECTOR_METRICS = [("mAP50", "map50", "pct"), ("mAP50-95", "map50_95", "pct"),
                    ("Precision", "precision", "pct"), ("Recall", "recall", "pct"),
                    ("ms/image", "ms_per_image", "ms")]

CONTEXT_RUNS = [("context_linear_probe", "SigLIP linear probe (cheap floor)"),
                ("context_before_base7b", "Before: Qwen2.5-VL-7B base"),
                ("context_after_lora7b", "After: 7B + LoRA distilled from 32B"),
                ("context_ref_teacher32b", "Reference: 32B teacher (upper bound)"),
                ("context_before_base7b_gold", "Before, hand-checked gold split"),
                ("context_after_lora7b_gold", "After, hand-checked gold split")]
BENCH_ORDER = ["bench_before", "bench_after"]

# Reported as missing when absent; the other known runs are optional.
EXPECTED = ["detector_before_yoloworld", "detector_after_yolo11s", "context_linear_probe",
            "context_before_base7b", "context_after_lora7b", "bench_before", "bench_after"]

DASH = "—"


def pct(x) -> str:
    return DASH if x is None else f"{x * 100:.1f}%"


def ms(x) -> str:
    return DASH if x is None else f"{x:.0f}"


def num(x, digits: int = 0) -> str:
    return DASH if x is None else f"{x:.{digits}f}"


def delta(before, after, kind: str) -> str:
    if before is None or after is None:
        return DASH
    d = after - before
    return f"{d * 100:+.1f} pp" if kind == "pct" else f"{d:+.1f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(r) + " |" for r in rows)]


def detector_section(results: dict) -> list[str]:
    runs = [(key, label) for key, label in DETECTOR_RUNS if key in results]
    if not runs:
        return []
    both = len(runs) == 2
    header = ["Metric", *(label for _, label in runs)] + (["Δ (after − before)"] if both else [])
    rows = []
    for label, field, kind in DETECTOR_METRICS:
        values = [results[key].get(field) for key, _ in runs]
        fmt = pct if kind == "pct" else (lambda v: num(v, 1))
        rows.append([label, *(fmt(v) for v in values)] + ([delta(*values, kind)] if both else []))
    return ["## Detector (D-Fire test split)", "",
            *table(header, rows), "",
            "Same held-out split for both rows. BEFORE has no smoke training: it is prompted with the "
            "text classes `smoke, fire`.", ""]


def context_section(results: dict) -> list[str]:
    known = [key for key, _ in CONTEXT_RUNS]
    extra = sorted(k for k in results if k.startswith("context_") and k not in known)
    labels = dict(CONTEXT_RUNS)
    rows = []
    for key in [k for k in known if k in results] + extra:
        r = results[key]
        latency = r.get("latency_ms_p50", r.get("latency_ms_per_image"))
        rows.append([f"`{r.get('name', key.removeprefix('context_'))}`", labels.get(key, r.get("model", DASH)),
                     num(r.get("n")), pct(r.get("source_type_acc")), pct(r.get("group_acc")),
                     pct(r.get("parse_fail_rate")), ms(latency), num(r.get("tokens_per_call"))])
    if not rows:
        return []
    header = ["Run", "What", "n", "Source-type acc", "Group acc", "Parse-fail", "p50 ms", "Tokens/call"]
    return ["## Context VLM (held-out split)", "",
            *table(header, rows), "",
            "Accuracy on teacher labels measures distillation (agreement with the 32B teacher), not ground "
            "truth; `_gold` rows are scored on hand-checked labels. Group = danger / benign / look-alike.", ""]


def bench_section(results: dict) -> list[str]:
    keys = [k for k in BENCH_ORDER if k in results]
    keys += sorted(k for k in results if k.startswith("bench_") and k not in BENCH_ORDER
                   and not k.startswith("bench_cloud_"))
    rows = [[f"`{results[k].get('name', k.removeprefix('bench_'))}`", pct(results[k].get("precision")),
             pct(results[k].get("recall")), num(results[k].get("false_alarms")), num(results[k].get("missed")),
             num(results[k].get("tokens_per_vlm_call")), num(results[k].get("frames_per_vlm_call"), 1),
             num(results[k].get("time_to_decision_s_p50"), 1), num(results[k].get("time_to_decision_s_p95"), 1)]
            for k in keys]
    if not rows:
        return []
    header = ["Run", "Precision", "Recall", "False alarms", "Missed", "Tokens/VLM call", "Frames per VLM call",
              "Decision p50 (sim s)", "Decision p95 (sim s)"]
    return ["## End-to-end (labeled clips, `scripts/bench.py`)", "",
            *table(header, rows), "",
            "Clip-level: a clip counts as an alert if the cascade raised ALERT on it. Time to decision is in "
            "simulated clip seconds (frames / fps) from event open to final severity; it excludes VLM latency.", ""]


# ---------------------------------------------------------------- edge vs cloud-only

def _keys(results: dict, prefix: str) -> list[str]:
    return sorted(k for k in results if k.startswith(prefix))


def _usd(x) -> str:
    return DASH if x is None else f"${x:.4f}"


def _mb(x) -> str:
    return DASH if x is None else f"{x / 1e6:.2f} MB"


def _edge_bench_keys(results: dict) -> list[str]:
    """Edge bench runs, BEFORE first, then AFTER, then any other (e.g. a recheck-30 run)."""
    keys = [k for k in BENCH_ORDER if k in results]
    return keys + [k for k in _keys(results, "bench_") if not k.startswith("bench_cloud_") and k not in keys]


def _edge_timed(b: dict | None) -> bool:
    return b is not None and bool(b.get("clips")) and all("first_alert_s" in c for c in b["clips"])


def _edge_label(b: dict, key: str) -> str:
    recheck = (b.get("config") or {}).get("recheck_s")
    return f"`{b.get('name', key)}` (recheck {recheck:g} s)" if recheck is not None else f"`{b.get('name', key)}`"


def _edge_alert_bytes_per_1000(b: dict) -> float | None:
    clips = b.get("clips") or []
    if not clips or any("alert_payload_bytes" not in c for c in clips):
        return None
    frames = sum(c.get("frames") or 0 for c in clips)
    return sum(c["alert_payload_bytes"] or 0 for c in clips) / frames * 1000 if frames else None


def _edge_knows_by_path(b: dict, profile) -> dict:
    """Edge, no outage: first ALERT + measured compute + outbox flush + alert POST (outage_scenario)."""
    return {c["path"]: edge_knows_s(c, profile, 0.0) for c in b.get("clips") or [] if c.get("label") == "alert"}


def _cloud_knows_by_path(b: dict, block: dict, name: str) -> dict:
    per_clip = (block.get("profiles") or {}).get(name, {}).get("first_alert_s_by_clip")
    if per_clip is None:
        return {}
    return {c["path"]: t for c, t in zip(b.get("clips") or [], per_clip) if c.get("label") == "alert"}


def _p50(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, 50)) if xs else None


def _cloud_rows(b: dict, key: str) -> list[tuple[str, dict]]:
    """(label, block) for each cloud decision rule / cadence present in a bench_cloud result."""
    k = b.get("frame_stride", 1)
    rows = [("per-frame, every frame" if k == 1 else f"per-frame, every {k}th frame", b)]
    if b.get("temporal"):
        rows.append((f"temporal (persist {b.get('persist')}, trend over {b.get('recheck_s', 0):g} s)", b["temporal"]))
    cad = b.get("cadence") or {}
    for rule in ("per_frame", "temporal"):
        if cad.get(rule):
            rows.append((f"{rule.replace('_', '-')}, one frame per {cad.get('seconds_between_frames', 0):g} s", cad[rule]))
    return rows


def edge_vs_cloud_section(results: dict) -> list[str]:
    ctx_cloud = [k for k in _keys(results, "context_") if results[k].get("cloud")]
    bench_cloud, outages = _keys(results, "bench_cloud_"), _keys(results, "outage_")
    if not (ctx_cloud or bench_cloud or outages):
        return []
    model = next((results[k].get("model") for k in bench_cloud + ctx_cloud if results[k].get("model")),
                 None) or next((results[k].get("cloud_model") for k in outages), None) or "unknown model"
    out = ["## Edge vs cloud-only (measured)", "",
           f"Cloud-only = `{model}` (as recorded by the runs; base weights: a hosted provider cannot run our "
           "LoRA) behind an OpenAI-compatible API; every frame is sent. Cloud latency is the real HTTPS round "
           "trip from the machine that ran the eval, minus its measured network baseline when a link profile "
           "is added; link profiles are a model layered on top of it.", ""]
    missing = [what for what, have in (("cloud context eval (`eval_context.py --cloud`)", ctx_cloud),
                                       ("cloud clip bench (`bench_cloud_*.json`)", bench_cloud),
                                       ("outage scenario (`outage_*.json`)", outages)) if not have]
    if missing:
        out += [f"_Not run yet, omitted: {', '.join(missing)}._", ""]

    if ctx_cloud:
        rows = []
        for key, where in (("context_before_base7b", "edge (Nano), base 7B"),
                           ("context_after_lora7b", "edge (Nano), 7B + LoRA")):
            if key in results:
                r = results[key]
                rows.append([f"`{r.get('name', key)}`", where, num(r.get("n")), pct(r.get("source_type_acc")),
                             pct(r.get("group_acc")), pct(r.get("parse_fail_rate")), ms(r.get("latency_ms_p50")),
                             ms(r.get("latency_ms_p95")), "0 (local)"])
        for key in ctx_cloud:
            r = results[key]
            fresh = r.get("latency_ms_p50") is not None
            p50 = r.get("latency_ms_p50") if fresh else r.get("cached_latency_ms_p50")
            p95 = r.get("latency_ms_p95") if fresh else r.get("cached_latency_ms_p95")
            per = [r.get("prompt_tokens_per_call"), r.get("completion_tokens_per_call")]
            billed = DASH if per[0] is None else f"{per[0]:.0f} in + {per[1]:.0f} out"
            where = f"cloud ({r.get('provider_host', '?')}, {r.get('response_format', '?')})"
            rows.append([f"`{r.get('name', key)}`", where + ("" if fresh else ", latency from cache"),
                         num(r.get("n")), pct(r.get("source_type_acc")), pct(r.get("group_acc")),
                         pct(r.get("parse_fail_rate")), ms(p50), ms(p95), billed])
        out += ["### Context classification, same held-out crops", "",
                *table(["Run", "Where", "n", "Source-type acc", "Group acc", "Parse-fail", "p50 ms", "p95 ms",
                        "Billed tokens / decision"], rows), "",
                "Edge latency is on-device inference; cloud latency includes the real network from the eval "
                "machine. A cloud run with `--limit` scores fewer crops (see n).", ""]

    edge_keys = _edge_bench_keys(results)
    if bench_cloud:
        b = results[bench_cloud[0]]
        priced = any((b.get("prices") or {}).get(k) for k in ("usd_per_mtok_in", "usd_per_mtok_out", "usd_per_gb"))
        first_profile = next(iter(b.get("profiles") or {}), None)
        rows = []
        for key in edge_keys:
            e = results[key]
            rows.append([_edge_label(e, key), "edge cascade", pct(e.get("recall")), num(e.get("false_alarms")),
                         num(e.get("missed")), "$0 API", _mb(_edge_alert_bytes_per_1000(e))])
        for label, block in _cloud_rows(b, bench_cloud[0]):
            fp = (block.get("profiles") or {}).get(first_profile, {})
            rows.append([f"`{b.get('name')}`", f"cloud-only, {label}", pct(block.get("recall")),
                         num(block.get("false_alarms")), num(block.get("missed")),
                         _usd(fp.get("usd_per_1000_captured_frames")) if priced else "unpriced",
                         _mb(fp.get("bytes_per_1000_captured_frames"))])
        out += ["### Tower clips (end-to-end)", "",
                *table(["Run", "Architecture / rule", "Recall", "False alarms", "Missed",
                        "$ / 1,000 camera frames", "Bytes up / 1,000 camera frames"], rows), "",
                "Recall and false alarms: no network model (every answered frame decides). Edge bytes = alert "
                "reports (with thumbnail) only; edge API cost is $0 (Nano hardware and power not included). "
                f"Cloud $ and bytes are on `{first_profile}` with policy `{b.get('policy', 'latest')}` = "
                "provider-reported tokens x `config/cost_inputs.json` prices; \"unpriced\" while those are 0.", ""]

        timed = [k for k in edge_keys if _edge_timed(results[k])]
        blocks = _cloud_rows(b, bench_cloud[0])
        header = ["Link", *(f"Edge {_edge_label(results[k], k)}" for k in timed),
                  *(f"Cloud {label}" for label, _ in blocks),
                  "Cloud decision delay p95 (s)", "Cloud max backlog (s)", "Cloud $ / 1,000 frames sent"]
        rows = []
        for name, p in (b.get("profiles") or {}).items():
            prof = PROFILES.get(name)
            rows.append([name,
                         *(num(_p50(_edge_knows_by_path(results[k], prof).values()), 1) if prof else DASH
                           for k in timed),
                         *(num(((blk.get("profiles") or {}).get(name) or {}).get("time_to_first_alert_s_p50"), 1)
                           for _, blk in blocks),
                         num(p.get("decision_delay_s_p95"), 1), num(p.get("max_backlog_s"), 1),
                         _usd(p.get("usd_per_1000_frames")) if priced else "unpriced"])
        notes = []
        if not timed:
            notes.append("Edge columns need a `bench.py` result with `first_alert_s` (re-run the edge bench).")
        rechecks = {(results[k].get("config") or {}).get("recheck_s") for k in timed}
        if timed and 30 not in rechecks and 30.0 not in rechecks:
            notes.append("The deployed setting is recheck_s = 30 s (`Settings().recheck_s`); the edge rows use "
                         f"{', '.join(f'{r:g} s' for r in sorted(x for x in rechecks if x is not None))}. Add "
                         "`python scripts/bench.py --name after_recheck30 --detector-weights ... --model context "
                         "--recheck-s 30` for the deployed timing.")
        windows = b.get("outages") or []
        if windows:
            notes.append(f"Cloud run had outage windows {windows} ({b.get('policy')}).")
        out += [f"### Time to dispatch per link, no outage (`{b.get('name')}`, policy `{b.get('policy', 'latest')}`)", "",
                *table(header, rows), "",
                "p50 sim seconds from clip start to when dispatch has the alert, over labelled-fire clips that "
                "alerted (each column over its own clips; see the paired table). Edge = first ALERT + measured "
                "on-device compute + outbox flush + one POST on a new HTTPS connection. Cloud frames share one "
                "modelled uplink." + (" " + " ".join(notes) if notes else ""), ""]

        # paired: clips where both the edge run and the cloud (per-frame) alerted
        primary = "bench_after" if "bench_after" in timed else (timed[-1] if timed else None)
        prows, deltas_rows = [], []
        for k in timed:
            for name in (b.get("profiles") or {}):
                prof = PROFILES.get(name)
                if prof is None:
                    continue
                ek, ck = _edge_knows_by_path(results[k], prof), _cloud_knows_by_path(b, b, name)
                both = [pth for pth in ek if ek[pth] is not None and ck.get(pth) is not None]
                prows.append([_edge_label(results[k], k), name, str(len(both)),
                              num(_p50([ek[x] for x in both]), 1), num(_p50([ck[x] for x in both]), 1),
                              num(_p50([ck[x] - ek[x] for x in both]), 1)])
                if k == primary and name == first_profile:
                    deltas_rows = [[f"`{Path(x).name}`", num(ek[x], 1), num(ck[x], 1), f"{ck[x] - ek[x]:+.1f}"]
                                   for x in both]
        if prows:
            out += ["### Paired: clips where both sides alerted", "",
                    *table(["Edge run", "Link", "n", "Edge p50 (s)", "Cloud per-frame p50 (s)",
                            "Δ cloud − edge p50 (s)"], prows), ""]
        if deltas_rows:
            out += [f"Per clip on `{first_profile}`, edge run {_edge_label(results[primary], primary)}:", "",
                    *table(["Clip", "Edge (s)", "Cloud (s)", "Δ (s)"], deltas_rows), ""]

    for key in outages:
        out += [outage_markdown(results[key])]
    return out


def render(results: dict) -> str:
    """results maps a result-file stem (e.g. "detector_after_yolo11s") to its parsed JSON."""
    out = ["# Before vs after fine-tuning", "",
           "Generated by `python scripts/compare.py` from `results/*.json`; do not edit by hand.", ""]
    missing = [f"`{k}.json`" for k in EXPECTED if k not in results]
    if missing:
        out += [f"_Missing (not run yet): {', '.join(missing)}._", ""]
    out += (detector_section(results) + context_section(results) + bench_section(results)
            + edge_vs_cloud_section(results))
    return "\n".join(out).rstrip() + "\n"


def load_results(results_dir) -> dict:
    return {p.stem: json.loads(p.read_text())
            for p in sorted(Path(results_dir).glob("*.json")) if p.stem.startswith(PREFIXES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/before_after.md")
    a = ap.parse_args()
    md = render(load_results(a.results_dir))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)  # derived from the JSON files, so regenerating it is always safe
    print(md)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

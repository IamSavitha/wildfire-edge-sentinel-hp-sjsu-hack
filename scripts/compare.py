"""Build results/before_after.md from whichever result files exist in results/.

Reads detector_*.json (scripts/eval_detector.py), context_*.json (scripts/eval_context.py,
scripts/linear_probe.py) and bench_*.json (scripts/bench.py). Missing files are skipped and
listed in a one-line note, so this can run at any point in the Nano schedule.

    python scripts/compare.py
"""
import argparse
import json
from pathlib import Path

PREFIXES = ("detector_", "context_", "bench_")

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
    return f"{d * 100:+.1f} pp" if kind == "pct" else f"{d:+.0f}"


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
        fmt = pct if kind == "pct" else ms
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
    keys += sorted(k for k in results if k.startswith("bench_") and k not in BENCH_ORDER)
    rows = [[f"`{results[k].get('name', k.removeprefix('bench_'))}`", pct(results[k].get("precision")),
             pct(results[k].get("recall")), num(results[k].get("false_alarms")), num(results[k].get("missed")),
             num(results[k].get("tokens_per_vlm_call")), num(results[k].get("frames_per_vlm_call"), 1)]
            for k in keys]
    if not rows:
        return []
    header = ["Run", "Precision", "Recall", "False alarms", "Missed", "Tokens/VLM call", "Frames per VLM call"]
    return ["## End-to-end (labeled clips, `scripts/bench.py`)", "",
            *table(header, rows), "",
            "Clip-level: a clip counts as an alert if the cascade raised ALERT on it.", ""]


def render(results: dict) -> str:
    """results maps a result-file stem (e.g. "detector_after_yolo11s") to its parsed JSON."""
    out = ["# Before vs after fine-tuning", "",
           "Generated by `python scripts/compare.py` from `results/*.json`; do not edit by hand.", ""]
    missing = [f"`{k}.json`" for k in EXPECTED if k not in results]
    if missing:
        out += [f"_Missing (not run yet): {', '.join(missing)}._", ""]
    out += detector_section(results) + context_section(results) + bench_section(results)
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

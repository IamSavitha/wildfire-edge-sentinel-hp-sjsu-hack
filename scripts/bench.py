"""End-to-end benchmark over labeled clips with a simulated clock (deterministic, no sleeps).
Writes results/bench_<name>.json. Run on the Nano with sentinel.main stopped (frees the GPU).

clips.csv rows are `path,label` where path is a video or a folder of time-ordered images and
label is `alert` (a fire that should page dispatch) or `no_alert` (campfire, fog, stack, BBQ...).

BEFORE: --name before --detector-weights yolov8s-worldv2.pt --detector-classes smoke,fire --model "<base id>" --recheck-s 5
AFTER:  --name after  --detector-weights models/smoke_yolo.pt --model context --recheck-s 5
Use the same --recheck-s for both runs, below the clip length, so a growing fire can escalate in-clip.
Ablations: --detector-only (any candidate = alert, no VLM), --full-frame (no crop).
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from sentinel.config import Settings, Tower, load_settings
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.replayer import frames
from sentinel.schema import Severity

try:
    from scripts.eval_context import check_output
except ModuleNotFoundError as e:  # run as `python scripts/bench.py`: scripts/ is on sys.path, not the repo root
    if e.name != "scripts":
        raise
    from eval_context import check_output

LABELS = {"alert", "no_alert"}


class NullVLM:
    """Detector-only ablation: no context model; the pipeline falls back to trend rules."""

    def classify(self, jpeg):
        return None, 0


def read_clips(path) -> list[dict]:
    """Reads clips.csv and checks labels up front, so a typo fails before an hour of replay."""
    with open(path, newline="") as f:
        rows = [{"path": r["path"].strip(), "label": (r.get("label") or "").strip()}
                for r in csv.DictReader(f) if r.get("path") and r["path"].strip()]
    bad = [r for r in rows if r["label"] not in LABELS]
    if bad:
        raise ValueError(f"labels must be one of {sorted(LABELS)}; bad rows: {bad[:3]}")
    return rows


def run_clip(path: str, settings: Settings, detector, vlm, metrics: Metrics) -> list[Severity]:
    """Replay one clip through a fresh pipeline; returns final severities plus any still-provisional one."""
    tower = Tower(id="bench", name=Path(path).stem, lat=37.0, lon=-121.0, source=path)
    esc = Escalator(Outbox(":memory:"), Link(), lambda *a: {}, lambda p: None, metrics)
    pipe = Pipeline({"bench": tower}, detector, vlm, esc, settings, metrics)
    for i, frame in enumerate(frames(path, settings.fps, loop=False)):
        pipe.process("bench", frame, now=i / settings.fps)
    return ([e.severity for e in pipe.history]
            + [e.severity for e in pipe.active.values() if e.severity is not None])


def score(rows: list[dict], predictions: list[bool]) -> dict:
    """Clip-level confusion counts. A clip is positive if it produced an ALERT."""
    if len(rows) != len(predictions):
        raise ValueError(f"{len(rows)} clips but {len(predictions)} predictions")
    tp = fp = fn = tn = 0
    for row, predicted in zip(rows, predictions):
        if row["label"] not in LABELS:
            raise ValueError(f"label must be one of {sorted(LABELS)}, got {row['label']!r}")
        actual = row["label"] == "alert"
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not predicted and not actual
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_alarms": fp, "missed": fn}


def summarize(name: str, scores: dict, metrics: dict, clips: list[dict], detector_only: bool) -> dict:
    calls = 0 if detector_only else metrics.get("vlm_calls", 0)
    return {
        "name": name,
        "detector_only": detector_only,
        "n_clips": len(clips),
        **scores,
        "tokens_per_vlm_call": metrics.get("vlm_tokens", 0) / calls if calls else 0,
        "frames_per_vlm_call": metrics.get("frames", 0) / calls if calls else None,
        "time_to_decision_s_p50": metrics.get("decision_s_p50"),
        "time_to_decision_s_p95": metrics.get("decision_s_p95"),
        "metrics": metrics,
        "clips": clips,
    }


def bench(name: str, rows: list[dict], settings: Settings, detector, vlm,
          detector_only: bool = False) -> dict:
    total, clips, predictions = Metrics(), [], []
    for row in rows:
        m = Metrics()
        severities = run_clip(row["path"], settings, detector, vlm, m)
        predicted = m.counters["candidates"] > 0 if detector_only else Severity.ALERT in severities
        predictions.append(predicted)
        clips.append({"path": row["path"], "label": row["label"], "predicted_alert": predicted,
                      "severities": [s.name for s in severities],
                      "frames": m.counters["frames"], "vlm_calls": m.counters["vlm_calls"]})
        total.merge(m)
    return summarize(name, score(rows, predictions), total.summary(), clips, detector_only)


def apply_overrides(s: Settings, a: argparse.Namespace) -> Settings:
    classes = [c.strip() for c in a.detector_classes.split(",") if c.strip()] if a.detector_classes else None
    return replace(
        s,
        full_frame=a.full_frame,
        vlm_model=a.model or s.vlm_model,
        vlm_timeout_s=a.timeout,
        recheck_s=a.recheck_s if a.recheck_s is not None else s.recheck_s,
        detector_weights=a.detector_weights or s.detector_weights,
        # new weights without classes means a fine-tuned checkpoint: drop any YOLO-World prompts
        detector_classes=classes if (classes or a.detector_weights) else s.detector_classes,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="result tag, e.g. before / after / detector_only")
    ap.add_argument("--clips", default="data/bench/clips.csv")
    ap.add_argument("--settings", default="config/settings.json")
    ap.add_argument("--model", help='served VLM id (base id, or "context" for the LoRA adapter)')
    ap.add_argument("--detector-weights", help="override settings.detector_weights")
    ap.add_argument("--detector-classes", help="comma-separated YOLO-World prompts in class-id order")
    ap.add_argument("--full-frame", action="store_true", help="send the whole frame, not the crop")
    ap.add_argument("--detector-only", action="store_true", help="no VLM: any candidate counts as alert")
    ap.add_argument("--recheck-s", type=float, default=None,
                    help="override settings.recheck_s in simulated seconds; set below clip length so growth can escalate")
    ap.add_argument("--timeout", type=float, default=30, help="per-VLM-request timeout, seconds")
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    return ap.parse_args(argv)


def main() -> None:
    a = parse_args()
    out = Path("results") / f"bench_{a.name}.json"
    check_output(out, a.force)
    s = apply_overrides(load_settings(a.settings), a)
    rows = read_clips(a.clips)
    if not rows:
        raise SystemExit(f"no clips in {a.clips}")

    from sentinel.detector import YoloDetector  # ultralytics is only needed on the Nano
    from sentinel.vlm_client import ContextVLM, ensure_served

    if a.detector_only:
        vlm = NullVLM()
    else:
        vlm = ContextVLM(s.vlm_model, s.vlm_base_url, s.vlm_timeout_s)
        ensure_served(vlm.client, s.vlm_model, s.vlm_base_url)
    detector = YoloDetector(s.detector_weights, classes=s.detector_classes)
    result = bench(a.name, rows, s, detector, vlm, a.detector_only)
    result["config"] = {"detector_weights": s.detector_weights, "detector_classes": s.detector_classes,
                        "vlm_model": None if a.detector_only else s.vlm_model,
                        "full_frame": s.full_frame, "recheck_s": s.recheck_s,
                        "vlm_timeout_s": s.vlm_timeout_s, "clips": a.clips}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "name", "precision", "recall", "false_alarms", "missed", "tokens_per_vlm_call",
        "frames_per_vlm_call", "time_to_decision_s_p50", "time_to_decision_s_p95")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Gather report artifacts (curves, plots, logs, label stats) into results/ on the Nano.
# Safe to re-run: it only copies; results/*.json from the eval scripts are left as they are.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/detector_training results/logs

# Detector fine-tuning: curves, confusion matrices, sample predictions, run config.
if [ -d runs/smoke ]; then
  cp runs/smoke/{results.csv,results.png,args.yaml} results/detector_training/ 2>/dev/null || true
  cp runs/smoke/*curve*.png runs/smoke/confusion_matrix*.png results/detector_training/ 2>/dev/null || true
  cp runs/smoke/val_batch*_{labels,pred}.jpg results/detector_training/ 2>/dev/null || true
fi

# Logs: strip carriage-return progress spam so they read cleanly.
for f in *.log; do
  [ -f "$f" ] || continue
  { tr '\r' '\n' < "$f" | grep -v -E '^\s*$' || true; } | tail -n 400 > "results/logs/$f"
done

# Teacher label distribution (what the student learns from).
if [ -f data/teacher/train.jsonl ]; then
  PY=$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
  "$PY" - <<'EOF'
import collections, json
from pathlib import Path
out = {}
for split in ("train", "heldout"):
    p = Path(f"data/teacher/{split}.jsonl")
    rows = [json.loads(l) for l in p.open()] if p.exists() else []
    out[split] = {"n": len(rows),
                  "source_type": dict(collections.Counter(r["label"]["source_type"] for r in rows).most_common())}
Path("results/teacher_labels_summary.json").write_text(json.dumps(out, indent=2))
print("teacher label summary:", {k: v["n"] for k, v in out.items()})
EOF
fi
echo "artifacts collected in results/"

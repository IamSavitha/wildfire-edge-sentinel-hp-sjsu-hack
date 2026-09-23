"""Score a context VLM against a labeled JSONL split and save results/context_<name>.json.

Rows are {"image": <crop path>, "label": <ContextResult dict>} (scripts/teacher_label.py output,
or a hand-checked gold set in the same format).

BEFORE: --model "<base id from /v1/models>" --name before_base7b
AFTER:  --model context --name after_lora7b      (vLLM serving the LoRA adapter as "context")
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sentinel.vlm_client import ContextVLM

GROUP = {"wildland": "danger", "structure": "danger", "vehicle": "danger",
         "controlled_burn": "benign", "campfire": "benign", "bbq_chimney": "benign",
         "industrial_stack": "benign", "fog_dust_cloud": "lookalike", "unknown": "unknown"}


def evaluate(rows, classify, read_bytes=lambda p: Path(p).read_bytes(), clock=time.perf_counter) -> dict:
    """Run classify(jpeg) -> (ContextResult | None, tokens) over rows and score it against row labels.

    A parse failure (None) counts as wrong for both accuracies.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("empty split")
    exact = group = fails = 0
    latencies, tokens = [], []
    per_source: dict[str, dict[str, int]] = {}
    for r in rows:
        jpeg = read_bytes(r["image"])
        t0 = clock()
        ctx, tok = classify(jpeg)
        latencies.append((clock() - t0) * 1000)
        tokens.append(tok)
        want = r["label"]["source_type"]
        bucket = per_source.setdefault(want, {"n": 0, "correct": 0})
        bucket["n"] += 1
        if ctx is None:
            fails += 1
            continue
        hit = ctx.source_type == want
        exact += hit
        bucket["correct"] += hit
        group += GROUP[ctx.source_type] == GROUP[want]
    n = len(rows)
    return {
        "n": n,
        "source_type_acc": exact / n,
        "group_acc": group / n,
        "parse_fail_rate": fails / n,
        "latency_ms_p50": float(np.percentile(latencies, 50)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "tokens_per_call": float(np.mean(tokens)),
        "per_source": per_source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help='served model id, or "context" for the LoRA adapter')
    ap.add_argument("--name", required=True, help="result tag, e.g. before_base7b / after_lora7b")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="data/teacher/heldout.jsonl")
    ap.add_argument("--timeout", type=float, default=60, help="per-request timeout, seconds")
    a = ap.parse_args()

    vlm = ContextVLM(a.model, a.base_url, timeout_s=a.timeout)
    with open(a.split) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = {"name": a.name, "model": a.model, "split": a.split, **evaluate(rows, vlm.classify)}

    out = Path("results") / f"context_{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

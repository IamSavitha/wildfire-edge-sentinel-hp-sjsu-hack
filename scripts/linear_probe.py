"""Cheap baseline: SigLIP image embeddings + logistic regression on teacher source_type labels.

Writes results/context_linear_probe.json with the same accuracy keys as scripts/eval_context.py.
Fast, but it only predicts source_type: no attendance, structures, road or description.
"""
import argparse
import json
import time
from pathlib import Path

try:
    from scripts.eval_context import GROUP, check_output
except ModuleNotFoundError as e:  # run as `python scripts/linear_probe.py`: scripts/ is on sys.path, not the repo root
    if e.name != "scripts":
        raise
    from eval_context import GROUP, check_output


def group_accuracy(y_true, y_pred) -> float:
    """Fraction of predictions in the same danger/benign/lookalike/unknown group as the label."""
    y_true, y_pred = list(y_true), list(y_pred)
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("need equal-length, non-empty label lists")
    return sum(GROUP[t] == GROUP[p] for t, p in zip(y_true, y_pred)) / len(y_true)


def load(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r["image"] for r in rows], [r["label"]["source_type"] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/teacher/train.jsonl")
    ap.add_argument("--heldout", default="data/teacher/heldout.jsonl")
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--device", default="cuda", help='"cuda" or "cpu"')
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args()
    path = Path("results") / "context_linear_probe.json"
    check_output(path, a.force)

    # lazy: --help and unit tests work without torch/transformers/sklearn
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(a.model).to(a.device).eval()
    proc = AutoProcessor.from_pretrained(a.model)

    def embed(paths):
        feats = []
        for i in range(0, len(paths), 64):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
            with torch.no_grad():
                f = model.get_image_features(**proc(images=imgs, return_tensors="pt").to(a.device))
            f = getattr(f, "pooler_output", f)  # newer transformers return a model output, not a tensor
            feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
        return np.concatenate(feats)

    xtr, ytr = load(a.train)
    xte, yte = load(a.heldout)
    if not xte:
        raise ValueError("empty split")
    clf = LogisticRegression(max_iter=2000).fit(embed(xtr), ytr)
    t0 = time.perf_counter()
    pred = list(clf.predict(embed(xte)))
    ms = (time.perf_counter() - t0) * 1000 / len(xte)

    out = {
        "name": "siglip_linear_probe",
        "model": a.model,
        "split": a.heldout,
        "n": len(yte),
        "source_type_acc": sum(t == p for t, p in zip(yte, pred)) / len(yte),
        "group_acc": group_accuracy(yte, pred),
        "latency_ms_per_image": ms,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

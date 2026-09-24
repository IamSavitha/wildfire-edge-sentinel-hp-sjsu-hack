"""Rebalance teacher-labelled training rows by source_type (deterministic) for a second LoRA run.

Classes with at least `min_class` rows are resampled to `target` rows (upsampled with repeats or
downsampled), capped at `cap`; rarer classes are repeated `rare_mult` times. Held-out rows are never used.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def balance(rows: list[dict], target: int = 600, cap: int = 700, min_class: int = 100,
            rare_mult: int = 5, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["label"]["source_type"]].append(r)
    out: list[dict] = []
    for cls in sorted(by_class):
        items = by_class[cls]
        if len(items) < min_class:
            out.extend(items * rare_mult)
            continue
        want = cap if len(items) > cap else min(target, cap)
        if len(items) >= want:
            out.extend(rng.sample(items, want))
        else:
            out.extend(items)
            out.extend(rng.choice(items) for _ in range(want - len(items)))
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/teacher/train.jsonl")
    ap.add_argument("--out", default="data/teacher/train_balanced.jsonl")
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--cap", type=int, default=700)
    ap.add_argument("--min-class", type=int, default=100)
    ap.add_argument("--rare-mult", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.src)]
    out = balance(rows, a.target, a.cap, a.min_class, a.rare_mult, a.seed)
    Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in out))
    counts: dict[str, int] = defaultdict(int)
    for r in out:
        counts[r["label"]["source_type"]] += 1
    print(f"{len(rows)} -> {len(out)} rows: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")


if __name__ == "__main__":
    main()

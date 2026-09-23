"""Unpack PyroNear pyro-sdis (parquet) into the YOLO layout, and sample benign tower images.

Source: data/pyro/data/{train,val}-*.parquet (columns image, annotations, image_name, ...).
pyro-sdis labels smoke as class id 1; we remap it to our smoke id 0.
Output: data/pyro_yolo/{train,val}/{images,labels}/ and, with --benign N, N smoke-free
lookout-tower images copied into data/benign/ (real look-alikes: sky, cloud, haze).
"""
import argparse
import glob
import random
from pathlib import Path

PYRO_SMOKE_ID = "1"
OUR_SMOKE_ID = "0"


def remap_labels(annotations: str) -> str:
    """Rewrite pyro-sdis YOLO lines to our class ids (smoke 1 -> 0)."""
    out = []
    for line in annotations.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        if parts[0] != PYRO_SMOKE_ID:
            raise ValueError(f"unexpected pyro-sdis class id {parts[0]!r}")
        out.append(" ".join([OUR_SMOKE_ID, *parts[1:]]))
    return "\n".join(out) + ("\n" if out else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/pyro/data")
    ap.add_argument("--out", default="data/pyro_yolo")
    ap.add_argument("--benign", type=int, default=300, help="smoke-free images to copy into data/benign")
    ap.add_argument("--benign-dir", default="data/benign")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pyarrow.parquet as pq

    negatives = []
    for split in ("train", "val"):
        files = sorted(glob.glob(f"{a.src}/{split}-*.parquet"))
        if not files:
            raise SystemExit(f"no {split} parquet files in {a.src}")
        img_dir, lbl_dir = Path(a.out, split, "images"), Path(a.out, split, "labels")
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    name = Path(row["image_name"]).name
                    (img_dir / name).write_bytes(row["image"]["bytes"])
                    labels = remap_labels(row["annotations"] or "")
                    (lbl_dir / f"{Path(name).stem}.txt").write_text(labels)
                    if not labels:
                        negatives.append(img_dir / name)
                    n += 1
        print(f"{split}: {n} images -> {a.out}/{split}")

    random.Random(a.seed).shuffle(negatives)
    Path(a.benign_dir).mkdir(parents=True, exist_ok=True)
    for p in negatives[: a.benign]:
        (Path(a.benign_dir) / f"pyro_{p.name}").write_bytes(p.read_bytes())
    print(f"benign: {min(a.benign, len(negatives))} of {len(negatives)} smoke-free images -> {a.benign_dir}")


if __name__ == "__main__":
    main()

"""Unpack the D-Fire Hugging Face mirror (parquet) into the YOLO folder layout.

Source: data/raw/dfire_hf/data/{train,test}-*.parquet with columns image, label, filename
(label = YOLO lines "cls cx cy w h", empty for images with no fire/smoke; 0=smoke, 1=fire).
Output: data/dfire/{train,test}/{images,labels}/<stem>.{jpg,txt}
"""
import argparse
import glob
from pathlib import Path


def write_row(row: dict, split_dir: Path) -> None:
    """Write one parquet row as an image file plus its YOLO label file."""
    name = Path(row["filename"]).name
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    (split_dir / "images" / name).write_bytes(row["image"]["bytes"])
    label = row["label"].strip()
    (split_dir / "labels" / f"{Path(name).stem}.txt").write_text(label + "\n" if label else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default="data/raw/dfire_hf/data")
    ap.add_argument("--out", default="data/dfire")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    for split in ("train", "test"):
        files = sorted(glob.glob(f"{a.src}/{split}-*.parquet"))
        if not files:
            raise SystemExit(f"no {split} parquet files in {a.src}")
        n = 0
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    write_row(row, Path(a.out) / split)
                    n += 1
        print(f"{split}: {n} images -> {a.out}/{split}")


if __name__ == "__main__":
    main()

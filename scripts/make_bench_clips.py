"""Build labelled end-to-end benchmark clips from real tower imagery.

alert:    FIgLib ignition sequences (data/demo/fire_*) and pyro-sdis camera sequences with smoke
no_alert: pyro-sdis camera sequences with no smoke in any frame
Each clip is a folder of time-ordered image symlinks under data/bench/<clip>/; writes data/bench/clips.csv.
pyro-sdis names look like <partner>_<camera>_<YYYY-MM-DDTHH-MM-SS>.jpg.
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def camera_of(name: str) -> str:
    """'force-06_courmettes-160_2024-01-08T12-44-06.jpg' -> 'force-06_courmettes-160'."""
    return Path(name).stem.rsplit("_", 1)[0]


def build_sequences(images_dir: Path, labels_dir: Path, min_frames: int) -> dict[str, dict]:
    """Group frames per camera (sorted by timestamp) and mark whether any frame has smoke."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(images_dir.glob("*.jpg")):
        groups[camera_of(p.name)].append(p)
    seqs = {}
    for cam, frames in groups.items():
        if len(frames) < min_frames:
            continue
        smoke = any((labels_dir / f"{f.stem}.txt").read_text().strip() for f in frames
                    if (labels_dir / f"{f.stem}.txt").exists())
        seqs[cam] = {"frames": frames, "smoke": smoke}
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pyro", default="data/pyro_yolo/val")
    ap.add_argument("--figlib", default="data/demo")
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--per-label", type=int, default=10, help="pyro clips per label")
    ap.add_argument("--min-frames", type=int, default=12)
    ap.add_argument("--max-frames", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def link_clip(name: str, frames: list[Path], label: str) -> None:
        d = out / name
        d.mkdir(exist_ok=True)
        for i, f in enumerate(frames[: a.max_frames]):
            dst = d / f"{i:04d}{f.suffix.lower()}"
            if not dst.exists():
                dst.symlink_to(f.resolve())
        rows.append({"path": str(d), "label": label})

    for seq in sorted(Path(a.figlib).glob("fire_*")):
        frames = sorted(p for p in seq.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        link_clip(f"figlib_{seq.name}", frames, "alert")

    seqs = build_sequences(Path(a.pyro) / "images", Path(a.pyro) / "labels", a.min_frames)
    rng = random.Random(a.seed)
    for label, want_smoke in (("alert", True), ("no_alert", False)):
        cams = sorted(c for c, s in seqs.items() if s["smoke"] == want_smoke)
        rng.shuffle(cams)
        for cam in cams[: a.per_label]:
            link_clip(f"pyro_{label}_{cam}", seqs[cam]["frames"], label)

    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(rows)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in ("alert", "no_alert")}
    print(f"{len(rows)} clips -> {out}/clips.csv {counts}")


if __name__ == "__main__":
    main()

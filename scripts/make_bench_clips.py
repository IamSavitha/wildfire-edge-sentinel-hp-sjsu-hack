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


def _has_smoke(labels_dir: Path, frame: Path) -> bool:
    lbl = labels_dir / f"{frame.stem}.txt"
    return lbl.exists() and bool(lbl.read_text().strip())


def build_sequences(images_dir: Path, labels_dir: Path, min_frames: int) -> dict[str, dict]:
    """Group frames per camera (sorted by timestamp); keep per-frame smoke flags."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(images_dir.glob("*.jpg")):
        groups[camera_of(p.name)].append(p)
    seqs = {}
    for cam, frames in groups.items():
        if len(frames) < min_frames:
            continue
        flags = [_has_smoke(labels_dir, f) for f in frames]
        seqs[cam] = {"frames": frames, "flags": flags, "smoke": any(flags)}
    return seqs


def smoke_window(frames: list[Path], flags: list[bool], max_frames: int, lead: int = 10) -> list[Path]:
    """A window that starts `lead` frames before the first smoky frame."""
    first = flags.index(True)
    start = max(0, first - lead)
    return frames[start:start + max_frames]


def clean_frames(frames: list[Path], flags: list[bool]) -> list[Path]:
    """Only the smoke-free frames of a camera, in time order (benign footage)."""
    return [f for f, s in zip(frames, flags) if not s]


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
        # FIgLib names carry the offset from ignition (…_-02340.jpg … _+02400.jpg): keep ignition onward
        post = [p for p in frames if "_+" in p.stem] or frames
        link_clip(f"figlib_{seq.name}", post, "alert")

    seqs = build_sequences(Path(a.pyro) / "images", Path(a.pyro) / "labels", a.min_frames)
    rng = random.Random(a.seed)
    smoky = sorted(c for c, s in seqs.items() if s["smoke"])
    rng.shuffle(smoky)
    for cam in smoky[: a.per_label]:
        s = seqs[cam]
        link_clip(f"pyro_alert_{cam}", smoke_window(s["frames"], s["flags"], a.max_frames), "alert")
    clean = sorted(c for c, s in seqs.items() if len(clean_frames(s["frames"], s["flags"])) >= a.min_frames)
    rng.shuffle(clean)
    for cam in clean[: a.per_label]:
        s = seqs[cam]
        link_clip(f"pyro_clear_{cam}", clean_frames(s["frames"], s["flags"]), "no_alert")

    with open(out / "clips.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(rows)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in ("alert", "no_alert")}
    print(f"{len(rows)} clips -> {out}/clips.csv {counts}")


if __name__ == "__main__":
    main()

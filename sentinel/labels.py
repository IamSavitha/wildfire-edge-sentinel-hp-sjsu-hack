"""YOLO label helpers shared by training scripts."""
from pathlib import Path


def yolo_boxes(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def write_dfire_yaml(root: str | Path, out: str | Path) -> Path:
    """Ultralytics data YAML for D-Fire (0=smoke, 1=fire), evaluated on its test split."""
    root = Path(root).resolve()  # Ultralytics resolves relative paths against its own datasets dir
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"path: {root}\ntrain: train/images\nval: test/images\nnames:\n  0: smoke\n  1: fire\n")
    return out

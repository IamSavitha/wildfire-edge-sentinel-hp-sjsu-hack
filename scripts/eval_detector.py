"""Evaluate a detector on the D-Fire test split and save results/detector_<name>.json. Run on the Nano.

BEFORE: --weights yolov8s-worldv2.pt --classes smoke,fire --name before_yoloworld  (zero-shot)
AFTER:  --weights models/smoke_yolo.pt --name after_yolo11s                        (fine-tuned)
"""
import argparse
import json
from pathlib import Path

from sentinel.labels import write_dfire_yaml

try:
    from scripts.eval_context import check_output
except ModuleNotFoundError as e:  # run as `python scripts/eval_detector.py`: scripts/ is on sys.path, not the repo root
    if e.name != "scripts":
        raise
    from eval_context import check_output

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def summarize(name: str, weights: str, classes: list[str] | None, box, speed: dict, names,
              n_images: int | None = None) -> dict:
    """Turn Ultralytics val metrics (metrics.box, metrics.speed, class names) into a JSON-able dict."""
    ap50 = [float(v) for v in box.ap50]
    # ap50 has one entry per class present in the split, ordered by ap_class_index
    idx = [int(i) for i in getattr(box, "ap_class_index", range(len(ap50)))]
    out = {
        "name": name,
        "weights": weights,
        "classes": classes,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class_map50": {names[i]: v for i, v in zip(idx, ap50)},
        "ms_per_image": float(speed["inference"]),
    }
    if n_images is not None:
        out["n_images"] = n_images
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True, help="result tag, e.g. before_yoloworld / after_yolo11s")
    ap.add_argument("--classes", default=None,
                    help="comma-separated text prompts for YOLO-World, in class-id order (e.g. smoke,fire)")
    ap.add_argument("--root", default="data/dfire")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help='CUDA device index, or "cpu"')
    ap.add_argument("--force", action="store_true", help="overwrite an existing results file")
    a = ap.parse_args()
    out = Path("results") / f"detector_{a.name}.json"
    check_output(out, a.force)
    classes = [c.strip() for c in a.classes.split(",")] if a.classes else None

    from ultralytics import YOLO  # lazy: --help and tests work without torch

    data_yaml = write_dfire_yaml(a.root, "runs/dfire.yaml")
    model = YOLO(a.weights)
    if classes:
        model.set_classes(classes)  # must match data yaml ids: 0=smoke, 1=fire
    m = model.val(data=str(data_yaml), imgsz=a.imgsz, device=a.device, split="val", plots=False)

    test_images = Path(a.root) / "test" / "images"
    n_images = sum(1 for p in test_images.iterdir() if p.suffix.lower() in IMAGE_EXTS) if test_images.is_dir() else None
    result = summarize(a.name, a.weights, classes, m.box, m.speed, m.names, n_images)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

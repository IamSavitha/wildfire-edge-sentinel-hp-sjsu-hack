"""Fine-tune YOLO on D-Fire (0=smoke, 1=fire). Run on the Nano."""
import argparse
import shutil
from pathlib import Path

from sentinel.labels import write_dfire_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/dfire")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    from ultralytics import YOLO  # lazy: --help works without torch

    data_yaml = write_dfire_yaml(a.root, "runs/dfire.yaml")
    model = YOLO(a.model)
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=0, project="runs", name="smoke", exist_ok=True)
    metrics = model.val(data=str(data_yaml), imgsz=a.imgsz, device=0, split="val")
    print(f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")

    Path("models").mkdir(exist_ok=True)
    shutil.copy(model.trainer.best, "models/smoke_yolo.pt")  # runs/smoke/weights/best.pt
    print("saved models/smoke_yolo.pt")


if __name__ == "__main__":
    main()

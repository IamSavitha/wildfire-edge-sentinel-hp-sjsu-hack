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
    ap.add_argument("--device", default="0", help='CUDA device index, or "cpu"')
    ap.add_argument("--name", default="smoke", help="run name: runs/<name>/ and runs/<name>.yaml")
    ap.add_argument("--out", default="models/smoke_yolo.pt", help="where to copy the best weights")
    a = ap.parse_args()

    from ultralytics import YOLO  # lazy: --help works without torch

    data_yaml = write_dfire_yaml(a.root, f"runs/{a.name}.yaml")
    model = YOLO(a.model)
    # absolute project dir: a relative one gets nested under runs/detect/
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=str(Path("runs").resolve()), name=a.name, exist_ok=True)
    metrics = model.val(data=str(data_yaml), imgsz=a.imgsz, device=a.device, split="val")
    print(f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(model.trainer.best, a.out)  # runs/<name>/weights/best.pt under the cwd
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()

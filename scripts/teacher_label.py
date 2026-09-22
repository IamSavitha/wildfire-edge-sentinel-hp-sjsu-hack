"""Crop D-Fire + benign images and label them with the local teacher VLM (concurrent)."""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from sentinel.imaging import crop_box, to_jpeg
from sentinel.labels import yolo_boxes
from sentinel.vlm_client import ContextVLM


def make_crops(images_dir: Path, labels_dir: Path, out_dir: Path, n: int, seed: int = 0) -> list[Path]:
    imgs = sorted(images_dir.glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    out = []
    for p in imgs[:n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = yolo_boxes(labels_dir / f"{p.stem}.txt", w, h)
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else (0, 0, w, h)
        crop = crop_box(img, box, pad=0.5 if boxes else 0.0, max_side=448)
        dst = out_dir / f"{images_dir.parent.name}_{p.stem}.jpg"
        dst.write_bytes(to_jpeg(crop, 90))
        out.append(dst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/dfire/train/images")
    ap.add_argument("--labels", default="data/dfire/train/labels")
    ap.add_argument("--benign", default="data/benign")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", required=True, help="teacher id from :8001/v1/models")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="data/teacher")
    a = ap.parse_args()

    out = Path(a.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops = make_crops(Path(a.images), Path(a.labels), crops_dir, a.n)
    crops += make_crops(Path(a.benign), Path(a.benign) / "_no_labels", crops_dir, 10_000)
    print(f"{len(crops)} crops to label")

    teacher = ContextVLM(a.model, a.base_url, timeout_s=180, max_tokens=200)

    def label(p: Path):
        return p, teacher.classify(p.read_bytes())[0]

    with ThreadPoolExecutor(a.workers) as ex, \
            open(out / "train.jsonl", "w") as ftr, open(out / "heldout.jsonl", "w") as fho:
        for i, (p, ctx) in enumerate(ex.map(label, crops)):
            if ctx is None:
                continue
            held = int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 100 < 15
            (fho if held else ftr).write(json.dumps({"image": str(p), "label": ctx.model_dump()}) + "\n")
            if i % 100 == 0:
                ftr.flush(); fho.flush()
                print(f"{i}/{len(crops)}", flush=True)


if __name__ == "__main__":
    main()

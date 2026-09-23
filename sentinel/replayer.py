"""Replays a video file or a folder of time-ordered images as a tower camera."""
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def frames(source: str, fps: float, loop: bool = True) -> Iterator[np.ndarray]:
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"no images in {source}")
        while True:
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    yield img
            if not loop:
                return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {source}")
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            if i == 0:
                raise ValueError(f"no readable frames in {source}")
            if not loop:
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if i % step == 0:
            yield img
        i += 1

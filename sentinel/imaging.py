"""Image helpers: context-preserving crops and JPEG encoding."""
import cv2
import numpy as np


def crop_box(frame: np.ndarray, box: tuple[float, float, float, float],
             pad: float = 0.5, max_side: int = 448) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = min(w - 1, max(0, int(x1 - pad * bw))), min(h - 1, max(0, int(y1 - pad * bh)))
    x2, y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
    x2, y2 = max(x2, min(w, x1 + 1)), max(y2, min(h, y1 + 1))
    crop = frame[y1:y2, x1:x2]
    scale = max_side / max(crop.shape[:2])
    if scale < 1:
        size = (int(crop.shape[1] * scale), int(crop.shape[0] * scale))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return crop


def to_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()

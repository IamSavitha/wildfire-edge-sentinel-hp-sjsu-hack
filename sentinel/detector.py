"""Stage 1: YOLO smoke/fire detector."""
import numpy as np

from sentinel.schema import Detection


class YoloDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # imported lazily so laptop tests don't need torch
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [Detection(r.names[int(c)], float(p), tuple(b))
                for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

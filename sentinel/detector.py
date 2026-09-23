"""Stage 1: YOLO smoke/fire detector (fine-tuned YOLO, or YOLO-World zero-shot with text classes)."""
import numpy as np

from sentinel.schema import Detection


class YoloDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640,
                 classes: list[str] | None = None, model=None):
        if model is None:
            from ultralytics import YOLO  # imported lazily so laptop tests don't need torch
            model = YOLO(weights)
        self.model = model
        if classes:
            self.model.set_classes(classes)  # YOLO-World: prompt order defines class ids
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [Detection(r.names[int(c)], float(p), tuple(b))
                for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

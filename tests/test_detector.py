from types import SimpleNamespace

import numpy as np
import pytest

from sentinel.detector import YoloDetector
from sentinel.schema import Detection


class _Seq:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.classes, self.predict_kwargs = None, None

    def set_classes(self, classes):
        self.classes = classes

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        boxes = SimpleNamespace(xyxy=_Seq([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
                                conf=_Seq([0.9, 0.5]), cls=_Seq([0.0, 1.0]))
        return [SimpleNamespace(names={0: "smoke", 1: "fire"}, boxes=boxes)]


def test_detector_builds_detections_from_result():
    model = FakeModel()
    det = YoloDetector("unused.pt", conf=0.3, imgsz=320, model=model)
    out = det(np.zeros((10, 10, 3), np.uint8))
    assert out == [Detection("smoke", 0.9, (1.0, 2.0, 3.0, 4.0)), Detection("fire", 0.5, (5.0, 6.0, 7.0, 8.0))]
    assert model.predict_kwargs == {"conf": 0.3, "imgsz": 320, "verbose": False}


def test_set_classes_only_when_given():
    plain = FakeModel()
    YoloDetector("unused.pt", model=plain)
    assert plain.classes is None
    world = FakeModel()
    YoloDetector("yolov8s-worldv2.pt", classes=["smoke", "fire"], model=world)
    assert world.classes == ["smoke", "fire"]


def test_classes_require_yolo_world_model():
    class PlainModel:
        def predict(self, frame, **kwargs):
            return []

    with pytest.raises(ValueError, match="YOLO-World"):
        YoloDetector("models/smoke_yolo.pt", classes=["smoke", "fire"], model=PlainModel())


def test_imgsz_for_matches_the_training_resolution():
    from sentinel.detector import imgsz_for
    from sentinel import webapp
    assert imgsz_for("models/joint_yolo.pt") == imgsz_for("/home/u/sentinel/models/tower_yolo.pt") == 960
    assert imgsz_for("models/smoke_yolo.pt") == imgsz_for("yolov8s-worldv2.pt") == 640
    assert webapp.imgsz_for is imgsz_for          # one definition, shared by the webapp and the incident map

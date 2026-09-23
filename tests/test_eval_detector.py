from types import SimpleNamespace

import pytest

from scripts.eval_detector import summarize


def _box(**kw):
    base = dict(map50=0.61, map=0.33, mp=0.7, mr=0.55, ap50=[0.5, 0.72])
    base.update(kw)
    return SimpleNamespace(**base)


def test_summarize_builds_result_dict():
    out = summarize("after_yolo11s", "models/smoke_yolo.pt", None, _box(),
                    {"preprocess": 1.0, "inference": 12.5, "postprocess": 0.8}, {0: "smoke", 1: "fire"})
    assert out == {
        "name": "after_yolo11s", "weights": "models/smoke_yolo.pt", "classes": None,
        "map50": 0.61, "map50_95": 0.33, "precision": 0.7, "recall": 0.55,
        "per_class_map50": {"smoke": 0.5, "fire": 0.72}, "ms_per_image": 12.5,
    }


def test_summarize_uses_ap_class_index_and_n_images():
    # Ultralytics reports ap50 only for classes present, in ap_class_index order
    box = _box(ap50=[0.4], ap_class_index=[1])
    out = summarize("before_yoloworld", "yolov8s-worldv2.pt", ["smoke", "fire"], box,
                    {"inference": 30.0}, ["smoke", "fire"], n_images=4306)
    assert out["classes"] == ["smoke", "fire"]
    assert out["per_class_map50"] == {"fire": pytest.approx(0.4)}
    assert out["n_images"] == 4306

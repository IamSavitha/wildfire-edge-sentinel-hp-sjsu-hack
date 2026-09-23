import json

import pytest

from sentinel.config import Settings, load_settings, load_towers


def test_missing_settings_file_gives_defaults(tmp_path):
    assert load_settings(tmp_path / "none.json") == Settings()


def test_settings_override(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"min_frames": 5, "vlm_model": "context"}))
    s = load_settings(p)
    assert s.min_frames == 5 and s.vlm_model == "context"


def test_unknown_setting_is_rejected(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"min_frame": 5}))
    with pytest.raises(ValueError, match="min_frame"):
        load_settings(p)


def test_load_towers(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"id": "t1", "name": "A", "lat": 1.0, "lon": 2.0, "source": "x"}]))
    towers = load_towers(p)
    assert towers["t1"].name == "A" and towers["t1"].benign_zones == []


def test_detector_classes_setting(tmp_path):
    assert Settings().detector_classes is None
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"detector_weights": "yolov8s-worldv2.pt", "detector_classes": ["smoke", "fire"]}))
    assert load_settings(p).detector_classes == ["smoke", "fire"]

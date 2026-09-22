# tests/test_teacher_label.py
from sentinel.labels import yolo_boxes


def test_yolo_boxes_to_pixels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n\n")
    assert yolo_boxes(p, w=100, h=50) == [(40.0, 15.0, 60.0, 35.0)]


def test_missing_label_file_means_no_boxes(tmp_path):
    assert yolo_boxes(tmp_path / "none.txt", 100, 50) == []

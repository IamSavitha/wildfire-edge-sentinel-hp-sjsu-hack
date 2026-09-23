# tests/test_teacher_label.py
from sentinel.labels import yolo_boxes


def test_yolo_boxes_to_pixels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n\n")
    assert yolo_boxes(p, w=100, h=50) == [(40.0, 15.0, 60.0, 35.0)]


def test_missing_label_file_means_no_boxes(tmp_path):
    assert yolo_boxes(tmp_path / "none.txt", 100, 50) == []


def test_make_crops_accepts_jpg_jpeg_png(tmp_path):
    import cv2
    import numpy as np

    from scripts.teacher_label import make_crops
    src, out = tmp_path / "benign", tmp_path / "crops"
    src.mkdir(); out.mkdir()
    img = np.full((40, 60, 3), 128, np.uint8)
    for name in ("a.jpg", "b.jpeg", "c.png", "d.PNG", "notes.txt"):
        if name.endswith("txt"):
            (src / name).write_text("x")
        else:
            cv2.imwrite(str(src / name), img)
    crops = make_crops(src, src / "_no_labels", out, 100)
    assert sorted(p.name.rsplit("_", 1)[1] for p in crops) == ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    assert all(p.exists() for p in crops)


def test_write_labels_counts_rows_and_skips_failures(tmp_path):
    import io
    from pathlib import Path

    from scripts.teacher_label import write_labels
    from sentinel.schema import ContextResult
    ctx = ContextResult(source_type="campfire", smoke_color="white", attended="yes",
                        near_structures=False, near_road=False, size_estimate="small",
                        description="Campfire.")
    ftr, fho = io.StringIO(), io.StringIO()
    pairs = [(Path(f"x{i}.jpg"), ctx if i % 2 else None) for i in range(10)]
    assert write_labels(pairs, ftr, fho, total=10) == 5
    assert len((ftr.getvalue() + fho.getvalue()).splitlines()) == 5
    assert write_labels([(Path("y.jpg"), None)], ftr, fho, total=1) == 0

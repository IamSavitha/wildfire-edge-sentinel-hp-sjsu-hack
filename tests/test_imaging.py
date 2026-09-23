import numpy as np

from sentinel.imaging import crop_box, to_jpeg

FRAME = np.zeros((480, 640, 3), np.uint8)


def test_crop_adds_padding():
    assert crop_box(FRAME, (100, 100, 200, 200), pad=0.5, max_side=448).shape == (200, 200, 3)


def test_crop_downscales_to_max_side():
    assert crop_box(FRAME, (0, 0, 640, 480), pad=0.0, max_side=448).shape == (336, 448, 3)


def test_crop_clips_to_frame():
    assert crop_box(FRAME, (600, 440, 640, 480), pad=0.5, max_side=448).shape == (60, 60, 3)


def test_to_jpeg_returns_jpeg_bytes():
    assert to_jpeg(FRAME)[:2] == b"\xff\xd8"


def test_box_on_right_edge_is_never_empty():
    crop = crop_box(FRAME, (640, 0, 640, 10), pad=0.0)
    assert crop.size > 0
    assert to_jpeg(crop)[:2] == b"\xff\xd8"


def test_box_on_bottom_edge_is_never_empty():
    crop = crop_box(FRAME, (0, 480, 10, 480), pad=0.0)
    assert crop.size > 0
    assert to_jpeg(crop)[:2] == b"\xff\xd8"


def test_thin_tall_crop_survives_downscale():
    assert crop_box(FRAME, (100, 0, 100.5, 480), pad=0.5, max_side=256).size > 0

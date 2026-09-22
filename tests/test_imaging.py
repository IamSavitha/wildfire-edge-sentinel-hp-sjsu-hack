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

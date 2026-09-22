import itertools

import cv2
import numpy as np

from sentinel.replayer import frames


def test_image_dir_loops(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.full((10, 10, 3), i * 50, np.uint8))
    got = list(itertools.islice(frames(str(tmp_path), fps=2), 4))
    assert len(got) == 4 and abs(int(got[3][0, 0, 0]) - 0) < 5


def test_image_dir_no_loop(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.zeros((10, 10, 3), np.uint8))
    assert len(list(frames(str(tmp_path), fps=2, loop=False))) == 3

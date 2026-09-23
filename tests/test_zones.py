from sentinel.zones import in_any_zone

SQUARE = [[0, 0], [50, 0], [50, 50], [0, 50]]


def test_box_center_inside_zone():
    assert in_any_zone((10, 10, 30, 30), [SQUARE])


def test_box_center_outside_zone():
    assert not in_any_zone((100, 100, 120, 120), [SQUARE])


def test_no_zones():
    assert not in_any_zone((10, 10, 30, 30), [])

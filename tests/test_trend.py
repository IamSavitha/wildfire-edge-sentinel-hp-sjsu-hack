from sentinel.trend import classify_trend


def test_growing():
    assert classify_trend(100, 140) == "growing"


def test_static():
    assert classify_trend(100, 110) == "static"


def test_dissipating():
    assert classify_trend(100, 50) == "dissipating"
    assert classify_trend(100, 0) == "dissipating"


def test_from_zero_baseline():
    assert classify_trend(0, 50) == "growing"
    assert classify_trend(0, 0) == "static"

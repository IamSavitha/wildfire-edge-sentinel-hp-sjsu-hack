from sentinel.metrics import Metrics


def test_counters_and_percentiles():
    m = Metrics()
    m.inc("frames")
    m.inc("frames", 2)
    for v in range(1, 101):
        m.time("detect_ms", v)
    s = m.summary()
    assert s["frames"] == 3 and s["detect_ms_p50"] == 50.5 and s["detect_ms_p95"] > 95


def test_merge():
    a, b = Metrics(), Metrics()
    a.inc("x"); b.inc("x", 2); b.time("t", 1.0)
    a.merge(b)
    assert a.counters["x"] == 3 and a.timings["t"] == [1.0]

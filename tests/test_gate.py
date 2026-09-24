from sentinel.gate import PersistenceGate
from sentinel.schema import Detection

D = Detection("smoke", 0.8, (0, 0, 10, 10))
LOW = Detection("smoke", 0.2, (0, 0, 10, 10))


def gate():
    return PersistenceGate(min_conf=0.4, min_frames=3, cooldown_s=60)


def test_fires_after_min_frames():
    g = gate()
    assert g.update("t1", [D], 0) is None
    assert g.update("t1", [D], 1) is None
    assert g.update("t1", [D], 2) == D


def test_low_confidence_resets_streak():
    g = gate()
    g.update("t1", [D], 0)
    g.update("t1", [D], 1)
    assert g.update("t1", [LOW], 2) is None
    assert g.update("t1", [D], 3) is None
    assert g.update("t1", [D], 4) is None
    assert g.update("t1", [D], 5) == D


def test_cooldown_blocks_refire():
    g = gate()
    for t in (0, 1, 2):
        g.update("t1", [D], t)
    for t in (3, 4, 5):
        assert g.update("t1", [D], t) is None
    assert g.update("t1", [D], 62) == D


def test_towers_are_independent():
    g = gate()
    g.update("t1", [D], 0)
    g.update("t1", [D], 1)
    assert g.update("t2", [D], 2) is None


def test_picks_highest_confidence():
    g = PersistenceGate(min_conf=0.4, min_frames=1, cooldown_s=60)
    best = Detection("fire", 0.95, (5, 5, 9, 9))
    assert g.update("t1", [D, best], 0) == best


def test_streak_reports_progress_and_reset_clears_it():
    g = gate()
    assert g.streak("t1") == 0
    g.update("t1", [D], 0)
    g.update("t1", [D], 1)
    assert g.streak("t1") == 2 and g.streak("t2") == 0
    g.reset_streak("t1")
    assert g.streak("t1") == 0
    assert g.update("t1", [D], 2) is None

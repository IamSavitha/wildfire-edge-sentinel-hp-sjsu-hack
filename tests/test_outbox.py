from sentinel.outbox import Outbox


def test_enqueue_and_due():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {"a": 1}, now=0)
    assert ob.due(now=0) == [("e1", {"a": 1}, 0)]


def test_enqueue_is_idempotent():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {"a": 1}, now=0)
    ob.enqueue("e1", {"a": 2}, now=0)
    assert ob.pending_count() == 1


def test_mark_sent_removes_from_due():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {}, now=0)
    ob.mark_sent("e1", now=1)
    assert ob.due(now=5) == [] and ob.sent_count() == 1


def test_failure_backs_off():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {}, now=0)
    ob.mark_failed("e1", now=0)
    assert ob.due(now=1) == []
    assert ob.due(now=2)[0][2] == 1


def test_survives_restart(tmp_path):
    path = str(tmp_path / "o.db")
    Outbox(path).enqueue("e1", {"x": 1}, now=0)
    assert Outbox(path).pending_count() == 1

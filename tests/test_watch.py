from pathlib import Path

from sentinel.watch import list_frames, summarize_batch, update_text


def test_list_frames_orders_figlib_by_time_and_keeps_others_last(tmp_path):
    for n in ("1700000120_+00060.jpg", "1700000000_-00060.jpg", "1700000060_+00000.jpg", "zzz.jpg", "notes.txt"):
        (tmp_path / n).write_bytes(b"x")
    fr = list_frames(tmp_path)
    assert [f.offset_s for f in fr] == [-60, 0, 60, None]
    assert fr[0].epoch == 1700000000 and fr[-1].path == Path(tmp_path / "zzz.jpg")


def pts(*areas, conf=0.8):
    return [{"conf": conf if a else 0.1, "area": a} for a in areas]


def test_summarize_batch_trends():
    grow = summarize_batch(pts(0.01, 0.01), pts(0.012, 0.02, 0.0), 0.4)
    assert grow["trend"] == "growing" and grow["detected_in"] == 2 and grow["area_ratio"] == 2.0
    assert summarize_batch(pts(0.02), pts(0.021), 0.4)["trend"] == "static"
    assert summarize_batch(pts(0.02), pts(0.01), 0.4)["trend"] == "dissipating"
    gone = summarize_batch(pts(0.02), pts(0, 0, 0), 0.4)
    assert gone["trend"] == "not_seen" and gone["detected_in"] == 0 and gone["max_conf"] == 0.1
    assert summarize_batch([], pts(0.01), 0.4)["trend"] == "growing"


def test_update_text_uses_measured_values():
    u = {**summarize_batch(pts(0.01), pts(0.02, 0.015), 0.4), "source_type": "wildland", "severity": "ALERT"}
    assert update_text(u) == "smoke in 2/2 frames · plume area ×2.00 · growing · wildland · ALERT"
    assert update_text(summarize_batch(pts(0.01), pts(0, 0), 0.4)) == "smoke in 0/2 frames · not visible this batch"

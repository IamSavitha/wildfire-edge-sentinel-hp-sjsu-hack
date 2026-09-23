from scripts.prepare_dfire import write_row


def test_writes_image_and_label(tmp_path):
    row = {"filename": "WEB001.jpg", "image": {"bytes": b"\xff\xd8jpg"}, "label": "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n"}
    write_row(row, tmp_path / "train")
    assert (tmp_path / "train/images/WEB001.jpg").read_bytes() == b"\xff\xd8jpg"
    assert (tmp_path / "train/labels/WEB001.txt").read_text() == "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n"


def test_negative_image_gets_empty_label(tmp_path):
    write_row({"filename": "AoF001.jpg", "image": {"bytes": b"x"}, "label": ""}, tmp_path / "test")
    assert (tmp_path / "test/labels/AoF001.txt").read_text() == ""

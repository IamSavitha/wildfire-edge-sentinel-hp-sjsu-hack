from sentinel.labels import write_dfire_yaml


def test_write_dfire_yaml_uses_absolute_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "dfire").mkdir(parents=True)
    out = write_dfire_yaml("data/dfire", "runs/sub/dfire.yaml")
    assert out.resolve() == (tmp_path / "runs" / "sub" / "dfire.yaml").resolve()
    lines = out.read_text().splitlines()
    assert f"path: {(tmp_path / 'data' / 'dfire').resolve()}" in lines
    assert "train: train/images" in lines and "val: test/images" in lines
    assert lines[lines.index("names:") + 1:] == ["  0: smoke", "  1: fire"]

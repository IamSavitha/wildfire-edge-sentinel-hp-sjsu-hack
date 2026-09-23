from scripts.make_bench_clips import build_sequences, camera_of


def test_camera_of_strips_timestamp():
    assert camera_of("force-06_courmettes-160_2024-01-08T12-44-06.jpg") == "force-06_courmettes-160"


def test_build_sequences_groups_by_camera_and_flags_smoke(tmp_path):
    imgs, lbls = tmp_path / "images", tmp_path / "labels"
    imgs.mkdir()
    lbls.mkdir()
    for cam, smoky in (("p_camA", True), ("p_camB", False), ("p_camC", False)):
        n = 3 if cam != "p_camC" else 1
        for i in range(n):
            name = f"{cam}_2024-01-0{i + 1}T10-00-00"
            (imgs / f"{name}.jpg").write_bytes(b"x")
            (lbls / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n" if smoky and i == 1 else "")
    seqs = build_sequences(imgs, lbls, min_frames=2)
    assert set(seqs) == {"p_camA", "p_camB"}          # camC too short
    assert seqs["p_camA"]["smoke"] and not seqs["p_camB"]["smoke"]
    assert [p.name for p in seqs["p_camA"]["frames"]] == sorted(p.name for p in seqs["p_camA"]["frames"])

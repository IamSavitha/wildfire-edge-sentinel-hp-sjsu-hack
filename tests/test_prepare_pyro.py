import pytest

from scripts.prepare_pyro import remap_labels


def test_remaps_pyro_smoke_to_our_smoke_id():
    assert remap_labels("1 0.5 0.5 0.1 0.2\n1 0.2 0.3 0.05 0.05") == "0 0.5 0.5 0.1 0.2\n0 0.2 0.3 0.05 0.05\n"


def test_empty_annotations_mean_no_labels():
    assert remap_labels("") == ""


def test_unexpected_class_id_is_rejected():
    with pytest.raises(ValueError):
        remap_labels("0 0.5 0.5 0.1 0.1")

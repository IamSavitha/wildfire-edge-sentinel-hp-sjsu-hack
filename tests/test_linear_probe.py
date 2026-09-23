import pytest

from scripts.linear_probe import group_accuracy


def test_group_accuracy_counts_same_group_as_correct():
    y_true = ["wildland", "campfire", "fog_dust_cloud", "unknown"]
    y_pred = ["structure", "bbq_chimney", "wildland", "unknown"]
    assert group_accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_group_accuracy_rejects_bad_input():
    with pytest.raises(ValueError):
        group_accuracy([], [])
    with pytest.raises(ValueError):
        group_accuracy(["wildland"], ["wildland", "campfire"])

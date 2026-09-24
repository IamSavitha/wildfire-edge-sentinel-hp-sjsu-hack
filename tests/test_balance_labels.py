from collections import Counter

from scripts.balance_labels import balance


def rows(cls, n):
    return [{"image": f"{cls}_{i}.jpg", "label": {"source_type": cls}} for i in range(n)]


def test_balance_resamples_to_target_caps_and_repeats_rare():
    data = rows("fog_dust_cloud", 915) + rows("unknown", 579) + rows("industrial_stack", 328) + rows("vehicle", 10)
    out = Counter(r["label"]["source_type"] for r in balance(data, target=600, cap=700, min_class=100, rare_mult=5))
    assert out["fog_dust_cloud"] == 700          # capped
    assert out["unknown"] == 600                 # downsampled to target
    assert out["industrial_stack"] == 600        # upsampled to target
    assert out["vehicle"] == 50                  # rare class repeated 5x


def test_balance_is_deterministic():
    data = rows("wildland", 200) + rows("campfire", 2)
    assert balance(data, seed=1) == balance(data, seed=1)

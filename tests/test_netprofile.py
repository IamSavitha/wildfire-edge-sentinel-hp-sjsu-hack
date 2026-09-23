import math

import pytest

from sentinel.netprofile import (EDGE_POST_RTTS, PROFILES, LinkProfile, Outages, get_profile, parse_outage,
                                 parse_profiles, rtt_s, serialize_s, transfer_s)


def test_presets():
    assert PROFILES["fiber"] == LinkProfile("fiber", 50_000, 100_000, 20, 0.0)
    assert PROFILES["lte"] == LinkProfile("lte", 5_000, 20_000, 60, 0.005)
    assert PROFILES["rural_cellular"] == LinkProfile("rural_cellular", 1_000, 5_000, 150, 0.02)
    assert PROFILES["satellite_geo"] == LinkProfile("satellite_geo", 512, 2_000, 650, 0.01)
    assert PROFILES["outage"].is_down


def test_transfer_is_rtt_plus_serialization_inflated_by_loss():
    assert transfer_s(125_000, PROFILES["fiber"]) == pytest.approx(0.02 + 0.02)
    assert transfer_s(125_000, PROFILES["rural_cellular"]) == pytest.approx((0.15 + 1.0) / 0.98)
    assert transfer_s(0, PROFILES["satellite_geo"]) == pytest.approx(0.65 / 0.99)
    assert transfer_s(10, PROFILES["outage"]) == math.inf


def test_get_and_parse_profiles():
    assert [p.name for p in parse_profiles("fiber, lte,")] == ["fiber", "lte"]
    with pytest.raises(ValueError, match="known"):
        get_profile("dialup")


def test_outages_down_and_next_up():
    o = Outages([(10, 20), (15, 30), (40, 50)])
    assert o.windows == [(10, 30), (40, 50)]
    assert not o.is_down(9.99) and o.is_down(10) and o.is_down(29.9) and not o.is_down(30)
    assert o.next_up(12) == 30 and o.next_up(35) == 35 and o.next_up(45) == 50
    assert not Outages() and Outages([(0, 1)])


def test_outage_validation():
    with pytest.raises(ValueError):
        Outages([(5, 5)])
    assert parse_outage("0,600") == (0.0, 600.0)
    for bad in ("5", "a,b", "10,2"):
        with pytest.raises(ValueError):
            parse_outage(bad)


def test_serialization_occupies_the_link_rtt_does_not():
    rural = PROFILES["rural_cellular"]
    assert serialize_s(125_000, rural) == pytest.approx(1.0 / 0.98)
    assert rtt_s(rural, 3) == pytest.approx(0.45 / 0.98)
    assert transfer_s(125_000, rural, rtts=EDGE_POST_RTTS) == pytest.approx((1.0 + 0.45) / 0.98)
    assert serialize_s(1, PROFILES["outage"]) == math.inf and rtt_s(PROFILES["outage"]) == math.inf

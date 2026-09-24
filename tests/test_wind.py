import json
import random

import pytest

from sentinel.geo import parse_wxt
from sentinel.wind import SensorEmulator, WindService

SEED = {"source": "test", "captured_utc": "2026-09-24T03:10:00Z", "readings": {
    "RM-WXT536": "1000|Dn=101D,Dm=200D,Dx=324D,Sn=0.3M,Sm=0.8M,Sx=2.3M,Ta=26.3C,Ua=56.6P",
    "LP-WXT536": "1000|Dn=272D,Dm=293D,Dx=323D,Sn=3.3M,Sm=4.8M,Sx=5.3M,Ta=23.2C,Ua=50.0P"}}


class Clock:
    def __init__(self, t=10_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def seed(tmp_path):
    p = tmp_path / "wx_seed.json"
    p.write_text(json.dumps(SEED))
    return p


def test_emulator_stays_near_the_real_reading(seed):
    clock = Clock()
    em = SensorEmulator(seed, clock=clock, rng=random.Random(1))
    for _ in range(500):
        clock.t += 10
        em.step()
        r = parse_wxt(em.read("LP-WXT536"))
        assert r["observed_at"] == int(clock.t)
        assert abs((r["dir_from_deg"] - 293 + 180) % 360 - 180) <= 30
        assert 4.8 * 0.6 - 1e-9 <= r["speed_mps"] <= 4.8 * 1.4 + 0.5
        assert r["gust_mps"] >= r["speed_mps"]
    assert em.read("NOPE") is None


def test_fresh_sensor_wins(tmp_path, seed):
    clock = Clock()
    em = SensorEmulator(seed, clock=clock)
    ws = WindService(em.read, tmp_path / "fc.json", clock=clock, sensor_label="emulated on-site sensor")
    w = ws.wind_for("RM-WXT536", "rm-e", manual={"dir_from_deg": 10, "speed_mps": 9})
    assert w["source"] == "sensor" and w["emulated"] and not w["stale"]
    assert w["spread_deg"] == 60


def test_stale_sensor_falls_back_to_manual_then_forecast_then_stale_sensor(tmp_path, seed):
    clock = Clock()
    em = SensorEmulator(seed, clock=clock)
    ws = WindService(em.read, tmp_path / "fc.json", clock=clock)
    clock.t += 3600                                            # sensor now an hour old
    assert ws.wind_for("RM-WXT536", "rm-e", manual={"dir_from_deg": 10, "speed_mps": 9,
                                                    "observed_at": clock.t})["source"] == "manual"
    assert ws.wind_for("RM-WXT536", "rm-e")["source"] == "sensor"          # stale, but all we have
    ws.remember_forecast("rm-e", {"wind_mph": [10.0, 20.0, 30.0], "wind_dir_deg": [270, 280, 290]})
    w = ws.wind_for("RM-WXT536", "rm-e")
    assert w["source"] == "forecast_cache"
    assert w["speed_mps"] == pytest.approx(4.4704) and w["dir_from_deg"] == 270
    clock.t += 3600 * 1.5                                      # second hour of the forecast
    assert ws.wind_for(None, "rm-e")["dir_from_deg"] == 280
    clock.t += 3600 * 10                                       # past the end: last hour, marked stale
    w = ws.wind_for(None, "rm-e")
    assert w["dir_from_deg"] == 290 and w["stale"]


def test_forecast_cache_persists_and_no_source_gives_none(tmp_path):
    clock = Clock()
    ws = WindService(None, tmp_path / "fc.json", clock=clock)
    assert ws.wind_for("RM-WXT536", "k") is None
    ws.remember_forecast("k", {"wind_mph": [5.0], "wind_dir_deg": [90]})
    again = WindService(None, tmp_path / "fc.json", clock=clock)
    assert again.wind_for(None, "k")["dir_from_deg"] == 90
    ws.remember_forecast("bad", {"wind_mph": [None], "wind_dir_deg": [None]})
    assert ws.wind_for(None, "bad") is None

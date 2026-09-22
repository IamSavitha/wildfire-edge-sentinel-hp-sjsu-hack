import json

from sentinel.cloud import fetch_burn_schedule, parse_open_meteo


def test_parse_open_meteo():
    data = {"hourly": {"time": ["t0", "t1"], "temperature_2m": [34.0, 36.5],
                       "wind_speed_10m": [15.0, 17.0], "wind_direction_10m": [270, 275]}}
    assert parse_open_meteo(data) == {"time": ["t0", "t1"], "temp_c": [34.0, 36.5],
                                      "wind_mph": [15.0, 17.0], "wind_dir_deg": [270, 275]}


def test_burn_schedule(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps(["t2"]))
    assert fetch_burn_schedule(p) == {"t2"}
    assert fetch_burn_schedule(tmp_path / "none.json") == set()

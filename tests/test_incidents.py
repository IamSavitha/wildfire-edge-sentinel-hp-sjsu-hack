from sentinel.incidents import IncidentStore


def obs(**kw):
    base = {"camera_id": "rm-e", "lat": 33.4, "lon": -117.0, "bearing_deg": 95.0, "severity": "MONITOR",
            "source_type": "wildland", "site_short": "Red Mountain", "name": None}
    return {**base, **kw}


def test_first_report_creates_a_named_incident(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    inc, created = s.record(obs(), now=100)
    assert created and inc["name"] == "Red Mountain Incident 1" and inc["detections"] == 1
    assert s.get(inc["id"])["status"] == "active"


def test_same_camera_and_bearing_merges_and_severity_only_rises(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    a, _ = s.record(obs(severity="ALERT"), now=100)
    b, created = s.record(obs(bearing_deg=100.0, severity="LOG", source_type="campfire"), now=200)
    assert not created and b["id"] == a["id"]
    assert b["severity"] == "ALERT" and b["detections"] == 2 and b["updated_at"] == 200
    assert [h["severity"] for h in b["history"]] == ["ALERT", "LOG"]


def test_different_bearing_camera_or_time_makes_a_new_incident(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    a, _ = s.record(obs(), now=100)
    assert s.record(obs(bearing_deg=130.0), now=110)[0]["id"] != a["id"]
    assert s.record(obs(camera_id="lp-w"), now=120)[0]["id"] != a["id"]
    assert s.record(obs(), now=100 + 3 * 3600)[0]["id"] != a["id"]
    assert len(s.all()) == 4
    assert s.all()[0]["seq"] == 4


def test_bearing_merge_wraps_through_north(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    a, _ = s.record(obs(bearing_deg=355.0), now=0)
    assert s.record(obs(bearing_deg=4.0), now=1)[0]["id"] == a["id"]


def test_reports_without_bearing_merge_by_distance(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    a, _ = s.record(obs(camera_id=None, bearing_deg=None), now=0)
    assert s.record(obs(camera_id=None, bearing_deg=None, lat=33.41), now=1)[0]["id"] == a["id"]
    assert s.record(obs(camera_id=None, bearing_deg=None, lat=33.6), now=2)[0]["id"] != a["id"]


def test_operator_pin_survives_later_reports_and_resolved_incidents_do_not_merge(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    a, _ = s.record(obs(), now=0)
    a.update(lat=33.5, lon=-116.9, loc_source="map_pin", county="Riverside")
    s.save(a)
    b, _ = s.record(obs(lat=33.41, loc_source="camera_bearing", county="San Diego"), now=5)
    assert (b["lat"], b["loc_source"], b["county"]) == (33.5, "map_pin", "Riverside")
    b["status"] = "resolved"
    s.save(b)
    assert s.record(obs(), now=6)[1] is True
    s.clear()
    assert s.all() == []


def test_replayed_old_recording_does_not_join_a_newer_incident(tmp_path):
    s = IncidentStore(str(tmp_path / "i.db"))
    live, _ = s.record(obs(), now=2_000_000_000)
    old, created = s.record(obs(), now=1_750_000_000)        # same camera and bearing, months earlier
    assert created and old["id"] != live["id"]

import io

import pytest
from PIL import Image

from sentinel.geo import (box_bearings, compass, county_at, destination, direction_spread, downwind_cone,
                          exif_gps, format_wxt, parse_wxt, pixel_bearing, sector)
from sentinel.incidents import haversine_km

RAW = ("1790194201|Dn=211D,Dm=082D,Dx=308D,Sn=0.6M,Sm=3.0M,Sx=5.8M,Ta=21.8C,Ua=16.0P,Pa=812.9H,"
       "Rc=10.79M,Rd=12130s,Ri=0.0M,Hc=0.0M,Hd=0s,Hi=0.0M,Vs=25.3V,Vr=3.655V")


def jpeg_with_gps(lat_dms, lat_ref, lon_dms, lon_ref) -> bytes:
    im = Image.new("RGB", (32, 32))
    exif = Image.Exif()
    exif.get_ifd(0x8825).update({1: lat_ref, 2: lat_dms, 3: lon_ref, 4: lon_dms})
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def test_exif_gps_reads_signed_degrees():
    data = jpeg_with_gps((33.0, 24.0, 2.844), "N", (117.0, 11.0, 25.692), "W")
    lat, lon = exif_gps(data)
    assert lat == pytest.approx(33.40079, abs=1e-5)
    assert lon == pytest.approx(-117.19047, abs=1e-5)


def test_exif_gps_none_without_tags_or_for_garbage():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG")
    assert exif_gps(buf.getvalue()) is None
    assert exif_gps(b"not an image") is None


def test_pixel_bearing_edges_and_centre():
    assert pixel_bearing(90, 90, 0.5) == pytest.approx(90)
    assert pixel_bearing(90, 90, 0.0) == pytest.approx(45)
    assert pixel_bearing(90, 90, 1.0) == pytest.approx(135)
    assert pixel_bearing(0, 90, 0.0) == pytest.approx(315)          # wraps through north
    # pinhole, not linear: a quarter of the way in is less than a quarter of the view off-centre
    assert pixel_bearing(0, 90, 0.75) == pytest.approx(26.565, abs=1e-3)


def test_box_bearings_order():
    lo, mid, hi = box_bearings(180, 90, (0, 10, 1000, 20), 2000)
    assert lo == pytest.approx(135) and mid < 180 and lo < mid < hi


def test_destination_distance_and_direction():
    lat, lon = destination(33.4, -117.19, 90, 10)
    assert haversine_km(33.4, -117.19, lat, lon) == pytest.approx(10, rel=1e-3)
    assert lat == pytest.approx(33.4, abs=0.01) and lon > -117.19
    lat, _ = destination(33.4, -117.19, 0, 10)
    assert lat > 33.4


def test_sector_is_closed_and_spans_the_angle():
    ring = sector(33.0, -117.0, 45, 135, 20, steps=10)
    assert ring[0] == ring[-1] == [-117.0, 33.0]
    assert len(ring) == 13
    assert all(haversine_km(33.0, -117.0, p[1], p[0]) == pytest.approx(20, rel=1e-3) for p in ring[1:-1])


def test_downwind_cone_points_away_from_the_wind_and_scales_with_speed():
    # wind FROM the west (270) blows smoke east
    c = downwind_cone(33.0, -117.0, 270, 10.0, hours=1)
    assert c["toward_deg"] == pytest.approx(90)
    assert c["length_km"] == pytest.approx(3.6)                  # 36 km/h * 10% * 1 h
    assert downwind_cone(33.0, -117.0, 270, 10.0, hours=3)["length_km"] == pytest.approx(10.8)
    assert downwind_cone(33.0, -117.0, 270, 0.0, hours=1)["length_km"] == pytest.approx(0.5)   # floor
    assert downwind_cone(33.0, -117.0, 270, 5.0, 1, spread_deg=90)["half_angle_deg"] == 60


def test_compass():
    assert [compass(d) for d in (0, 44, 90, 200, 359)] == ["N", "NE", "E", "SSW", "N"]


def test_county_at_polygon_and_multipolygon_with_hole():
    square = [[-118, 33], [-116, 33], [-116, 34], [-118, 34], [-118, 33]]
    hole = [[-117.2, 33.4], [-116.8, 33.4], [-116.8, 33.6], [-117.2, 33.6], [-117.2, 33.4]]
    fc = {"features": [
        {"properties": {"name": "Holey"}, "geometry": {"type": "Polygon", "coordinates": [square, hole]}},
        {"properties": {"name": "Island"}, "geometry": {"type": "MultiPolygon", "coordinates": [
            [[[-120, 35], [-119, 35], [-119, 36], [-120, 35]]], [[[-117.1, 33.45], [-116.9, 33.45],
                                                                   [-116.9, 33.55], [-117.1, 33.45]]]]}}]}
    assert county_at(33.2, -117.5, fc) == "Holey"
    assert county_at(33.46, -117.0, fc) == "Island"           # inside the hole, inside the island
    assert county_at(40.0, -100.0, fc) is None


def test_parse_wxt_real_hpwren_record():
    r = parse_wxt(RAW)
    assert r["observed_at"] == 1790194201
    assert (r["dir_from_deg"], r["speed_mps"], r["gust_mps"]) == (82.0, 3.0, 5.8)
    assert (r["dir_min_deg"], r["dir_max_deg"], r["temp_c"], r["rh_pct"]) == (211.0, 308.0, 21.8, 16.0)


def test_parse_wxt_rejects_missing_wind_and_wrong_units():
    with pytest.raises(ValueError):
        parse_wxt("1|Ta=20.0C")
    with pytest.raises(ValueError):
        parse_wxt("1|Dm=090D,Sm=3.0K")


def test_format_wxt_round_trips():
    r = parse_wxt(RAW)
    back = parse_wxt(format_wxt(123, r))
    assert back["observed_at"] == 123
    assert back["dir_from_deg"] == 82 and back["speed_mps"] == 3.0 and back["gust_mps"] == 5.8


def test_direction_spread_wraps_and_clamps():
    assert direction_spread(350, 10) == 10          # 20 deg range -> 10, clamped at the floor
    assert direction_spread(211, 308) == pytest.approx(48.5)
    assert direction_spread(101, 324) == 60
    assert direction_spread(None, 5) == 20


def test_ray_intersection_finds_the_crossing_point():
    from sentinel.geo import ray_intersection
    fire = (33.20, -116.95)
    cams = [(33.40079, -117.19047), (32.70153, -116.76457)]
    bearings = []
    for lat, lon in cams:     # bearing from camera to fire via a tiny step search on destination()
        best = min(range(3600), key=lambda d: haversine_km(*destination(lat, lon, d / 10, haversine_km(lat, lon, *fire)), *fire))
        bearings.append(best / 10)
    lat, lon, d1, d2 = ray_intersection(*cams[0], bearings[0], *cams[1], bearings[1])
    assert haversine_km(lat, lon, *fire) < 0.3
    assert d1 == pytest.approx(haversine_km(*cams[0], *fire), rel=0.02)
    # behind a camera, near-parallel, or too far away: no fix
    assert ray_intersection(*cams[0], (bearings[0] + 180) % 360, *cams[1], bearings[1]) is None
    assert ray_intersection(33.0, -117.0, 90, 33.1, -117.0, 92) is None
    assert ray_intersection(*cams[0], bearings[0], *cams[1], bearings[1], max_km=5) is None


def test_triangulate_three_cameras_and_rejects_bad_geometry():
    from sentinel.geo import triangulate
    fire = (33.1, -116.9)
    cams = [(33.40079, -117.19047), (32.70153, -116.76457), (33.36302, -116.83622)]
    sights = []
    for lat, lon in cams:
        best = min(range(3600), key=lambda d: haversine_km(*destination(lat, lon, d / 10, haversine_km(lat, lon, *fire)), *fire))
        sights.append({"lat": lat, "lon": lon, "bearing_deg": best / 10})
    t = triangulate(sights)
    assert t["n"] == 3 and haversine_km(t["lat"], t["lon"], *fire) < 0.5 and t["spread_km"] < 0.5
    assert triangulate(sights[:1]) is None
    sights[0]["bearing_deg"] = (sights[0]["bearing_deg"] + 180) % 360     # one camera looking away
    assert triangulate(sights) is None


def test_parse_wxt_skips_values_the_station_flags_invalid():
    r = parse_wxt("1790194201|Dn=000#,Dm=245D,Dx=000#,Sn=0.0#,Sm=1.2M,Sx=0.0#,Ta=20.1C")
    assert r["dir_from_deg"] == 245 and r["speed_mps"] == 1.2
    assert "speed_min_mps" not in r and "gust_mps" not in r and "dir_min_deg" not in r
    with pytest.raises(ValueError):
        parse_wxt("1|Dn=000#,Dm=000#,Sm=0.0#")

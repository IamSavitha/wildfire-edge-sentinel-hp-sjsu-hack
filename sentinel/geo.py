"""Offline geometry for the incident map: EXIF GPS, bearing from a tower camera, sectors, the
downwind cone, county lookup and the on-site weather station (Vaisala WXT5xx) record format."""
import io
import math

from PIL import Image

EARTH_KM = 6371.0088
GPS_IFD = 0x8825
MPS_TO_KMH = 3.6
# Rule of thumb for grass/open fuels: forward spread ~10% of the 10 m open wind speed
# (Cruz & Alexander 2019, "The 10% wind speed rule of thumb"). Indicative only, not a spread model.
SPREAD_FRACTION = 0.10


# ---------------------------------------------------------------- location sources

def _dms(value, ref) -> float:
    d, m, s = (float(x) for x in value)
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def exif_gps(data: bytes) -> tuple[float, float] | None:
    """(lat, lon) from an image's EXIF GPS tags, or None if the image has none."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            gps = im.getexif().get_ifd(GPS_IFD)
    except Exception:  # noqa: BLE001 - an unreadable header just means "no GPS"
        return None
    try:
        lat = _dms(gps[2], gps.get(1, "N"))
        lon = _dms(gps[4], gps.get(3, "E"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None
    return lat, lon


def valid_latlon(lat: float, lon: float) -> bool:
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


# ---------------------------------------------------------------- bearings and shapes

def norm_deg(d: float) -> float:
    return d % 360.0


def pixel_bearing(azimuth_deg: float, hfov_deg: float, x_frac: float) -> float:
    """Compass bearing of an image column (0 = left edge, 1 = right edge) for a pinhole camera
    pointed at azimuth_deg with horizontal field of view hfov_deg."""
    off = math.degrees(math.atan((2 * x_frac - 1) * math.tan(math.radians(hfov_deg / 2))))
    return norm_deg(azimuth_deg + off)


def box_bearings(azimuth_deg: float, hfov_deg: float, box, width: int) -> tuple[float, float, float]:
    """(left, centre, right) bearings of a detection box (x1, y1, x2, y2) in a frame `width` px wide."""
    x1, _, x2, _ = box
    return (pixel_bearing(azimuth_deg, hfov_deg, x1 / width),
            pixel_bearing(azimuth_deg, hfov_deg, (x1 + x2) / 2 / width),
            pixel_bearing(azimuth_deg, hfov_deg, x2 / width))


def destination(lat: float, lon: float, bearing_deg: float, dist_km: float) -> tuple[float, float]:
    """Great-circle point dist_km from (lat, lon) along bearing_deg."""
    p1, l1, b = math.radians(lat), math.radians(lon), math.radians(bearing_deg)
    d = dist_km / EARTH_KM
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def sector(lat: float, lon: float, start_deg: float, end_deg: float, radius_km: float,
           steps: int = 24) -> list[list[float]]:
    """Closed GeoJSON ring ([lon, lat] pairs) of a pie slice clockwise from start_deg to end_deg."""
    span = (end_deg - start_deg) % 360 or 360.0
    ring = [[lon, lat]]
    for i in range(steps + 1):
        la, lo = destination(lat, lon, start_deg + span * i / steps, radius_km)
        ring.append([lo, la])
    ring.append([lon, lat])
    return ring


def downwind_cone(lat: float, lon: float, wind_from_deg: float, speed_mps: float, hours: float,
                  spread_deg: float = 20.0, min_km: float = 0.5) -> dict:
    """Area smoke/fire would move into over `hours` if it spread at 10% of the wind speed.
    Wind direction is where the wind blows FROM (meteorological), so the cone points the other way."""
    toward = norm_deg(wind_from_deg + 180)
    half = min(max(spread_deg, 10.0), 60.0)
    length = max(min_km, speed_mps * MPS_TO_KMH * SPREAD_FRACTION * hours)
    return {"toward_deg": toward, "half_angle_deg": half, "length_km": length,
            "ring": sector(lat, lon, toward - half, toward + half, length)}


def compass(deg: float) -> str:
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return pts[int((norm_deg(deg) + 11.25) // 22.5) % 16]


# ---------------------------------------------------------------- counties (offline point-in-polygon)

def _in_ring(lon: float, lat: float, ring) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _in_polygon(lon: float, lat: float, rings) -> bool:
    return _in_ring(lon, lat, rings[0]) and not any(_in_ring(lon, lat, h) for h in rings[1:])


def county_at(lat: float, lon: float, counties: dict) -> str | None:
    """Name of the county (GeoJSON FeatureCollection, properties.name) containing the point."""
    for f in counties.get("features", []):
        g = f.get("geometry") or {}
        polys = [g["coordinates"]] if g.get("type") == "Polygon" else g.get("coordinates", []) \
            if g.get("type") == "MultiPolygon" else []
        if any(_in_polygon(lon, lat, p) for p in polys):
            return f["properties"].get("name")
    return None


# ---------------------------------------------------------------- on-site weather station

_WXT_UNITS = {"Dn": "dir_min_deg", "Dm": "dir_from_deg", "Dx": "dir_max_deg",
              "Sn": "speed_min_mps", "Sm": "speed_mps", "Sx": "gust_mps", "Ta": "temp_c", "Ua": "rh_pct"}


def parse_wxt(raw: str) -> dict:
    """Parse one HPWREN WXT5xx record, "<epoch>|Dn=211D,Dm=082D,...,Sm=3.0M,...", into floats.
    Wind direction is where the wind blows FROM; speeds are m/s (the M unit). A value whose unit is "#"
    is the station flagging it invalid, and is skipped."""
    ts, _, body = raw.partition("|")
    out = {"observed_at": float(ts)}
    for item in body.split(","):
        key, _, val = item.partition("=")
        name = _WXT_UNITS.get(key.strip())
        if name and len(val) > 1 and val[-1] != "#":
            if key.strip()[0] == "S" and val[-1] != "M":
                raise ValueError(f"expected m/s wind speed, got {item!r}")
            out[name] = float(val[:-1])
    if "dir_from_deg" not in out or "speed_mps" not in out:
        raise ValueError("record has no mean wind (Dm/Sm)")
    return out


def format_wxt(ts: float, r: dict) -> str:
    """Inverse of parse_wxt for the fields the emulator writes."""
    return (f"{int(ts)}|Dn={r['dir_min_deg']:03.0f}D,Dm={r['dir_from_deg']:03.0f}D,Dx={r['dir_max_deg']:03.0f}D,"
            f"Sn={r['speed_min_mps']:.1f}M,Sm={r['speed_mps']:.1f}M,Sx={r['gust_mps']:.1f}M,"
            f"Ta={r['temp_c']:.1f}C,Ua={r['rh_pct']:.1f}P")


def direction_spread(dir_min: float | None, dir_max: float | None, default: float = 20.0) -> float:
    """Half the observed direction range (clockwise Dn -> Dx), clamped to 10-60 degrees."""
    if dir_min is None or dir_max is None:
        return default
    return min(max(((dir_max - dir_min) % 360) / 2, 10.0), 60.0)


# ---------------------------------------------------------------- triangulation (two or more towers)

def _to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Local flat projection in km around (lat0, lon0); fine at tower-network scale (< 100 km)."""
    return (math.radians(lon - lon0) * EARTH_KM * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * EARTH_KM)


def _from_xy(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    return (lat0 + math.degrees(y / EARTH_KM),
            lon0 + math.degrees(x / (EARTH_KM * math.cos(math.radians(lat0)))))


def ray_intersection(lat1: float, lon1: float, b1: float, lat2: float, lon2: float, b2: float,
                     max_km: float = 60.0) -> tuple[float, float, float, float] | None:
    """Where the bearing lines from two cameras cross: (lat, lon, km from camera 1, km from camera 2),
    or None if they are near-parallel, cross behind a camera, or further than max_km from either."""
    lat0, lon0 = (lat1 + lat2) / 2, (lon1 + lon2) / 2
    x1, y1 = _to_xy(lat1, lon1, lat0, lon0)
    x2, y2 = _to_xy(lat2, lon2, lat0, lon0)
    d1 = (math.sin(math.radians(b1)), math.cos(math.radians(b1)))
    d2 = (math.sin(math.radians(b2)), math.cos(math.radians(b2)))
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < math.sin(math.radians(5)):          # within 5 degrees of parallel: no reliable fix
        return None
    t1 = ((x2 - x1) * d2[1] - (y2 - y1) * d2[0]) / den
    t2 = ((x2 - x1) * d1[1] - (y2 - y1) * d1[0]) / den
    if t1 <= 0.2 or t2 <= 0.2 or t1 > max_km or t2 > max_km:
        return None
    lat, lon = _from_xy(x1 + t1 * d1[0], y1 + t1 * d1[1], lat0, lon0)
    return lat, lon, t1, t2


def triangulate(sightings: list[dict], max_km: float = 60.0) -> dict | None:
    """Best point for bearings from two or more cameras ({lat, lon, bearing_deg} each): the least-squares
    point closest to all bearing lines, kept only if every line reaches it in front of its camera.
    Returns {lat, lon, n, spread_km} where spread_km is the largest distance from the point to a line."""
    if len(sightings) < 2:
        return None
    lat0 = sum(s["lat"] for s in sightings) / len(sightings)
    lon0 = sum(s["lon"] for s in sightings) / len(sightings)
    a11 = a12 = a22 = b1 = b2 = 0.0
    lines = []
    for s in sightings:
        x, y = _to_xy(s["lat"], s["lon"], lat0, lon0)
        dx, dy = math.sin(math.radians(s["bearing_deg"])), math.cos(math.radians(s["bearing_deg"]))
        nx, ny = dy, -dx                                  # unit normal of the bearing line
        c = nx * x + ny * y
        a11 += nx * nx; a12 += nx * ny; a22 += ny * ny; b1 += nx * c; b2 += ny * c
        lines.append((x, y, dx, dy, nx, ny, c))
    det = a11 * a22 - a12 * a12
    if abs(det) < 1e-6:
        return None
    px, py = (b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det
    spread = 0.0
    for x, y, dx, dy, nx, ny, c in lines:
        along = (px - x) * dx + (py - y) * dy
        if along <= 0.2 or along > max_km:
            return None
        spread = max(spread, abs(nx * px + ny * py - c))
    lat, lon = _from_xy(px, py, lat0, lon0)
    return {"lat": lat, "lon": lon, "n": len(sightings), "spread_km": spread}


def line_offset_km(cam_lat: float, cam_lon: float, bearing_deg: float, lat: float, lon: float) -> tuple[float, float]:
    """(distance along the camera's bearing line, perpendicular distance from it) of a point, in km.
    A negative 'along' means the point is behind the camera."""
    x, y = _to_xy(lat, lon, cam_lat, cam_lon)
    dx, dy = math.sin(math.radians(bearing_deg)), math.cos(math.radians(bearing_deg))
    return x * dx + y * dy, abs(x * dy - y * dx)

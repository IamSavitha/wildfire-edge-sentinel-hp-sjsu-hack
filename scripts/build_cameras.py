"""Build config/cameras.json from the FIgLib recordings on disk and HPWREN's published camera list.

Each recording folder is <date>_<Fire name>_<site>-<dir>-mobo-<c|m>; the camera's position, heading and
field of view come from https://www.hpwren.ucsd.edu/cameras/sites.js (saved by setup_map_assets.sh), and its
weather station from the stations listed in wxtcf.json. Also refreshes state/wx_seed.json with one real
reading per station (from the saved wxtcf.json) so the sensor emulator covers every camera.

    python scripts/build_cameras.py            # after scripts/fetch_figlib.sh
"""
import argparse
import datetime
import json
import re
from pathlib import Path

SEQ_RE = re.compile(r"^(\d{8})_(.+?)_([a-z0-9]+)-([a-z]+)-mobo-([cm])$")


def load_sites(path: Path) -> dict:
    t = path.read_text()
    t = t[t.index("{"):t.rindex("}") + 1]
    t = re.sub(r"//[^\n]*", "", t)
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    return json.loads(t)


def fire_name(raw: str) -> str:
    """"JunctionFire" -> "Junction Fire", "Fire" -> "Fire"."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw).replace("Fir ", "Fire ").strip()


def main(argv=None):
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", default=str(home / "sentinel-assets"))
    ap.add_argument("--out", default="config/cameras.json")
    a = ap.parse_args(argv)
    assets = Path(a.assets)
    sites = load_sites(assets / "reference" / "sites.js")
    wx = json.loads((assets / "reference" / "wxtcf.json").read_text())

    cams: dict[str, dict] = {}
    for seq in sorted(p.name for p in (assets / "figlib").iterdir() if p.is_dir() and not p.name.startswith(".")):
        m = SEQ_RE.match(seq)
        if not m:
            print("skip (name)", seq)
            continue
        date, fire, site, direction, imager = m.groups()
        cam_key = f"{site}-{direction}-mobo-{imager}"
        s = sites.get(site)
        c = (s or {}).get("cams", {}).get(cam_key)
        if not s or not c:
            print("skip (camera not in sites.js)", seq)
            continue
        cid = f"{site}-{direction}"
        station = next((k for k in (f"{site.upper()}-WXT536", f"{site.upper()}-WXT520") if k in wx), None)
        cam = cams.setdefault(cid, {
            "id": cid, "name": f"{c['name']} ({ {'n': 'north', 'e': 'east', 's': 'south', 'w': 'west'}.get(direction, direction)})",
            "site": s["name"], "lat": s["lat"], "lon": s["long"], "elev_m": s.get("elev"),
            "azimuth_deg": c["azimuth"], "hfov_deg": c["horizontalView"], "wx_station": station,
            "hpwren_cam": cam_key, "recordings": []})
        cam["recordings"].append({"folder": seq, "fire": fire_name(fire),
                                  "date": f"{date[:4]}-{date[4:6]}-{date[6:]}"})
    out = sorted(cams.values(), key=lambda c: c["id"])
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out)} cameras, {sum(len(c['recordings']) for c in out)} recordings -> {a.out}")

    seed_path = assets / "state" / "wx_seed.json"
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else {"readings": {}}
    for c in out:
        if c["wx_station"] and c["wx_station"] not in seed["readings"]:
            seed["readings"][c["wx_station"]] = wx[c["wx_station"]]
    seed.update(source="https://cdn.hpwren.ucsd.edu/RT/wxtcf.json",
                captured_utc=seed.get("captured_utc") or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"))
    seed_path.write_text(json.dumps(seed, indent=1))
    print("sensor seed stations:", ", ".join(sorted(seed["readings"])))


if __name__ == "__main__":
    main()

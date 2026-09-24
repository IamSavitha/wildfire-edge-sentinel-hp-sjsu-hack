"""Incident map: an offline wildfire incident dashboard for the lookout-tower network.

Every detection the edge cascade confirms is placed on an offline map of California (Protomaps
vector tiles served from this box), tagged with a location, a county and the wind from the tower's
own weather station, with a bearing wedge from the camera and an indicative downwind cone.
Only ALERTs leave the device, through the same escalation rule as the rest of the system.

    python -m sentinel.incident_map            # then open http://localhost:8100
"""
import argparse
import hashlib
import json
import logging
import re
import shutil
import threading
import time
import uuid
from types import SimpleNamespace
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from sentinel.assistant import Assistant
from sentinel.config import Tower
from sentinel.geo import (box_bearings, compass, county_at, destination, downwind_cone, exif_gps,
                          line_offset_km, ray_intersection, sector, triangulate, valid_latlon)
from sentinel.imaging import crop_box, to_jpeg
from sentinel.incidents import CROSS_AGREE_KM, CROSS_MAX_KM, CROSS_WINDOW_S, SEVERITY_RANK, IncidentStore, haversine_km
from sentinel.report import build_report, render_text
from sentinel.schema import ContextResult
from sentinel.severity import assess, fallback_severity
from sentinel.watch import (TIMELINE_CAP, WatchSession, list_frames, merge_streams, summarize_batch, time_steps,
                            update_text)
from sentinel.webapp import _read_body, decode_image, list_served_models, parse_multipart
from sentinel.webapp_logic import _thumbnail, cloud_frame_tokens, escalate, qwen_image_tokens, run_pass
from sentinel.wind import SensorEmulator, WindService

PAGE = Path(__file__).parent / "static" / "incident_map.html"
HOME = Path.home()
DEFAULT_ASSETS = HOME / "sentinel-assets"
MAP_FILE = "sierra-pacific.pmtiles"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
CAMERA_VIEW_KM = 25.0          # how far the camera field-of-view fans are drawn
WEDGE_KM = 30.0                # how far an incident's bearing wedge is drawn
BENIGN_CAP = 24
PHOTO_CAP = 36
UPDATES_CAP = 80
STATION_MAX_KM = 40.0
FRAMES_CAP = 400
FRAME_SIDE = 1280
_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_FIGLIB_RE = re.compile(r"_([+-]\d+)\.[A-Za-z]+$")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

log = logging.getLogger(__name__)


def load_cameras(path: str | Path) -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(Path(path).read_text())}


def _sid(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def recording_dirs(cam: dict, roots: list[Path]) -> list[tuple[Path, dict]]:
    """(folder, recording) for each FIgLib recording of a camera found under one of the roots."""
    recs = list(cam.get("recordings") or [])
    if cam.get("figlib"):                                   # older single-recording config
        recs.append({"folder": cam["figlib"], "fire": None, "date": None})
    out = []
    for rec in recs:
        for root in roots:
            d = root / rec["folder"]
            if d.is_dir():
                out.append((d, rec))
                break
    return out


def recording_key(rec: dict) -> str:
    return f"{rec['date']}_{rec['fire']}" if rec.get("fire") and rec.get("date") else rec["folder"]


def scan_recordings(roots: list[Path], cameras: dict[str, dict]) -> dict[str, dict]:
    """Recordings grouped by fire: one fire seen by several towers is one entry with several cameras."""
    groups: dict[str, dict] = {}
    for cam in cameras.values():
        for folder, rec in recording_dirs(cam, roots):
            key = recording_key(rec)
            g = groups.setdefault(key, {"key": key, "fire": rec.get("fire") or folder.name, "date": rec.get("date"),
                                        "cameras": [], "folders": {}})
            g["cameras"].append(cam["id"])
            g["folders"][cam["id"]] = folder
    return dict(sorted(groups.items(), key=lambda kv: (kv[1]["date"] or "", kv[0]), reverse=True))


def pick_flame_photos(dfire_split: Path, n: int = PHOTO_CAP) -> list[Path]:
    """Close-up photos with visible flames from a D-Fire split (label class 1 = fire covering >= 3% of
    the frame), for demo field reports. Deterministic: sorted by name, first n that qualify."""
    out = []
    for lab in sorted((dfire_split / "labels").glob("*.txt")):
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
        fire = [float(r[3]) * float(r[4]) for r in rows if len(r) == 5 and r[0] == "1"]
        smoke = any(r[0] == "0" for r in rows if len(r) == 5)
        img = dfire_split / "images" / (lab.stem + ".jpg")
        if fire and max(fire) >= 0.03 and smoke and img.exists():
            out.append(img)
            if len(out) >= n:
                break
    return out


def scan_samples(roots: list[Path], benign_dir: Path | None, cameras: dict[str, dict],
                 photo_dir: Path | None = None) -> dict[str, dict]:
    """Real FIgLib frames for each camera (offset from ignition in the name) plus a few benign
    smoke-free tower frames that can be sent through any camera."""
    out: dict[str, dict] = {}
    for cam in cameras.values():
        for folder, rec in recording_dirs(cam, roots):
            for p in sorted(folder.iterdir()):
                m = _FIGLIB_RE.search(p.name)
                if p.suffix.lower() in IMAGE_EXT and m:
                    out[_sid(p)] = {"id": _sid(p), "camera_id": cam["id"], "kind": "figlib",
                                    "recording": recording_key(rec), "fire": rec.get("fire"),
                                    "offset_s": int(m.group(1)), "name": f"{folder.name}/{p.name}", "path": p}
    if photo_dir and (photo_dir / "images").is_dir():
        for p in pick_flame_photos(photo_dir):
            out[_sid(p)] = {"id": _sid(p), "camera_id": None, "kind": "photo", "offset_s": None,
                            "recording": None, "fire": None, "name": f"photo/{p.name}", "path": p}
    if benign_dir and benign_dir.is_dir():
        files = sorted(p for p in benign_dir.rglob("*") if p.suffix.lower() in IMAGE_EXT)[:BENIGN_CAP]
        for p in files:
            out[_sid(p)] = {"id": _sid(p), "camera_id": None, "kind": "benign", "offset_s": None,
                            "recording": None, "fire": None, "name": f"benign/{p.name}", "path": p}
    return out


def default_detector_factory(weights: str):
    from sentinel.detector import YoloDetector
    return YoloDetector(weights, conf=0.1)


class NetworkState(BaseModel):
    online: bool


class WatchRequest(BaseModel):
    recording: str                      # a key from /api/recordings, or "all"
    interval_s: float = 2.0
    batch_frames: int = 5
    time_mode: str = "live"
    start_offset_s: int | None = -600


class Question(BaseModel):
    question: str
    selected: str | None = None


class IncidentPatch(BaseModel):
    lat: float | None = None
    lon: float | None = None
    status: str | None = None
    name: str | None = None
    wind_from_deg: float | None = None
    wind_speed_mps: float | None = None


def create_map_app(*, cameras: dict[str, dict], assets_dir: str | Path = DEFAULT_ASSETS,
                   data_dir: str | Path | None = None, state_dir: str | Path | None = None,
                   figlib_dir: str | Path | None = None,
                   detector_weights: str = str(HOME / "sentinel" / "models" / "tower_yolo.pt"),
                   photo_detector_weights: str | None = str(HOME / "sentinel" / "models" / "smoke_yolo.pt"),
                   detector_factory: Callable[[str], Callable] | None = None,
                   vlm_factory: Callable[[str], object] | None = None, vlm_model: str = "context",
                   vlm_base_url: str = "http://localhost:8000/v1", vlm_timeout_s: float = 60.0,
                   model_lister: Callable[[], list[str]] | None = None,
                   forecast_fn: Callable[[float, float], dict] | None = None,
                   sensor: SensorEmulator | None = None, sensor_interval_s: float = 10.0,
                   counties: dict | None = None, range_km: float = 10.0, min_conf: float = 0.4,
                   gate_frames: int = 2, assistant_model: str = "base7b",
                   llm_factory: Callable[[], object] | None = None,
                   demo_scenarios: str | Path | None = "config/demo_scenarios.json",
                   start_online: bool = False, clock: Callable[[], float] = time.time) -> FastAPI:
    assets_dir = Path(assets_dir)
    state_dir = Path(state_dir) if state_dir else assets_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    detector_factory = detector_factory or default_detector_factory
    if vlm_factory is None:
        def vlm_factory(model: str):
            from sentinel.vlm_client import ContextVLM
            return ContextVLM(model, vlm_base_url, timeout_s=vlm_timeout_s)
    model_lister = model_lister or (lambda: list_served_models(vlm_base_url))
    if llm_factory is None:
        def llm_factory():
            from sentinel.vlm_client import _openai_client
            return _openai_client(vlm_base_url, vlm_timeout_s)
    if forecast_fn is None:
        from sentinel.cloud import fetch_forecast as forecast_fn
    if counties is None:
        cpath = assets_dir / "web" / "data" / "ca_counties.geojson"
        counties = json.loads(cpath.read_text()) if cpath.exists() else {"features": []}

    store = IncidentStore(str(state_dir / "incidents.db"))
    data_dir = Path(data_dir) if data_dir else None
    winds = WindService(sensor.read if sensor else None, state_dir / "forecast_cache.json", clock=clock,
                        sensor_label="emulated on-site sensor (replaying real HPWREN readings)"
                        if sensor else "on-site sensor")
    roots = [Path(figlib_dir) if figlib_dir else assets_dir / "figlib"] + ([data_dir / "demo"] if data_dir else [])
    samples = scan_samples(roots, data_dir / "benign" if data_dir else None, cameras,
                           data_dir / "dfire" / "test" if data_dir else None)
    recordings = scan_recordings(roots, cameras)
    frames_dir = state_dir / "frames"
    state = {"online": start_online}
    watches: dict[str, WatchSession] = {}
    watch_lock = threading.Lock()
    stats = {"frames": 0, "ignored": 0, "vlm_calls": 0, "vlm_tokens": 0, "image_bytes_seen": 0,
             "bytes_up": 0, "alerts_sent": 0, "alerts_queued": 0, "forecast_fetches": 0}
    stats_lock = threading.Lock()
    gpu_lock = threading.Lock()
    cache: dict = {}

    def bump(**kw):
        with stats_lock:
            for k, v in kw.items():
                stats[k] += v

    def detector(kind: str = "tower"):
        """Tower frames use the tower-tuned model; close-up photos (field reports) the D-Fire model,
        which is far better on them (D-Fire mAP50 0.787 vs 0.106 for the tower model)."""
        weights = photo_detector_weights if kind == "photo" and photo_detector_weights and \
            Path(photo_detector_weights).exists() else detector_weights
        try:
            mtime = Path(weights).stat().st_mtime
        except OSError:
            mtime = None
        key = (weights, mtime)          # a retrained model written to the same path is picked up automatically
        if key not in cache:
            for k in [k for k in cache if isinstance(k, tuple) and k[0] == weights]:
                del cache[k]
            if mtime is not None and cache.get("loaded", {}).get(weights) not in (None, mtime):
                log.info("detector weights changed on disk, reloading %s", weights)
            cache[key] = detector_factory(weights)
            cache.setdefault("loaded", {})[weights] = mtime
        return cache[key], Path(weights).stem

    def vlm():
        if "vlm" not in cache:
            cache["vlm"] = vlm_factory(vlm_model)
        return cache["vlm"]

    def forecast_key(cam: dict | None, lat: float, lon: float) -> str:
        return cam["id"] if cam else f"{lat:.2f},{lon:.2f}"

    def cached_forecast(key: str):
        def fetch(lat, lon):
            fc = forecast_fn(lat, lon)
            bump(forecast_fetches=1)
            winds.remember_forecast(key, fc)
            return fc
        return fetch

    def do_escalate(inc_like: dict, key: str) -> dict:
        esc = escalate(inc_like, state["online"], forecast_fn=cached_forecast(key))
        if esc["decision"] == "sent":
            bump(bytes_up=esc["bytes_up"], alerts_sent=1)
        elif esc["decision"] == "queued":
            bump(alerts_queued=1)
        esc["at"] = clock()
        return esc

    def flush_outbox() -> int:
        """Link is back: send every queued ALERT, oldest first."""
        sent = 0
        for inc in sorted(store.all(), key=lambda i: i["created_at"]):
            if (inc.get("escalation") or {}).get("decision") != "queued" or not inc.get("report"):
                continue
            esc = do_escalate({"severity": "ALERT", "report": inc["report"]},
                              forecast_key(cameras.get(inc.get("camera_id")), inc["lat"], inc["lon"]))
            if esc["decision"] == "sent":
                bump(alerts_queued=-1)
                sent += 1
            inc["escalation"] = {**esc, "queued_at": inc["escalation"].get("at")}
            store.save(inc)
        return sent

    # ------------------------------------------------------------ views

    def station_near(lat: float, lon: float) -> tuple[dict | None, float | None]:
        """Nearest tower with a weather station, within STATION_MAX_KM (a field report has no station of its own)."""
        best = min(((haversine_km(lat, lon, c["lat"], c["lon"]), c) for c in cameras.values() if c.get("wx_station")),
                   key=lambda x: x[0], default=(None, None))
        return (best[1], best[0]) if best[0] is not None and best[0] <= STATION_MAX_KM else (None, None)

    def wind_at(cam: dict | None, lat: float, lon: float, manual: dict | None = None) -> dict | None:
        src, km = (cam, 0.0) if cam and cam.get("wx_station") else station_near(lat, lon)
        w = winds.wind_for(src.get("wx_station") if src else None, forecast_key(cam, lat, lon), manual)
        if w and src is not None and src is not cam and w.get("source") == "sensor":
            w = {**w, "label": f"{w['label']}, nearest tower station ({src['site'].split(',')[0]}, {km:.0f} km)"}
        return w

    def key_frame(inc: dict) -> int | None:
        """The frame to show first: the strongest smoke frame the VLM analysed, else the strongest smoke
        frame, else the latest. (The latest is often after the plume has gone.)"""
        frames = inc.get("frames") or []
        if not frames:
            return None
        for pool in ([f for f in frames if f.get("smoke") and f.get("analysis", {}).get("context")],
                     [f for f in frames if f.get("smoke")]):
            if pool:
                return max(pool, key=lambda f: (f.get("conf") or 0, f["n"]))["n"]
        return frames[-1]["n"]

    def wind_of(inc: dict) -> dict | None:
        return wind_at(cameras.get(inc.get("camera_id")), inc["lat"], inc["lon"], inc.get("manual_wind"))

    def incident_view(inc: dict, hours: float, full: bool = False) -> dict:
        cam = cameras.get(inc.get("camera_id"))
        wind = wind_of(inc)
        drop = {"report", "thumbnail_b64"} | (set() if full else {"history", "timeline", "updates", "wind_at_report",
                                                                   "frames"})
        v = {k: val for k, val in inc.items() if k not in drop}
        v["last_update"] = (inc.get("updates") or [{}])[-1].get("text")
        v["n_updates"] = len(inc.get("updates") or [])
        v["watching"] = any(inc["id"] in w.incident_ids and w.state == "running" for w in watches.values())
        v["n_frames"] = len(inc.get("frames") or [])
        v["key_frame"] = key_frame(inc)
        v["wind"] = wind
        v["cone"] = None
        if wind and wind.get("speed_mps") is not None:
            v["cone"] = downwind_cone(inc["lat"], inc["lon"], wind["dir_from_deg"], wind["speed_mps"],
                                      hours, wind.get("spread_deg", 20.0))
        sights = list((inc.get("sightings") or {}).values())
        if not sights and cam and inc.get("bearing_lo") is not None:
            sights = [{"camera_id": cam["id"], "lat": cam["lat"], "lon": cam["lon"],
                       "bearing_lo": inc["bearing_lo"], "bearing_hi": inc["bearing_hi"]}]
        v["wedges"] = []
        for sg in sights:
            lo, hi = sg["bearing_lo"], sg["bearing_hi"]
            if (hi - lo) % 360 < 2:                    # keep a sliver visible for tiny boxes
                lo, hi = lo - 1, hi + 1
            v["wedges"].append(sector(sg["lat"], sg["lon"], lo, hi, WEDGE_KM))
        v["cameras_seeing"] = [sg.get("camera_name") or sg["camera_id"] for sg in sights]
        return v

    def camera_view(cam: dict) -> dict:
        half = cam["hfov_deg"] / 2
        return {**cam, "fov": sector(cam["lat"], cam["lon"], cam["azimuth_deg"] - half,
                                     cam["azimuth_deg"] + half, CAMERA_VIEW_KM),
                "wind": winds.sensor(cam.get("wx_station"))}

    def summary(incs: list[dict]) -> dict:
        active = [i for i in incs if i["status"] == "active"]
        by = {s: sum(1 for i in active if i["severity"] == s) for s in ("ALERT", "MONITOR", "LOG")}
        queued = sum(1 for i in incs if (i.get("escalation") or {}).get("decision") == "queued")
        with stats_lock:
            s = dict(stats)
        return {**s, "incidents_total": len(incs), "active": len(active), "active_fires": by["ALERT"] + by["MONITOR"],
                "by_severity": by, "outbox": queued, "counties": sorted({i["county"] for i in active if i.get("county")})}

    # ------------------------------------------------------------ the report pipeline

    def locate(cam: dict | None, gps, form_ll, best, width: int) -> dict:
        if gps:
            return {"lat": gps[0], "lon": gps[1], "loc_source": "exif_gps",
                    "loc_note": "GPS tags in the image"}
        if form_ll:
            return {"lat": form_ll[0], "lon": form_ll[1], "loc_source": "reported_gps",
                    "loc_note": "coordinates sent with the image"}
        if cam and best is not None:
            lo, mid, hi = box_bearings(cam["azimuth_deg"], cam["hfov_deg"], best["box"], width)
            lat, lon = destination(cam["lat"], cam["lon"], mid, range_km)
            return {"lat": lat, "lon": lon, "loc_source": "camera_bearing", "bearing_deg": mid,
                    "bearing_lo": lo, "bearing_hi": hi, "range_km": range_km,
                    "loc_note": f"bearing {mid:.0f}° ({compass(mid)}) measured from {cam['name']}; "
                                f"distance assumed {range_km:g} km, drag the pin to correct"}
        return {"lat": cam["lat"], "lon": cam["lon"], "loc_source": "camera_site",
                "loc_note": f"no box to take a bearing from: placed at {cam['name']}"}

    def observation(cam: dict | None, loc: dict, severity: str, ctx: dict | None, report: dict, thumb: str,
                    conf: float | None, info: dict, vlm_status: str | None, name: str | None) -> dict:
        """One sighting, ready for the store: location, county, wind and an enriched dispatch report."""
        county = county_at(loc["lat"], loc["lon"], counties)
        ctx = ctx or {}
        wind = wind_at(cam, loc["lat"], loc["lon"])
        report = dict(report)
        report.update(lat=round(loc["lat"], 5), lon=round(loc["lon"], 5), location_source=loc["loc_source"],
                      camera_lat=cam["lat"] if cam else None, camera_lon=cam["lon"] if cam else None,
                      county=county, bearing_deg=loc.get("bearing_deg"),
                      wind_on_site=({k: wind.get(k) for k in ("dir_from_deg", "speed_mps", "gust_mps", "source",
                                                              "emulated", "observed_at")} if wind else None))
        report["report_text"] = render_text(report)
        return {**loc, "county": county, "camera_id": cam["id"] if cam else None,
                "camera_name": cam["name"] if cam else None,
                "site_short": (cam["site"].split(",")[0] if cam else (county or "Field")),
                "severity": severity, "source_type": ctx.get("source_type") or "unknown",
                "description": ctx.get("description"), "context": ctx or None, "conf": conf,
                "frame_name": info.get("name"), "frame_offset_s": info.get("offset_s"), "name": name,
                "vlm_status": vlm_status, "thumbnail_b64": thumb, "wind_at_report": wind,
                "report": report, "report_text": report["report_text"]}

    def commit(obs: dict, now: float) -> tuple[dict, bool]:
        """Store a sighting; escalate only the first time an incident reaches ALERT."""
        prior = store.find_match(obs, now)
        prior_rank = SEVERITY_RANK.get(prior["severity"], 0) if prior else -1
        if obs["severity"] == "ALERT" and prior_rank < SEVERITY_RANK["ALERT"]:
            obs["escalation"] = do_escalate({"severity": "ALERT", "report": obs["report"]},
                                            forecast_key(cameras.get(obs.get("camera_id")), obs["lat"], obs["lon"]))
        elif prior is None:
            obs["escalation"] = {"decision": "logged", "bytes_up": 0, "at": now,
                                 "note": "Kept on the device; only ALERTs leave the tower."}
        return store.record(obs, now)

    def process(image: np.ndarray, data: bytes, info: dict, cam: dict | None, form_ll,
                name: str | None, force_vlm: bool, demo: dict | None = None) -> dict:
        gps = exif_gps(data)
        if cam is None and not gps and not form_ll:
            raise HTTPException(400, "choose a camera, or send the image with lat/lon (or GPS tags)")
        now = clock()
        tower = Tower(cam["id"], cam["name"], cam["lat"], cam["lon"], "") if cam else \
            Tower("field", "Field report", *(gps or form_ll), "")
        served = served_models()
        det, det_name = detector("tower" if cam and info.get("kind") != "photo" else "photo")
        with gpu_lock:
            r = run_pass(image, det, vlm() if vlm_model in served else None, tower,
                         min_conf=min_conf, force_vlm=force_vlm, now=now)
        bump(frames=1, image_bytes_seen=len(data), vlm_calls=int(r["vlm_calls"] or 0), vlm_tokens=int(r["tokens"] or 0))
        result = {k: r[k] for k in ("severity", "best", "detections", "vlm_status", "vlm_ms", "detect_ms",
                                     "tokens", "context", "width", "height")}
        result["vlm_served"] = vlm_model in served
        result["thumbnail_b64"] = r["thumbnail_b64"]
        if r["severity"] == "IGNORE":
            bump(ignored=1)
            return {"image": info, "result": result, "incident": None,
                    "note": "No smoke above the gate: counted on the device, nothing stored or sent."}
        loc = locate(cam, gps, form_ll, r["best"], r["width"])
        obs = observation(cam, loc, r["severity"], r["context"], r["report"], r["thumbnail_b64"],
                          (r["best"] or {}).get("conf"), info, r["vlm_status"], name)
        inc, created = commit(obs, now)
        boxes = [SimpleNamespace(box=d["box"], conf=d["conf"], cls=d.get("cls", "smoke")) for d in r["detections"]]
        analysis = {k: r.get(k) for k in ("vlm_status", "vlm_ms", "tokens", "tokens_in", "tokens_out", "sent_size",
                                          "crop_image_tokens", "context", "vlm_input")}
        analysis.update(severity=r["severity"], report_text=inc.get("report_text"), escalation=inc.get("escalation"),
                        why="single photo reported: detector, then the VLM on the strongest box")
        n = save_frame(inc, image, boxes, {"t": now, "offset_s": info.get("offset_s"), "conf": (r["best"] or {}).get("conf", 0),
                                           "smoke": r["best"] is not None, "camera_id": cam["id"] if cam else None,
                                           "camera_name": cam["name"] if cam else "field report",
                                           "detect_ms": round(r["detect_ms"], 1), "source": info.get("name"),
                                           "detector": det_name,
                                           "analysis": analysis})
        if demo:
            inc["demo"] = demo
        inc.setdefault("updates", []).append({"t": now, "kind": "report", "severity": r["severity"],
                                              "source_type": obs["source_type"], "trend": None,
                                              "text": f"single frame reported ({info.get('name') or 'upload'}) · "
                                                      f"{obs['source_type'].replace('_', ' ')} · {r['severity']}"})
        store.save(inc)
        return {"image": info, "result": result, "incident": incident_view(inc, 1.0), "created": created, "frame_n": n}

    # ------------------------------------------------------------ continuous monitoring (watch)

    def served_models() -> list[str]:
        try:
            return model_lister()
        except Exception:  # noqa: BLE001 - VLM down: the detector result still counts
            log.warning("could not list served models")
            return []

    def classify_crop(image: np.ndarray, box) -> tuple[dict | None, int, dict]:
        """VLM on the detection crop. Returns (context, tokens, analysis record for the frame viewer)."""
        if vlm_model not in served_models():
            return None, 0, {"vlm_status": "unavailable"}
        crop = crop_box(image, box, 0.5, 448)
        t0 = time.perf_counter()
        model = vlm()
        ctx, tokens = model.classify(to_jpeg(crop))
        ms = (time.perf_counter() - t0) * 1000
        bump(vlm_calls=1, vlm_tokens=int(tokens or 0))
        usage = getattr(model, "last_usage", None) or {}
        ch, cw = crop.shape[:2]
        rec = {"vlm_status": "ok" if ctx is not None else "failed", "vlm_ms": ms, "tokens": int(tokens or 0),
               "tokens_in": usage.get("prompt_tokens"), "tokens_out": usage.get("completion_tokens"),
               "sent_size": [cw, ch], "crop_image_tokens": qwen_image_tokens(cw, ch),
               "context": ctx.model_dump() if ctx is not None else None}
        return rec["context"], int(tokens or 0), rec

    def draw_boxes(out: np.ndarray, dets, scale: float) -> None:
        """Boxes with "smoke 0.79" labels: solid red above the gate, thin grey below it."""
        H, W = out.shape[:2]
        thick = max(2, int(round(max(H, W) / 450)))
        font = max(0.45, max(H, W) / 1800)
        for d in sorted(dets, key=lambda d: d.conf):
            x1, y1, x2, y2 = (int(v * scale) for v in d.box)
            strong = d.conf >= min_conf
            col = (40, 40, 220) if strong else (170, 170, 170)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thick if strong else max(1, thick - 1))
            label = f"{getattr(d, 'cls', 'smoke')} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
            ty = max(th + 4, y1 - 3)
            cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), col, -1)
            cv2.putText(out, label, (x1 + 3, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), 1, cv2.LINE_AA)

    def save_frame(inc: dict, img: np.ndarray, dets, meta: dict) -> int | None:
        """Keep a viewable copy (boxes drawn, <= FRAME_SIDE px) of a frame in the incident's filmstrip."""
        frames = inc.setdefault("frames", [])
        if len(frames) >= FRAMES_CAP:
            return None
        h, w = img.shape[:2]
        scale = min(1.0, FRAME_SIDE / max(h, w))
        out = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img.copy()
        draw_boxes(out, dets, scale)
        n = len(frames)
        d = frames_dir / inc["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{n}.jpg").write_bytes(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
        frames.append({"n": n, "w": w, "h": h, "dets": [
            {"cls": getattr(d, "cls", "smoke"), "conf": round(float(d.conf), 3), "box": [round(float(v), 1) for v in d.box],
             "strong": d.conf >= min_conf} for d in sorted(dets, key=lambda d: -d.conf)][:12],
            "full_frame_image_tokens": cloud_frame_tokens(w, h), **meta})
        return n

    def relocate(inc: dict) -> None:
        """Two or more towers see it: put the pin where their bearing lines cross."""
        if inc.get("loc_source") in ("map_pin", "exif_gps", "reported_gps"):
            return
        sights = list((inc.get("sightings") or {}).values())
        fix = triangulate(sights) if len(sights) >= 2 else None
        if fix is None:
            return
        inc.update(lat=fix["lat"], lon=fix["lon"], loc_source="triangulated",
                   county=county_at(fix["lat"], fix["lon"], counties),
                   loc_note=f"bearings from {fix['n']} towers cross here "
                            f"(lines agree within {fix['spread_km']:.1f} km)", triangulation=fix)

    def same_fire(a: dict, b: dict) -> bool:
        """Do two active incidents describe one fire? Their triangulated fixes agree, or one's tower line
        passes (in front of the tower) within CROSS_AGREE_KM of the other's fix."""
        if abs(a["updated_at"] - b["updated_at"]) > CROSS_WINDOW_S:
            return False
        fixed = [x for x in (a, b) if x.get("loc_source") == "triangulated"]
        if not fixed:                   # one tower each: their current lines must cross in front of both
            for sa in (a.get("sightings") or {}).values():
                for sb in (b.get("sightings") or {}).values():
                    if sa["camera_id"] != sb["camera_id"] and ray_intersection(
                            sa["lat"], sa["lon"], sa["bearing_deg"], sb["lat"], sb["lon"], sb["bearing_deg"], CROSS_MAX_KM):
                        return True
            return False
        if len(fixed) == 2:
            return haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) <= CROSS_AGREE_KM
        if len(fixed) == 1:
            fix, other = fixed[0], (b if fixed[0] is a else a)
            for sg in (other.get("sightings") or {}).values():
                along, off = line_offset_km(sg["lat"], sg["lon"], sg["bearing_deg"], fix["lat"], fix["lon"])
                if 0.2 < along <= CROSS_MAX_KM and off <= CROSS_AGREE_KM:
                    return True
        return False

    def merge_incidents(keep: dict, other: dict) -> dict:
        """Fold `other` into `keep`: towers, frames (files renumbered), timeline, updates, strongest severity."""
        src, dst = frames_dir / other["id"], frames_dir / keep["id"]
        dst.mkdir(parents=True, exist_ok=True)
        frames = keep.setdefault("frames", [])
        for f in other.get("frames") or []:
            if len(frames) >= FRAMES_CAP:
                break
            old = src / f"{f['n']}.jpg"
            if old.exists():
                n = len(frames)
                old.replace(dst / f"{n}.jpg")
                frames.append({**f, "n": n})
        shutil.rmtree(src, ignore_errors=True)
        keep["sightings"] = {**(other.get("sightings") or {}), **(keep.get("sightings") or {})}
        for k in ("updates", "timeline", "history"):
            keep[k] = sorted((keep.get(k) or []) + (other.get(k) or []), key=lambda u: u.get("t", u.get("at", 0)))
        keep["updates"] = keep["updates"][-UPDATES_CAP:]
        keep["timeline"] = keep["timeline"][-TIMELINE_CAP:]
        keep["detections"] = keep.get("detections", 0) + other.get("detections", 0)
        keep["created_at"] = min(keep["created_at"], other["created_at"])
        keep["updated_at"] = max(keep["updated_at"], other["updated_at"])
        if SEVERITY_RANK.get(other["severity"], 0) > SEVERITY_RANK.get(keep["severity"], 0):
            for k in ("severity", "context", "source_type", "description", "report", "report_text"):
                if other.get(k) is not None:
                    keep[k] = other[k]
        rank = {"sent": 3, "queued": 2}
        if rank.get((other.get("escalation") or {}).get("decision"), 0) > rank.get((keep.get("escalation") or {}).get("decision"), 0):
            keep["escalation"] = other["escalation"]
        keep.setdefault("updates", []).append({"t": keep["updated_at"], "kind": "merged",
                                               "text": f"merged with {other['name']} #{other['seq']}: the same fire seen by more towers"})
        relocate(keep)
        return keep

    def consolidate(inc: dict) -> dict:
        """After an incident's position changes, fold in any other active incident that is the same fire."""
        for other in store.all():
            if other["id"] == inc["id"] or other["status"] != "active" or not same_fire(inc, other):
                continue
            keep, gone = (inc, other) if (len(inc.get("sightings") or {}), -inc["created_at"]) >= \
                (len(other.get("sightings") or {}), -other["created_at"]) else (other, inc)
            keep = merge_incidents(keep, gone)
            store.save(keep)
            store.delete(gone["id"])
            for w in watches.values():
                for c in w.cams.values():
                    if c.incident_id == gone["id"]:
                        c.incident_id = keep["id"]
                w.incident_ids = list(dict.fromkeys(keep["id"] if i == gone["id"] else i for i in w.incident_ids))
            inc = keep
        return inc

    def sighting(cam: dict, best, width: int, t: float) -> dict:
        lo, mid, hi = box_bearings(cam["azimuth_deg"], cam["hfov_deg"], best.box, width)
        return {"camera_id": cam["id"], "camera_name": cam["name"], "lat": cam["lat"], "lon": cam["lon"],
                "bearing_deg": mid, "bearing_lo": lo, "bearing_hi": hi, "t": t}

    def watch_step(s: WatchSession) -> None:
        fr = s.frames[s.idx]
        s.idx += 1
        img = cv2.imread(str(fr.path), cv2.IMREAD_COLOR)
        if img is None:
            return
        cam = cameras[fr.camera_id]
        c = s.cams[fr.camera_id]
        t = fr.epoch if s.time_mode == "recorded" and fr.epoch else clock()
        t0 = time.perf_counter()
        with gpu_lock:
            det, det_name = detector("tower")
            dets = list(det(img))
        detect_ms = (time.perf_counter() - t0) * 1000
        h, w = img.shape[:2]
        strong = [d for d in dets if d.conf >= min_conf]
        best = max(strong, key=lambda d: d.conf, default=None)
        top = max(dets, key=lambda d: d.conf, default=None)
        point = {"t": t, "offset_s": fr.offset_s, "conf": round(top.conf, 3) if top else 0.0,
                 "area": (best.area / (w * h)) if best else 0.0, "camera_id": cam["id"]}
        c.points.append(point)
        c.frames_done += 1
        s.frames_done += 1
        c.latest_offset = fr.offset_s
        c.latest_thumb = _thumbnail(img, dets, min_conf)
        bump(frames=1, image_bytes_seen=fr.path.stat().st_size, ignored=0 if best else 1)
        if best is not None:
            c.last_hit = (img, best)
        meta = {"t": t, "offset_s": fr.offset_s, "conf": point["conf"], "smoke": best is not None,
                "camera_id": cam["id"], "camera_name": cam["name"], "detect_ms": round(detect_ms, 1),
                "source": f"{fr.path.parent.name}/{fr.path.name}", "detector": det_name}
        info = {"name": f"{fr.path.parent.name}/{fr.path.name}", "offset_s": fr.offset_s}

        if c.incident_id is None:
            c.recent = (c.recent + [(img, dets, meta)])[-gate_frames:]
            c.streak = c.streak + 1 if best else 0
            if c.streak < gate_frames:
                return
            ctx, tokens, rec = classify_crop(img, best.box)
            s.vlm_calls += 1 if tokens else 0
            s.vlm_tokens += tokens
            ctx_obj = ContextResult(**ctx) if ctx else None
            sev = assess(ctx_obj, None) if ctx_obj else fallback_severity(None)
            meta["analysis"] = {**rec, "severity": sev.name, "why": f"smoke held for {gate_frames} frames in a row: "
                                                                      "the VLM was called once to classify it"}
            tower = Tower(cam["id"], cam["name"], cam["lat"], cam["lon"], "")
            report = build_report(uuid.uuid4().hex[:12], tower, sev, ctx_obj, None, best.conf, t, c.latest_thumb)
            loc = locate(cam, None, None, {"box": best.box}, w)
            obs = observation(cam, loc, sev.name, ctx, report, c.latest_thumb, best.conf, info,
                              "ok" if ctx else "unavailable", s.fire_name)
            obs["watch_session"] = s.id
            obs["sighting"] = sighting(cam, best, w, t)
            inc, created = commit(obs, t)
            joined = not created and inc.get("camera_id") != cam["id"]
            inc.setdefault("updates", []).append({
                "t": t, "offset_s": fr.offset_s, "kind": "opened" if created else "joined", "severity": sev.name,
                "trend": None, "camera_id": cam["id"], "source_type": obs["source_type"], "vlm_called": bool(ctx),
                "tokens": tokens,
                "text": (f"{cam['name']}: plume held for {gate_frames} frames: "
                         + ("incident opened" if created else "second tower sees it, bearings cross" if joined
                            else "same plume again")
                         + f" · {obs['source_type'].replace('_', ' ')} · {sev.name}")})
            inc["detections"] += gate_frames - 1          # record() counted one; the gate saw gate_frames
            inc["timeline"] = ((inc.get("timeline") or []) + c.points[-gate_frames:])[-TIMELINE_CAP:]
            meta["analysis"].update(report_text=inc.get("report_text"), escalation=inc.get("escalation"))
            for im, ds, mt in c.recent:
                n = save_frame(inc, im, ds, mt)
            c.last_hit_n = n
            if len(inc.get("sightings") or {}) >= 2:
                before = inc.get("loc_source")
                relocate(inc)
                if inc.get("loc_source") == "triangulated" and before != "triangulated":
                    inc["updates"][-1]["text"] += " · location triangulated"
            store.save(inc)
            inc = consolidate(inc)
            c.incident_id = inc["id"]
            if inc["id"] not in s.incident_ids:
                s.incident_ids.append(inc["id"])
            c.prev_batch, c.batch, c.recent = c.points[-gate_frames:], [], []
            return

        c.batch.append(point)
        inc = store.get(c.incident_id)
        if inc is None:             # reset while watching
            s.stop.set()
            return
        inc["timeline"] = ((inc.get("timeline") or []) + [point])[-TIMELINE_CAP:]
        n = save_frame(inc, img, dets, meta)
        if best is not None and n is not None:
            c.last_hit_n = n
        if len(c.batch) < s.batch_frames:
            store.save(inc)
            return
        summ = summarize_batch(c.prev_batch, c.batch, min_conf)
        ctx, tokens, rec = None, 0, None
        if summ["detected_in"] and c.last_hit is not None:
            ctx, tokens, rec = classify_crop(c.last_hit[0], c.last_hit[1].box)
            s.vlm_calls += 1 if tokens else 0
            s.vlm_tokens += tokens
            # keep this tower's bearing current as the plume moves in the frame
            inc.setdefault("sightings", {})[cam["id"]] = sighting(cam, c.last_hit[1], c.last_hit[0].shape[1], t)
            relocate(inc)
            store.save(inc)
            own_id = inc["id"]
            inc = consolidate(inc)
            if inc["id"] != own_id:          # folded into another incident: our frame numbers changed
                c.last_hit_n = None
        ctx_obj = ContextResult(**ctx) if ctx else (ContextResult(**inc["context"]) if inc.get("context") else None)
        trend = summ["trend"] if summ["trend"] != "not_seen" else None
        sev = (assess(ctx_obj, trend) if ctx_obj else fallback_severity(trend)).name
        upgraded = SEVERITY_RANK[sev] > SEVERITY_RANK.get(inc["severity"], 0)
        update = {**summ, "t": t, "offset_s": fr.offset_s, "kind": "batch", "severity": sev, "camera_id": cam["id"],
                  "source_type": (ctx or {}).get("source_type"), "vlm_called": bool(ctx), "tokens": tokens}
        update["text"] = (f"{cam['name']}: " if len(inc.get("sightings") or {}) > 1 else "") + update_text(update)
        if upgraded:
            inc["severity"] = sev
            if ctx:
                inc.update(context=ctx, source_type=ctx["source_type"], description=ctx.get("description"))
            if sev == "ALERT" and (inc.get("escalation") or {}).get("decision") not in ("sent", "queued"):
                rep = dict(inc["report"])
                rep.update(severity="ALERT", trend=trend, lat=round(inc["lat"], 5), lon=round(inc["lon"], 5),
                           location_source=inc.get("loc_source"),
                           detected_at=datetime.fromtimestamp(t, tz=timezone.utc).isoformat())
                rep["report_text"] = render_text(rep)
                inc["report"], inc["report_text"] = rep, rep["report_text"]
                inc["escalation"] = do_escalate({"severity": "ALERT", "report": rep},
                                                forecast_key(cam, inc["lat"], inc["lon"]))
            update["text"] += f" (raised to {sev})"
        if rec is not None and c.last_hit_n is not None and c.last_hit_n < len(inc.get("frames") or []):
            inc["frames"][c.last_hit_n]["analysis"] = {
                **rec, "severity": sev, "trend": summ["trend"], "batch": update["text"],
                "why": f"situation update after {summ['frames']} frames: the VLM re-checked the latest smoke crop",
                "report_text": inc.get("report_text"), "escalation": inc.get("escalation") if upgraded else None}
        if summ["detected_in"]:
            inc["thumbnail_b64"] = _thumbnail(c.last_hit[0], [c.last_hit[1]], min_conf)
        inc["detections"] = inc.get("detections", 0) + summ["detected_in"]
        inc["trend"] = summ["trend"]
        inc["updated_at"] = max(inc["updated_at"], t)
        inc["updates"] = ((inc.get("updates") or []) + [update])[-UPDATES_CAP:]
        store.save(inc)
        c.prev_batch, c.batch = c.batch, []

    def watch_run(s: WatchSession) -> None:
        starts = set(time_steps(s.frames))
        try:
            while not s.stop.is_set() and s.idx < len(s.frames):
                if s.interval_s and s.idx in starts and s.idx:
                    if s.stop.wait(s.interval_s):
                        break
                watch_step(s)
            s.state = "stopped" if s.stop.is_set() else "done"
            if s.state == "done" and s.time_mode == "recorded":
                for iid in s.incident_ids:          # a replayed recording is history, not a live fire
                    inc = store.get(iid)
                    if inc:
                        inc["status"] = "resolved"
                        inc.setdefault("updates", []).append({"t": inc["updated_at"], "kind": "closed",
                                                              "text": "recording ended: archived as history"})
                        store.save(inc)
        except Exception as exc:  # noqa: BLE001
            log.exception("watch %s failed", s.id)
            s.state, s.error = "error", f"{type(exc).__name__}: {exc}"[:300]

    def watch_many(sessions: list[WatchSession]) -> None:
        for s in sessions:
            if s.stop.is_set():
                s.state = "stopped"
                continue
            watch_run(s)

    def handle_report(body: bytes, ctype: str) -> dict:
        kind = ctype.split(";", 1)[0].strip().lower()
        if kind == "multipart/form-data":
            try:
                fields, files = parse_multipart(body, ctype)
            except Exception:  # noqa: BLE001
                raise HTTPException(400, "malformed multipart body") from None
        elif kind == "application/x-www-form-urlencoded":
            fields, files = parse_qs(body.decode("utf-8", "replace")), {}
        else:
            raise HTTPException(415, "send multipart/form-data")

        def field(n, default=""):
            return (fields.get(n) or [default])[0].strip()

        cam_id = field("camera_id")
        cam = cameras.get(cam_id) if cam_id else None
        if cam_id and cam is None:
            raise HTTPException(400, f"unknown camera {cam_id!r}")
        form_ll = None
        if field("lat") or field("lon"):
            try:
                form_ll = (float(field("lat")), float(field("lon")))
            except ValueError:
                raise HTTPException(400, "lat/lon must be numbers") from None
            if not valid_latlon(*form_ll):
                raise HTTPException(400, "lat/lon out of range")
        upload = files.get("image")
        if upload is not None and upload[1]:
            filename, data = upload
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "image larger than 15 MB")
            info = {"source": "upload", "name": Path(filename).name[:120], "kind": "photo"}
        elif field("sample_id"):
            s = samples.get(field("sample_id")) if _ID_RE.match(field("sample_id")) else None
            if s is None:
                raise HTTPException(404, "unknown sample")
            data = s["path"].read_bytes()
            info = {"source": "sample", "name": s["name"], "offset_s": s["offset_s"],
                    "kind": "photo" if s["kind"] == "photo" else "tower"}
            if cam is None and s["camera_id"] and not form_ll:
                cam = cameras[s["camera_id"]]
        else:
            raise HTTPException(400, "attach an image or choose a sample")
        image = decode_image(data)
        info.update(width=int(image.shape[1]), height=int(image.shape[0]), bytes=len(data))
        force = field("force_vlm").lower() in ("1", "true", "on", "yes")
        return process(image, data, info, cam, form_ll, field("name")[:80] or None, force)

    # ------------------------------------------------------------ demo scenarios

    demo = {"state": "idle", "results": {}, "error": None}
    scenarios = json.loads(Path(demo_scenarios).read_text()) if demo_scenarios and Path(demo_scenarios).exists() else []

    def run_demo() -> None:
        """Mock inputs, real analysis: each photo goes through the detector + VLM + rules on this device."""
        try:
            for sc in scenarios:
                res = demo["results"][sc["id"]] = {"status": "running"}
                if sc["kind"] == "photo":
                    path = data_dir / sc["image"] if data_dir else Path(sc["image"])
                    data = path.read_bytes()
                    image = decode_image(data)
                    info = {"source": "demo", "name": path.name, "kind": "photo",
                            "width": int(image.shape[1]), "height": int(image.shape[0]), "bytes": len(data)}
                    r = process(image, data, info, None, (sc["lat"], sc["lon"]), sc.get("name"), False,
                                demo={"id": sc["id"], "title": sc["title"], "expect": sc["expect"]})
                    inc = r["incident"]
                    rr = r["result"]
                    res.update(status="done", severity=rr["severity"], detect_ms=rr["detect_ms"],
                               best=(rr["best"] or {}).get("conf"), tokens=rr["tokens"],
                               thumb_b64=None if inc else rr["thumbnail_b64"],
                               source_type=(rr["context"] or {}).get("source_type"),
                               incident_id=inc["id"] if inc else None, frame_n=r.get("frame_n"),
                               escalation=((inc or {}).get("escalation") or {}).get("decision"))
                elif sc["kind"] == "watch":
                    g = recordings.get(sc["recording"])
                    if g is None:
                        res.update(status="skipped", note="recording not on this device")
                        continue
                    out = start_watch(WatchRequest(recording=sc["recording"], interval_s=sc.get("interval_s", 1.0),
                                                   batch_frames=sc.get("batch_frames", 5), time_mode="live",
                                                   start_offset_s=sc.get("start_offset_s", -300)))
                    res.update(status="watching", session=out["sessions"][0]["id"])
                else:
                    res.update(status="ready")
            demo["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            log.exception("demo load failed")
            demo["state"], demo["error"] = "error", f"{type(exc).__name__}: {exc}"[:300]

    def demo_view() -> dict:
        out = []
        for sc in scenarios:
            res = dict(demo["results"].get(sc["id"]) or {"status": "not loaded"})
            if res.get("session") in watches:
                w = watches[res["session"]]
                res.update(state=w.state, progress=f"{w.idx}/{len(w.frames)}", incident_ids=w.incident_ids)
                incs = [store.get(i) for i in w.incident_ids]
                incs = [i for i in incs if i]
                if incs:
                    res.update(incident_id=incs[0]["id"], severity=incs[0]["severity"],
                               escalation=(incs[0].get("escalation") or {}).get("decision"),
                               towers=len(incs[0].get("sightings") or {}), loc_source=incs[0].get("loc_source"))
            out.append({**sc, "result": res})
        return {"state": demo["state"], "error": demo["error"], "scenarios": out}

    # ------------------------------------------------------------ app

    @asynccontextmanager
    async def lifespan(app):
        stop = threading.Event()
        if sensor is not None:
            threading.Thread(target=sensor.run, args=(stop, sensor_interval_s), name="wx-sensor",
                             daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="Wildfire Edge Sentinel incident map", lifespan=lifespan)
    app.state.store = store
    app.state.winds = winds
    for mount, sub in (("/assets", assets_dir / "web"), ("/tiles", assets_dir / "maps")):
        if sub.is_dir():
            app.mount(mount, StaticFiles(directory=sub), name=mount.strip("/"))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text()

    @app.get("/api/config")
    def config():
        try:
            served, err = model_lister(), None
        except Exception as exc:  # noqa: BLE001
            served, err = [], str(exc)[:200]
        return {"map_url": f"/tiles/{MAP_FILE}", "map_available": (assets_dir / "maps" / MAP_FILE).exists(),
                "cameras": [camera_view(c) for c in cameras.values()], "online": state["online"],
                "vlm_model": vlm_model, "vlm_served": vlm_model in served, "models_error": err,
                "detector_weights": detector_weights, "detector_available": Path(detector_weights).exists(),
                "range_km": range_km, "min_conf": min_conf,
                "sensor": {"emulated": sensor is not None, "source": sensor.source if sensor else None,
                           "captured": sensor.captured if sensor else None}}

    @app.get("/api/state")
    def get_state(hours: float = 1.0):
        hours = min(max(hours, 0.25), 12.0)
        incs = store.all()
        return {"online": state["online"], "hours": hours, "generated_at": clock(), "summary": summary(incs),
                "cameras": [camera_view(c) for c in cameras.values()],
                "incidents": [incident_view(i, hours) for i in incs]}

    @app.get("/api/samples")
    def list_samples():
        return {"samples": [{k: v for k, v in s.items() if k != "path"} for s in samples.values()]}

    @app.get("/api/sample/{sample_id}")
    def get_sample(sample_id: str):
        s = samples.get(sample_id) if _ID_RE.match(sample_id) else None
        if s is None:
            raise HTTPException(404, "unknown sample")
        img = cv2.imread(str(s["path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(404, "sample unreadable")
        h, w = img.shape[:2]
        scale = min(1.0, 480 / max(h, w))
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes(),
                        media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.post("/api/report")
    async def report(request: Request):
        body = await _read_body(request, MAX_UPLOAD_BYTES + 256 * 1024)
        return await run_in_threadpool(handle_report, body, request.headers.get("content-type", ""))

    @app.get("/api/incidents/{incident_id}")
    def get_incident(incident_id: str, hours: float = 1.0):
        inc = store.get(incident_id)
        if inc is None:
            raise HTTPException(404, "unknown incident")
        return {**incident_view(inc, hours, full=True), "report_text": inc.get("report_text")}

    @app.get("/api/incidents/{incident_id}/thumb")
    def get_thumb(incident_id: str):
        import base64
        inc = store.get(incident_id)
        if inc is None or not inc.get("thumbnail_b64"):
            raise HTTPException(404, "no thumbnail")
        return Response(base64.b64decode(inc["thumbnail_b64"]), media_type="image/jpeg")

    @app.patch("/api/incidents/{incident_id}")
    def patch_incident(incident_id: str, p: IncidentPatch):
        inc = store.get(incident_id)
        if inc is None:
            raise HTTPException(404, "unknown incident")
        if (p.lat is None) != (p.lon is None):
            raise HTTPException(400, "send lat and lon together")
        if p.lat is not None:
            if not valid_latlon(p.lat, p.lon):
                raise HTTPException(400, "lat/lon out of range")
            inc.update(lat=p.lat, lon=p.lon, loc_source="map_pin", county=county_at(p.lat, p.lon, counties),
                       loc_note="placed on the map by the operator")
        if p.status is not None:
            if p.status not in ("active", "resolved"):
                raise HTTPException(400, "status must be active or resolved")
            inc["status"] = p.status
        if p.name:
            inc["name"] = p.name[:80]
        if (p.wind_from_deg is None) != (p.wind_speed_mps is None):
            raise HTTPException(400, "send wind direction and speed together")
        if p.wind_from_deg is not None:
            if p.wind_speed_mps < 0 or p.wind_speed_mps > 80:
                raise HTTPException(400, "wind speed out of range")
            inc["manual_wind"] = {"dir_from_deg": p.wind_from_deg % 360, "speed_mps": p.wind_speed_mps,
                                  "observed_at": clock()}
        inc["updated_at"] = clock()
        store.save(inc)
        return incident_view(inc, 1.0)

    @app.post("/api/network")
    def set_network(s: NetworkState):
        was = state["online"]
        state["online"] = s.online
        sent = flush_outbox() if s.online and not was else 0
        return {"online": state["online"], "flushed": sent}

    # ------------------------------------------------------------ watch endpoints

    @app.get("/api/recordings")
    def list_recordings():
        return {"recordings": [{"key": g["key"], "fire": g["fire"], "date": g["date"], "cameras": g["cameras"],
                                "camera_names": [cameras[c]["name"] for c in g["cameras"]]} for g in recordings.values()]}

    @app.post("/api/watch")
    def start_watch(req: WatchRequest):
        keys = list(recordings) if req.recording == "all" else [req.recording]
        if any(k not in recordings for k in keys):
            raise HTTPException(404, f"unknown recording {req.recording!r}")
        if req.time_mode not in ("live", "recorded"):
            raise HTTPException(400, "time_mode must be live or recorded")
        sessions = []
        with watch_lock:
            busy = {c for w in watches.values() if w.state == "running" for c in w.camera_ids}
            for key in sorted(keys, key=lambda k: recordings[k]["date"] or ""):
                g = recordings[key]
                if busy & set(g["cameras"]):
                    raise HTTPException(409, f"already watching {sorted(busy & set(g['cameras']))}")
                streams = []
                for cid in g["cameras"]:
                    fr = [f for f in list_frames(g["folders"][cid], cid)
                          if req.start_offset_s is None or f.offset_s is None or f.offset_s >= req.start_offset_s]
                    streams.append(fr)
                frames = merge_streams(streams)
                if not frames:
                    continue
                s = WatchSession(uuid.uuid4().hex[:8], f"{g['fire']} · {g['date']}" if g["date"] else g["fire"],
                                 g["cameras"], frames, max(0.0, min(req.interval_s, 60.0)),
                                 max(2, min(req.batch_frames, 30)), req.time_mode, clock(),
                                 fire_name=g["fire"] if g["fire"] and g["fire"] != "Fire" else None)
                watches[s.id] = s
                sessions.append(s)
        if not sessions:
            raise HTTPException(404, "no frames in that recording")
        threading.Thread(target=watch_many, args=(sessions,), name="watch", daemon=True).start()
        return {"sessions": [s.view() for s in sessions]}

    @app.get("/api/watch")
    def list_watches():
        return {"sessions": [w.view() for w in sorted(watches.values(), key=lambda w: -w.started_at)][:20]}

    @app.get("/api/watch/{sid}/frame/{camera_id}")
    def watch_frame(sid: str, camera_id: str):
        import base64
        w = watches.get(sid)
        c = w.cams.get(camera_id) if w else None
        if c is None or not c.latest_thumb:
            raise HTTPException(404, "no frame yet")
        return Response(base64.b64decode(c.latest_thumb), media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.delete("/api/watch/{sid}")
    def stop_watch(sid: str):
        w = watches.get(sid)
        if w is None:
            raise HTTPException(404, "unknown session")
        w.stop.set()
        return w.view()

    @app.get("/api/incidents/{incident_id}/frames/{n}")
    def incident_frame(incident_id: str, n: int, w: int | None = None):
        if not _ID_RE.match(incident_id) or n < 0:
            raise HTTPException(404, "no such frame")
        path = frames_dir / incident_id / f"{n}.jpg"
        if not path.exists():
            raise HTTPException(404, "no such frame")
        data = path.read_bytes()
        if w:                                   # small copy for list thumbnails
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            ih, iw = img.shape[:2]
            s = min(1.0, max(64, min(w, 640)) / iw)
            img = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
            data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    # ------------------------------------------------------------ assistant

    assistant = Assistant(llm_factory, assistant_model, store, wind_of, clock=clock)
    app.state.assistant = assistant

    @app.post("/api/ask")
    async def ask(q: Question):
        if not q.question.strip():
            raise HTTPException(400, "ask a question")
        return await run_in_threadpool(assistant.ask, q.question.strip()[:500], q.selected)

    @app.get("/api/demo")
    def get_demo():
        return demo_view()

    @app.post("/api/demo/load")
    def load_demo():
        if demo["state"] == "running":
            raise HTTPException(409, "demo scenarios are already loading")
        if not scenarios:
            raise HTTPException(404, "no demo scenarios configured")
        reset()
        state["online"] = False
        demo.update(state="running", results={}, error=None)
        threading.Thread(target=run_demo, name="demo", daemon=True).start()
        return demo_view()

    @app.post("/api/reset")
    def reset():
        for w in watches.values():
            w.stop.set()
        store.clear()
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)
        with stats_lock:
            for k in stats:
                stats[k] = 0
        return {"ok": True}

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--assets", default=str(DEFAULT_ASSETS), help="offline map, web libs and state (outside git)")
    ap.add_argument("--data-dir", default=str(HOME / "sentinel" / "data"), help="benign frames (+ older data/demo)")
    ap.add_argument("--figlib-dir", default=None, help="FIgLib recordings (default <assets>/figlib)")
    ap.add_argument("--cameras", default="config/cameras.json")
    ap.add_argument("--detector-weights", default=str(HOME / "sentinel" / "models" / "tower_yolo.pt"),
                    help="detector for tower camera frames")
    ap.add_argument("--photo-detector-weights", default=str(HOME / "sentinel" / "models" / "smoke_yolo.pt"),
                    help="detector for close-up photos and uploads (D-Fire model)")
    ap.add_argument("--vlm-base-url", default="unix:///opt/hp/zrt/run/vllm-base7b.sock",
                    help="the served VLM; the zrt proxy only routes the base label, so LoRA goes via the socket")
    ap.add_argument("--vlm-model", default="context")
    ap.add_argument("--vlm-timeout", type=float, default=60.0)
    ap.add_argument("--range-km", type=float, default=10.0, help="assumed distance along a camera bearing")
    ap.add_argument("--wx-seed", default=None, help="real WXT readings to emulate (default <assets>/state/wx_seed.json)")
    ap.add_argument("--no-sensor", action="store_true", help="no on-site sensor emulation")
    ap.add_argument("--start-online", action="store_true", help="start with the uplink up (default: offline)")
    ap.add_argument("--gate-frames", type=int, default=2, help="frames in a row with smoke before an incident opens")
    ap.add_argument("--demo-scenarios", default="config/demo_scenarios.json")
    ap.add_argument("--assistant-model", default="base7b", help="served model for the Ranger assistant")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed = Path(a.wx_seed or Path(a.assets) / "state" / "wx_seed.json")
    sensor = None if a.no_sensor or not seed.exists() else SensorEmulator(seed)
    app = create_map_app(cameras=load_cameras(a.cameras), assets_dir=a.assets, data_dir=a.data_dir,
                         figlib_dir=a.figlib_dir,
                         detector_weights=a.detector_weights, photo_detector_weights=a.photo_detector_weights,
                         vlm_model=a.vlm_model,
                         vlm_base_url=a.vlm_base_url, vlm_timeout_s=a.vlm_timeout, sensor=sensor,
                         range_km=a.range_km, start_online=a.start_online, gate_frames=a.gate_frames,
                         assistant_model=a.assistant_model, demo_scenarios=a.demo_scenarios)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()

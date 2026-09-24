"""Demo web app: run the BEFORE and AFTER cascades on one image side by side, and watch usage and
edge-vs-cloud economics live. `python -m sentinel.webapp`, then open http://localhost:8095
(SSH-tunnel the port from a laptop)."""
import argparse
import hashlib
import io
import json
import logging
import mimetypes
import re
import threading
import time
from contextlib import asynccontextmanager
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from sentinel.config import Tower
from sentinel.detector import imgsz_for  # noqa: F401 - re-exported; shared with the incident map
from sentinel.live_metrics import effective_prices
from sentinel.monitor import PriceUpdate, _load_prices
from sentinel.webapp_logic import TRUTHS, Session, escalate, load_benchmarks, run_pass

PAGE = Path(__file__).parent / "static" / "webapp.html"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SAMPLE_CAP = 60
SAMPLE_THUMB_SIDE = 200
MODELS_TTL_S = 10.0
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_FIGLIB_RE = re.compile(r"_([+-]\d+)\.[A-Za-z]+$")   # FIgLib: <epoch>_<offset s from ignition>.jpg

DETECTORS = {
    "yoloworld": {"label": "YOLO-World zero-shot", "weights": "yolov8s-worldv2.pt",
                  "classes": ["smoke", "fire"], "imgsz": 640},
    "yolo11s": {"label": "YOLO11s fine-tuned on D-Fire", "weights": "models/smoke_yolo.pt",
                "classes": None, "imgsz": 640},
}


def detector_specs(before_weights: str, after_weights: str, after_imgsz: int | None = None) -> dict:
    return {"yoloworld": {**DETECTORS["yoloworld"], "weights": before_weights, "imgsz": imgsz_for(before_weights)},
            "yolo11s": {**DETECTORS["yolo11s"], "weights": after_weights,
                        "imgsz": after_imgsz or imgsz_for(after_weights)}}
PIPELINES = {
    "before": {"label": "BEFORE fine-tuning", "detector": "yoloworld", "vlm": "base7b",
               "vlm_label": "Qwen2.5-VL-7B base", "optional": False},
    "after": {"label": "AFTER fine-tuning", "detector": "yolo11s", "vlm": "context",
              "vlm_label": "Qwen2.5-VL-7B + LoRA", "optional": False},
    "teacher": {"label": "Teacher reference", "detector": "yolo11s", "vlm": "teacher32b",
                "vlm_label": "Qwen2.5-VL-32B-AWQ teacher", "optional": True},
}
DEMO_TOWER = Tower("demo", "Demo Lookout", 37.1606, -121.8983, "", temp_c=25.0)
GROUPS = (("dfire", "dfire_test"), ("demo", "tower_fire_sequence"), ("benign", "benign_tower"))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------- request parsing

def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, list[str]], dict[str, tuple]]:
    """multipart/form-data -> ({field: [values]}, {field: (filename, bytes)}), stdlib only
    (python-multipart is not a dependency)."""
    msg = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("not a multipart body")
    fields: dict[str, list[str]] = {}
    files: dict[str, tuple] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            files[name] = (filename, data)
        else:
            fields.setdefault(name, []).append(data.decode("utf-8", "replace"))
    return fields, files


async def _read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"request larger than {limit // (1024 * 1024)} MB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"request larger than {limit // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def decode_image(data: bytes) -> np.ndarray:
    """Validate and decode an uploaded image (JPEG/PNG/WebP/BMP) to BGR, or raise HTTP 413/415."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Image.DecompressionBombError:
        raise HTTPException(413, "image has too many pixels") from None
    except Exception:  # noqa: BLE001 - anything PIL cannot open is not an image we accept
        raise HTTPException(415, "not a supported image (JPEG, PNG, WebP or BMP)") from None
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(413, "image has too many pixels")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(415, "image could not be decoded")
    return img


# ---------------------------------------------------------------- samples

def _group(d: Path) -> str:
    parts = [p.lower() for p in d.parts]
    for key, group in GROUPS:
        if any(key in p for p in parts):
            return group
    return d.name or "samples"


def _hint(path: Path, group: str) -> tuple[str, str]:
    """(label hint, suggested ground truth) for a sample image."""
    if group == "dfire_test":
        label = path.parent.parent / "labels" / (path.stem + ".txt")
        try:
            classes = {line.split()[0] for line in label.read_text().splitlines() if line.strip()}
        except OSError:
            return "D-Fire test image (no label file)", "unknown"
        names = [n for c, n in (("0", "smoke"), ("1", "fire")) if c in classes]
        if names:
            return "D-Fire label: " + " + ".join(names), "wildfire"
        return "D-Fire label: no smoke or fire", "no_wildfire"
    if group == "tower_fire_sequence":
        m = _FIGLIB_RE.search(path.name)
        if m:
            off = int(m.group(1))
            when = f"{abs(off) // 60} min {'after' if off >= 0 else 'before'} ignition"
            return f"FIgLib tower sequence, {when}", "wildfire" if off >= 0 else "no_wildfire"
        return "Tower fire sequence", "unknown"
    if group == "benign_tower":
        return f"Benign look-alike ({path.parent.name})", "no_wildfire"
    return path.parent.name, "unknown"


def scan_samples(dirs, cap: int = SAMPLE_CAP) -> list[dict]:
    """Up to `cap` sample images, spread evenly (deterministically) over each configured dir."""
    existing = [Path(d) for d in dirs if Path(d).is_dir()]
    if not existing or cap <= 0:
        return []
    per_dir = -(-cap // len(existing))
    out: list[dict] = []
    for d in existing:
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXT and p.is_file())
        k = min(per_dir, len(files))
        picked = [files[i * len(files) // k] for i in range(k)] if k else []
        group = _group(d)
        for p in picked:
            hint, truth = _hint(p, group)
            out.append({"id": hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:12],
                        "name": p.name, "group": group, "hint": hint, "truth": truth, "path": p})
    return out[:cap]


# ---------------------------------------------------------------- defaults for the Nano

def list_served_models(base_url: str, timeout_s: float = 2.0, transport=None) -> list[str]:
    """Model ids from an OpenAI-compatible server; base_url may be unix:///path/to.sock."""
    if base_url.startswith("unix://"):
        transport = transport or httpx.HTTPTransport(uds=base_url[len("unix://"):])
        base_url = "http://localhost/v1"
    with httpx.Client(transport=transport, timeout=timeout_s) as c:
        r = c.get(base_url.rstrip("/") + "/models")
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def default_detector_factory(spec: dict):
    from sentinel.detector import YoloDetector   # ultralytics is imported lazily inside
    # low display threshold: weak boxes are shown (greyed) even though the 0.4 gate drops them
    return YoloDetector(spec["weights"], conf=0.1, imgsz=spec.get("imgsz", 640), classes=spec.get("classes"))


class NetworkState(BaseModel):
    online: bool


# ---------------------------------------------------------------- app

def create_web_app(*, detector_factory: Callable[[dict], Callable] | None = None,
                   vlm_factory: Callable[[str], object] | None = None,
                   model_lister: Callable[[], list[str]] | None = None,
                   forecast_fn: Callable[[float, float], dict] | None = None,
                   monitor=None, prices_path="config/cost_inputs.json",
                   sample_dirs=(), results_dir="results", tower: Tower | None = None,
                   detector_available: Callable[[dict], bool] | None = None,
                   detectors: dict | None = None, pipelines: dict | None = None,
                   vlm_base_url: str = "http://localhost:8000/v1", vlm_timeout_s: float = 60.0,
                   min_conf: float = 0.4, max_upload_bytes: int = MAX_UPLOAD_BYTES,
                   clock: Callable[[], float] = time.time, warmup: bool = False) -> FastAPI:
    detectors = detectors or DETECTORS
    pipelines = pipelines or PIPELINES
    tower = tower or DEMO_TOWER
    detector_factory = detector_factory or default_detector_factory
    detector_available = detector_available or (lambda spec: Path(spec["weights"]).exists())
    if vlm_factory is None:
        def vlm_factory(model: str):
            from sentinel.vlm_client import ContextVLM
            return ContextVLM(model, vlm_base_url, timeout_s=vlm_timeout_s)
    model_lister = model_lister or (lambda: list_served_models(vlm_base_url))
    if forecast_fn is None:
        from sentinel.cloud import fetch_forecast as forecast_fn

    prices = _load_prices(prices_path)
    session = Session()
    state = {"online": True}
    samples = {s["id"]: s for s in scan_samples(sample_dirs)}
    thumbs: dict[str, bytes] = {}
    gpu_lock = threading.Lock()        # one model call at a time on the shared GPU
    cache_lock = threading.Lock()
    det_cache: dict[str, Callable] = {}
    vlm_cache: dict[str, object] = {}
    models_cache: dict = {"ts": None, "names": [], "error": None}

    def served_models() -> tuple[list[str], str | None]:
        with cache_lock:
            ts = models_cache["ts"]
            if ts is not None and clock() - ts < MODELS_TTL_S:
                return models_cache["names"], models_cache["error"]
        try:
            names, err = [str(n) for n in model_lister()], None
        except Exception as exc:  # noqa: BLE001 - a down server just means "nothing served"
            names, err = [], f"{type(exc).__name__}: {exc}"[:200]
        with cache_lock:
            models_cache.update(ts=clock(), names=names, error=err)
        return names, err

    def pipeline_info(served: list[str]) -> list[dict]:
        out = []
        for name, p in pipelines.items():
            spec = detectors[p["detector"]]
            det_ok, vlm_ok = bool(detector_available(spec)), p["vlm"] in served
            out.append({"name": name, "label": p["label"], "detector": p["detector"],
                        "detector_label": spec["label"], "detector_weights": spec["weights"],
                        "vlm": p["vlm"], "vlm_label": p["vlm_label"], "optional": p["optional"],
                        "detector_available": det_ok, "vlm_served": vlm_ok,
                        "available": det_ok and (vlm_ok or not p["optional"])})
        return out

    def get_detector(key: str):
        # called under gpu_lock: loading weights touches the GPU too
        if key not in det_cache:
            det_cache[key] = detector_factory(detectors[key])
        return det_cache[key]

    def get_vlm(model: str):
        with cache_lock:
            if model not in vlm_cache:
                vlm_cache[model] = vlm_factory(model)
            return vlm_cache[model]

    def session_payload() -> dict:
        return {"session": session.summary(), "economics": session.economics(prices),
                "prices": effective_prices(prices), "online": state["online"]}

    def warm_detectors() -> None:
        """Run each available detector once so the first judged click skips the cold CUDA/CLIP load."""
        keys = dict.fromkeys(p["detector"] for p in pipelines.values())
        for key in keys:
            if not detector_available(detectors[key]):
                continue
            try:
                with gpu_lock:
                    get_detector(key)(np.zeros((640, 640, 3), np.uint8))
                log.info("warmed detector %s", key)
            except Exception:  # noqa: BLE001 - a failed warmup only means a slower first run
                log.exception("detector warmup failed for %s", key)

    @asynccontextmanager
    async def lifespan(app):
        if monitor is not None:
            monitor.start()
        app.state.warmup_thread = None
        if warmup:
            t = threading.Thread(target=warm_detectors, name="detector-warmup", daemon=True)
            app.state.warmup_thread = t
            t.start()
        yield
        if monitor is not None:
            monitor.stop()

    app = FastAPI(title="Wildfire Edge Sentinel demo", lifespan=lifespan)
    app.state.session = session
    app.state.warmup_thread = None

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE.read_text()

    @app.get("/api/config")
    def config():
        served, err = served_models()
        return {"pipelines": pipeline_info(served), "served_models": served, "models_error": err,
                "detectors": {k: {"label": v["label"], "weights": v["weights"],
                                  "available": bool(detector_available(v))} for k, v in detectors.items()},
                "benchmarks": load_benchmarks(results_dir), "prices": effective_prices(prices),
                "online": state["online"], "min_conf": min_conf, "truths": list(TRUTHS),
                "tower": {"id": tower.id, "name": tower.name, "lat": tower.lat, "lon": tower.lon}}

    @app.get("/api/samples")
    def list_samples():
        return {"samples": [{k: v for k, v in s.items() if k != "path"} for s in samples.values()]}

    @app.get("/api/sample/{sample_id}")
    def get_sample(sample_id: str, thumb: bool = False):
        s = samples.get(sample_id) if _ID_RE.match(sample_id) else None
        if s is None:
            raise HTTPException(404, "unknown sample")
        if thumb:
            if sample_id not in thumbs:
                img = cv2.imread(str(s["path"]), cv2.IMREAD_COLOR)
                if img is None:
                    raise HTTPException(404, "sample unreadable")
                h, w = img.shape[:2]
                scale = min(1.0, SAMPLE_THUMB_SIDE / max(h, w))
                img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                                 interpolation=cv2.INTER_AREA)
                thumbs[sample_id] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            return Response(thumbs[sample_id], media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=3600"})
        media = mimetypes.guess_type(s["path"].name)[0] or "application/octet-stream"
        return Response(s["path"].read_bytes(), media_type=media, headers={"Cache-Control": "max-age=3600"})

    def run_pipeline(name: str, image: np.ndarray, force_vlm: bool, served: list[str]) -> dict:
        p = pipelines[name]
        spec = detectors[p["detector"]]
        base = {"pipeline": name, "label": p["label"], "detector": p["detector"],
                "detector_label": spec["label"], "vlm": p["vlm"], "vlm_label": p["vlm_label"],
                "vlm_served": p["vlm"] in served}
        if not detector_available(spec):
            return {**base, "error": f"detector weights not found: {spec['weights']}"}
        vlm = get_vlm(p["vlm"]) if base["vlm_served"] else None
        with gpu_lock:
            try:
                detector = get_detector(p["detector"])
                result = run_pass(image, detector, vlm, tower, min_conf=min_conf,
                                  force_vlm=force_vlm, now=clock())
            except Exception as exc:  # noqa: BLE001 - one broken pass must not sink the other
                log.exception("pipeline %s failed", name)
                return {**base, "error": f"{type(exc).__name__}: {exc}"[:300]}
        return {**base, **result}

    def analyze_sync(image: np.ndarray, names: list[str], truth: str, force_vlm: bool,
                     image_bytes: int) -> list[dict]:
        online = state["online"]
        served, _ = served_models()
        forecasts: dict = {}

        def forecast_once(lat, lon):     # one forecast per request, shared by the passes
            if (lat, lon) not in forecasts:
                try:
                    forecasts[(lat, lon)] = (forecast_fn(lat, lon), None)
                except Exception as exc:  # noqa: BLE001
                    forecasts[(lat, lon)] = (None, exc)
            value, exc = forecasts[(lat, lon)]
            if exc is not None:
                raise exc
            return value

        out = []
        for name in names:
            r = run_pipeline(name, image, force_vlm, served)
            if "error" not in r:
                esc = escalate(r, online, forecast_fn=forecast_once)
                r["escalation"] = esc
                r["correct"] = session.record(name, r, esc, truth=truth, image_bytes=image_bytes,
                                              online=online)
                r.pop("report", None)       # the text and thumbnail are returned separately
            out.append(r)
        session.record_request(online)
        return out

    def handle_analyze(body: bytes, ctype: str) -> dict:
        """Parse, validate, decode and run: all blocking work, so it runs in the threadpool."""
        kind = ctype.split(";", 1)[0].strip().lower()
        if kind == "multipart/form-data":
            try:
                fields, files = parse_multipart(body, ctype)
            except Exception:  # noqa: BLE001
                raise HTTPException(400, "malformed multipart body") from None
        elif kind == "application/x-www-form-urlencoded":
            fields, files = parse_qs(body.decode("utf-8", "replace")), {}
        else:
            raise HTTPException(415, "send multipart/form-data (image or sample_id)")

        def field(name, default=""):
            return (fields.get(name) or [default])[0].strip()

        truth = field("truth", "unknown") or "unknown"
        if truth not in TRUTHS:
            raise HTTPException(400, f"truth must be one of {list(TRUTHS)}")
        names = [n.strip() for v in fields.get("pipelines", ["before,after"]) for n in v.split(",") if n.strip()]
        names = list(dict.fromkeys(names))
        unknown = [n for n in names if n not in pipelines]
        if not names or unknown:
            raise HTTPException(400, f"unknown pipeline(s): {unknown}; choose from {list(pipelines)}")
        force_vlm = field("force_vlm").lower() in ("1", "true", "on", "yes")

        upload = files.get("image")
        if upload is not None and upload[1]:
            filename, data = upload
            if len(data) > max_upload_bytes:
                raise HTTPException(413, f"image larger than {max_upload_bytes // (1024 * 1024)} MB")
            image = decode_image(data)
            info = {"source": "upload", "name": Path(filename).name[:120]}
        elif field("sample_id"):
            sid = field("sample_id")
            s = samples.get(sid) if _ID_RE.match(sid) else None
            if s is None:
                raise HTTPException(404, "unknown sample")
            data = s["path"].read_bytes()
            image = decode_image(data)
            info = {"source": "sample", "name": s["name"], "id": sid}
        else:
            raise HTTPException(400, "attach an image or choose a sample")
        info.update(width=int(image.shape[1]), height=int(image.shape[0]), bytes=len(data))

        results = analyze_sync(image, names, truth, force_vlm, len(data))
        return {"image": info, "truth": truth, "online": state["online"], "results": results,
                **session_payload()}

    @app.post("/api/analyze")
    async def analyze(request: Request):
        body = await _read_body(request, max_upload_bytes + 256 * 1024)
        return await run_in_threadpool(handle_analyze, body, request.headers.get("content-type", ""))

    @app.post("/api/network")
    def set_network(s: NetworkState):
        state["online"] = s.online
        return {"online": state["online"]}

    @app.get("/api/session")
    def get_session():
        live = None
        if monitor is not None:
            try:
                d = monitor.latest()
                live = {"ts": d.get("ts"), "models": d.get("models") or [], "system": d.get("system") or {}}
            except Exception as exc:  # noqa: BLE001 - the page still works without live metrics
                live = {"ts": None, "models": [], "system": {}, "error": str(exc)[:200]}
        return {**session_payload(), "live": live}

    @app.post("/api/prices")
    def set_prices(update: PriceUpdate):
        prices.update(update.model_dump(exclude_none=True))
        return effective_prices(prices)

    @app.post("/api/session/reset")
    def reset():
        session.reset()
        return session_payload()

    return app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--vlm-base-url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible endpoint; models are chosen by name. For LoRA adapters use the "
                         "backend socket: unix:///opt/hp/zrt/run/vllm-base7b.sock (the zrt proxy only "
                         "routes the served label)")
    ap.add_argument("--vlm-timeout", type=float, default=60.0, help="per VLM request, seconds")
    ap.add_argument("--run-dir", default="/opt/hp/zrt/run", help="zrt run dir with vllm-*.json/.sock")
    ap.add_argument("--prices", default="config/cost_inputs.json", help="price/cost inputs JSON")
    ap.add_argument("--samples", action="append", default=None,
                    help="sample image dir (repeatable; default data/dfire/test/images, data/demo, data/benign)")
    ap.add_argument("--results", default="results", help="dir with benchmark results/*.json")
    ap.add_argument("--towers", default="config/towers.json", help="first tower is used for reports")
    ap.add_argument("--before-vlm", default="base7b", help="served name of the BEFORE VLM")
    ap.add_argument("--after-vlm", default="context", help="served name of the AFTER VLM (LoRA)")
    ap.add_argument("--teacher-vlm", default="teacher32b", help="served name of the optional teacher")
    ap.add_argument("--before-weights", default=DETECTORS["yoloworld"]["weights"])
    ap.add_argument("--after-weights", default=DETECTORS["yolo11s"]["weights"])
    ap.add_argument("--after-imgsz", type=int,
                    help="AFTER detector inference size (default: 960 for tower/joint weights, else 640)")
    ap.add_argument("--no-monitor", action="store_true", help="skip live vLLM metrics")
    args = ap.parse_args(argv)

    detectors = detector_specs(args.before_weights, args.after_weights, args.after_imgsz)
    pipelines = {"before": {**PIPELINES["before"], "vlm": args.before_vlm},
                 "after": {**PIPELINES["after"], "vlm": args.after_vlm},
                 "teacher": {**PIPELINES["teacher"], "vlm": args.teacher_vlm}}
    tower = DEMO_TOWER
    try:
        towers = json.loads(Path(args.towers).read_text())
        if towers:
            tower = Tower(**towers[0])
    except (OSError, ValueError, TypeError):
        pass
    monitor = None
    if not args.no_monitor:
        from sentinel.live_metrics import system_stats
        from sentinel.monitor import Monitor, fetch_json, fetch_uds_metrics
        monitor = Monitor(args.run_dir, _load_prices(args.prices), None, fetch_uds_metrics, time.time,
                          fetch_json, system_stats)
    app = create_web_app(monitor=monitor, prices_path=args.prices, warmup=True,
                         sample_dirs=args.samples or ["data/dfire/test/images", "data/demo", "data/benign"],
                         results_dir=args.results, tower=tower, detectors=detectors, pipelines=pipelines,
                         vlm_base_url=args.vlm_base_url, vlm_timeout_s=args.vlm_timeout)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

"""The console's shared services, built once and handed to every app it mounts.

Before the console, each server (demo app, incident map) loaded its own detector, kept its own VLM
clients, its own online/offline flag and its own outbox, so one fire could not be followed across
screens. `Services` owns one of each: one weights load per file, one VLM client per model, one GPU
lock, one uplink (`Link`) and one durable outbox behind `Delivery` (ntfy push + optional dispatch).
Everything that touches the GPU or the network is injected, so tests run on fakes."""
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sentinel.detector import imgsz_for
from sentinel.escalation import Link
from sentinel.live import Delivery
from sentinel.outbox import Outbox
from sentinel.shadow import CloudShadow

log = logging.getLogger(__name__)

MODELS_TTL_S = 10.0
_FROM_ENV = object()      # build_services(notifier=...) default: read NTFY_TOPIC_URL


def default_detector_factory(weights: str, imgsz: int | None = None):
    from sentinel.detector import YoloDetector   # ultralytics is imported lazily inside
    # low display threshold: weak boxes are shown (greyed) even though the 0.4 gate drops them
    return YoloDetector(weights, conf=0.1, imgsz=imgsz or imgsz_for(weights))


def _mtime(weights: str) -> float | None:
    try:
        return Path(weights).stat().st_mtime
    except OSError:
        return None


@dataclass(eq=False)
class Services:
    link: Link
    delivery: Delivery
    detector_factory: Callable[[str, int | None], Callable] = default_detector_factory
    vlm_factory: Callable[[str], object] | None = None
    model_lister: Callable[[], list[str]] = list
    clock: Callable[[], float] = time.time
    shadow: CloudShadow = field(default_factory=CloudShadow)
    node_name: str = "Sentinel edge node"
    gpu_lock: threading.Lock = field(default_factory=threading.Lock)    # one model call at a time

    def __post_init__(self):
        self.started_at = self.clock()
        self._cache_lock = threading.Lock()      # guards the caches below; held during a load (one per file)
        self._detectors: dict[tuple[str, int], tuple[float | None, Callable]] = {}
        self._vlms: dict[str, object] = {}
        self._models = {"ts": None, "names": [], "error": None}
        self._listeners: list[Callable[[bool], None]] = []
        self._link_lock = threading.Lock()
        self.active_vlm: str | None = None       # the console's model switch; None = each app's default

    # ------------------------------------------------------------ models

    def detector(self, weights: str, imgsz: int | None = None) -> Callable:
        """The detector for these weights at this size, loaded once; reloaded if the file is rewritten."""
        size = imgsz or imgsz_for(weights)
        key, mtime = (weights, size), _mtime(weights)
        with self._cache_lock:
            hit = self._detectors.get(key)
            if hit is None or hit[0] != mtime:
                hit = (mtime, self.detector_factory(weights, size))
                self._detectors[key] = hit
            return hit[1]

    def vlm(self, model: str):
        with self._cache_lock:
            if model not in self._vlms:
                self._vlms[model] = self.vlm_factory(model)
            return self._vlms[model]

    def served_models(self) -> tuple[list[str], str | None]:
        """Model ids the VLM server offers, cached MODELS_TTL_S; a down server means nothing is served."""
        with self._cache_lock:
            ts = self._models["ts"]
            if ts is not None and self.clock() - ts < MODELS_TTL_S:
                return list(self._models["names"]), self._models["error"]
        try:
            names, err = [str(n) for n in self.model_lister()], None
        except Exception as exc:  # noqa: BLE001
            names, err = [], f"{type(exc).__name__}: {exc}"[:200]
        with self._cache_lock:
            self._models.update(ts=self.clock(), names=names, error=err)
        return list(names), err

    # ------------------------------------------------------------ uplink

    def on_link_change(self, cb: Callable[[bool], None]) -> None:
        """cb(online) runs after every uplink change and after a flush that sent something, so apps can
        refresh what they show about queued alerts."""
        self._listeners.append(cb)

    def _tell(self, online: bool) -> None:
        for cb in list(self._listeners):
            try:
                cb(online)
            except Exception:  # noqa: BLE001 - one broken listener must not break the toggle
                log.exception("uplink listener failed")

    def set_online(self, online: bool) -> int:
        """Flip the one uplink. Down→up flushes the outbox; returns the number of ALERTs sent."""
        with self._link_lock:
            was = self.link.online
            self.link.online = bool(online)
        if was == self.link.online:
            return 0
        sent = self._flush() if self.link.online else 0
        self._tell(self.link.online)
        return sent

    def _flush(self) -> int:
        try:
            return self.delivery.flush(self.clock())
        except Exception:  # noqa: BLE001 - a broken flush must not break the caller
            log.exception("outbox flush failed")
            return 0

    def flush(self) -> int:
        """Send what is due (retry loop and after new ALERTs); tells listeners only if something went out."""
        sent = self._flush()
        if sent:
            self._tell(self.link.online)
        return sent


def build_services(*, outbox_path: str = ":memory:", notifier=_FROM_ENV, dispatch_fn=None,
                   forecast_fn: Callable[[float, float], dict] | None = None,
                   detector_factory: Callable[[str, int | None], Callable] | None = None,
                   vlm_factory: Callable[[str], object] | None = None,
                   model_lister: Callable[[], list[str]] | None = None,
                   vlm_base_url: str = "http://localhost:8000/v1", vlm_timeout_s: float = 60.0,
                   start_online: bool = True, node_name: str = "Sentinel edge node",
                   shadow: CloudShadow | None = None, clock: Callable[[], float] = time.time) -> Services:
    """Services with the Nano defaults: YOLO weights, ContextVLM on `vlm_base_url` (unix:// works),
    ntfy from NTFY_TOPIC_URL, open-meteo forecasts, and a durable outbox at `outbox_path`."""
    if notifier is _FROM_ENV:
        from sentinel.notify import NtfyNotifier
        notifier = NtfyNotifier.from_env()
    if forecast_fn is None:
        from sentinel.cloud import fetch_forecast as forecast_fn
    if vlm_factory is None:
        def vlm_factory(model: str):
            from sentinel.vlm_client import ContextVLM
            return ContextVLM(model, vlm_base_url, timeout_s=vlm_timeout_s)
    if model_lister is None:
        def model_lister():
            from sentinel.webapp import list_served_models
            return list_served_models(vlm_base_url)
    if outbox_path != ":memory:":
        Path(outbox_path).parent.mkdir(parents=True, exist_ok=True)
    link = Link(online=start_online)
    delivery = Delivery(Outbox(outbox_path), link, forecast_fn, notifier=notifier, dispatch_fn=dispatch_fn,
                        clock=clock)
    return Services(link=link, delivery=delivery, detector_factory=detector_factory or default_detector_factory,
                    vlm_factory=vlm_factory, model_lister=model_lister, clock=clock,
                    shadow=shadow or CloudShadow(), node_name=node_name)

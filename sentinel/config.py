"""Settings and tower configuration (JSON files under config/)."""
import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Settings:
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    vlm_timeout_s: float = 5.0
    detector_weights: str = "models/smoke_yolo.pt"
    detector_classes: list[str] | None = None  # set for YOLO-World zero-shot, e.g. ["smoke", "fire"]
    fps: float = 2.0
    min_conf: float = 0.4
    min_frames: int = 3
    cooldown_s: float = 120.0
    recheck_s: float = 30.0
    max_rechecks: int = 4
    crop_pad: float = 0.5
    crop_max_side: int = 448
    full_frame: bool = False
    full_frame_max_side: int = 1280
    dispatch_url: str = "http://localhost:9000/dispatch"
    db_path: str = "data/outbox.db"
    dashboard_port: int = 8080


@dataclass
class Tower:
    id: str
    name: str
    lat: float
    lon: float
    source: str
    temp_c: float = 25.0
    benign_zones: list = field(default_factory=list)


def load_settings(path: str | Path = "config/settings.json") -> Settings:
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    unknown = set(data) - {f.name for f in fields(Settings)}
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    return Settings(**data)


def load_towers(path: str | Path = "config/towers.json") -> dict[str, Tower]:
    return {t["id"]: Tower(**t) for t in json.loads(Path(path).read_text())}

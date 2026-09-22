# Wildfire Edge Sentinel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `development/reference/executing-plans-guide.md` to implement this plan task-by-task.

**Goal:** Build an edge-first wildfire smoke/fire detection system on the HP ZGX Nano that classifies fire *context* (wildland vs campfire vs fog, attended, growing, near structures), assigns severity, writes a dispatch report locally, and escalates to the cloud only for ALERT events and only for data the edge cannot have. Every cost, latency and token claim is measured.

**Architecture:** One Python service (`sentinel/`) runs a 3-stage cascade: a fine-tuned YOLO detector on every frame, then a LoRA-adapted Qwen2.5-VL context classifier served locally by vLLM (`zrt serve`) once per candidate event, then a deterministic trend and severity rule engine. ALERT reports go to a SQLite outbox that flushes to the cloud (Open-Meteo forecast plus a simulated dispatch webhook) only when the link is up. A FastAPI dashboard drives the demo. Training (YOLO fine-tune, teacher→student LoRA distillation) happens on the Nano.

**Tech Stack:** Python 3.10+, Ultralytics YOLO11, vLLM via HP ZRT, Qwen2.5-VL-7B-Instruct (student) and Qwen2.5-VL-32B-Instruct-AWQ (teacher), PEFT/TRL (LoRA), SigLIP + scikit-learn (baseline), FastAPI/uvicorn, SQLite, OpenCV, httpx, openai SDK, pytest.

**Design doc:** `docs/plans/2026-09-22-wildfire-edge-sentinel-design.md`

---

## Schedule at a glance (deadline Fri Sept 25, 8pm; internal target 6pm)

| Day | Nano (GPU, runs in background) | Laptop (pure Python, TDD, no GPU needed) |
|---|---|---|
| **Day 0: Tue 9/22 eve** | T2 bring-up, serve base VLM · T3 download data | T1 repo scaffold |
| **Day 1: Wed 9/23** | T4 train detector (hours) → T12 teacher labeling (overnight) | T5–T11, T13–T15 core logic |
| **Day 2: Thu 9/24** | T16 LoRA training (hours) → T24 serve adapter + eval | T17–T23 pipeline, escalation, dashboard, end-to-end |
| **Day 3: Fri 9/25** | T25 benchmarks | T26 cost model · T27 README · T28 demo, video, deck, submission |

**Key trick:** the laptop work is pure logic with fakes, so it never waits on the GPU. Kick off every long GPU job *before* starting laptop tasks.

**Cut lines if behind (drop in this order):** linear-probe baseline (T24 part C) → feedback button → LoRA distillation (serve the base VLM; still demo-able) → full-frame token ablation. **Never cut:** escalation policy, outbox, benchmarks, cost model. Those are what the brief scores.

**Ports on the Nano:** vLLM student `8000` · vLLM teacher `8001` · dashboard `8080` · dispatch stub `9000`.

**Git:** this project gets its **own new repo** (it must be public for submission). Do *not* use the home-directory repo. Commit steps below apply once the repo exists.

---

## Day 0: Tuesday evening

### Task 1: Repo scaffold

**Files:**
- Create: `wildfire-edge-sentinel/pyproject.toml`
- Create: `wildfire-edge-sentinel/requirements-dev.txt`, `requirements.txt`, `requirements-train.txt`
- Create: `wildfire-edge-sentinel/sentinel/__init__.py`, `tests/__init__.py`
- Create: `wildfire-edge-sentinel/.gitignore`
- Create: `wildfire-edge-sentinel/config/towers.json`, `config/settings.json`, `config/burn_schedule.json`

**Step 1: Create the directory and files**

`pyproject.toml`:
```toml
[project]
name = "wildfire-edge-sentinel"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

`requirements-dev.txt` (laptop, no GPU):
```
numpy
opencv-python-headless
pydantic>=2
fastapi
uvicorn
httpx
openai>=1.40
pytest
```

`requirements.txt` (Nano runtime):
```
-r requirements-dev.txt
ultralytics
```

`requirements-train.txt` (Nano training):
```
transformers
peft
trl
accelerate
datasets
pillow
scikit-learn
```

`.gitignore`:
```
.venv/
__pycache__/
data/
runs/
models/*.pt
adapters/
results/*.tmp
*.db
```

`config/towers.json` (the demo sources are filled in during T23):
```json
[
  {"id": "t1", "name": "Mt. Umunhum Lookout", "lat": 37.1606, "lon": -121.8983,
   "source": "data/demo/t1", "temp_c": 31.0, "benign_zones": []},
  {"id": "t2", "name": "Loma Prieta Ridge", "lat": 37.1114, "lon": -121.8430,
   "source": "data/demo/t2", "temp_c": 29.5, "benign_zones": []}
]
```

`config/settings.json`:
```json
{}
```

`config/burn_schedule.json` (simulated cloud data):
```json
[]
```

`sentinel/__init__.py` and `tests/__init__.py`: empty.

**Step 2: Create the venv and verify pytest runs**

```bash
cd wildfire-edge-sentinel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```
Expected: `no tests ran` (exit code 5). That's fine.

**Step 3: Create a public GitHub repo and push**

```bash
git init && git add . && git commit -m "chore: scaffold wildfire-edge-sentinel"
gh repo create wildfire-edge-sentinel --public --source . --push
```

---

### Task 2: Nano bring-up and base VLM serving

This is manual, on the Nano over ZTK SSH. There are no tests. Each step has an expected output.

**Step 1: Connect via ZTK** (hack guide pages 3–4). Verify:
```bash
nvidia-smi
```
Expected: GB10 listed.

**Step 2: Install and configure ZRT**
```bash
sudo snap install --classic zrt
zrt config set proxy.auth.type none && zrt config set proxy.tls.enabled false
zrt status
```

**Step 3: Pull and serve the student VLM** (leave room for training and the teacher)
```bash
zrt pull Qwen/Qwen2.5-VL-7B-Instruct
zrt serve hf:Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.30 --enable-prefix-caching
```
Run it in `tmux` so it survives SSH drops: `tmux new -s vlm`.

**Step 4: Record the served model id**
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```
Write the exact `id` (it may be `hf:Qwen/Qwen2.5-VL-7B-Instruct`) into `config/settings.json`:
```json
{"vlm_model": "hf:Qwen/Qwen2.5-VL-7B-Instruct"}
```

**Step 5: Smoke-test chat**
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id>","messages":[{"role":"user","content":"Say OK"}],"max_tokens":5}'
```
Expected: JSON containing `"OK"`.

**Step 6: Clone the repo on the Nano and set up Python**
```bash
cd ~ && git clone https://github.com/<you>/wildfire-edge-sentinel.git sentinel && cd sentinel
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If this prints `False` or `ModuleNotFoundError`, install CUDA PyTorch for GB10 (aarch64) exactly as the **NVIDIA DGX Spark playbook** says (hack guide resource list), *then*:
```bash
python3 -m venv .venv --system-site-packages && source .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```
Expected: the device name prints.

---

### Task 3: Datasets

**Files:**
- Create: `scripts/download_data.sh`

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# Downloads training/eval data onto the Nano. Verify each dataset's license and record it in README.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dfire data/benign data/demo data/bench data/pyro

# 1) D-Fire (YOLO format; classes 0=smoke, 1=fire). Hosted via link on
#    https://github.com/gaiasd/DFireDataset. Download on laptop if needed, then:
#    scp D-Fire.zip hpX@<nano-ip>:~/sentinel/data/dfire/ && unzip there.
if [ ! -d data/dfire/train ]; then
  echo "MANUAL: place D-Fire under data/dfire/{train,test}/{images,labels}"; fi

# 2) PyroNear lookout-tower smoke (optional extra detector data / eval).
huggingface-cli download pyronear/pyro-sdis --repo-type dataset --local-dir data/pyro || \
  echo "WARN: pyro-sdis download failed; continuing with D-Fire only"

# 3) FIgLib (HPWREN Fire Ignition Library) sequences for demo + trend eval:
#    browse https://www.hpwren.ucsd.edu/FIgLib/ and download 4-6 sequences into
#    data/demo/<name>/ (each a folder of time-ordered JPGs).
echo "MANUAL: FIgLib sequences -> data/demo/<sequence>/*.jpg"

# 4) Benign look-alikes (campfire, BBQ, chimney, stack, fog, low cloud): 150-300 JPGs
#    from openly licensed sources -> data/benign/*.jpg ; short clips -> data/demo/benign_*/
echo "MANUAL: benign images -> data/benign/*.jpg"
```

**Step 2: Run it and verify the D-Fire layout and class ids**
```bash
chmod +x scripts/download_data.sh && ./scripts/download_data.sh
ls data/dfire/train/images | head -3; ls data/dfire/train/labels | head -3
cat data/dfire/train/labels/$(ls data/dfire/train/labels | head -1)
```
Expected: YOLO lines `cls cx cy w h`, where cls ∈ {0, 1}. Confirm 0=smoke and 1=fire in the D-Fire README. If the folder names differ, adjust the paths in T4 and T12 (don't rename the data).

**Step 3: Inspect pyro-sdis before using it.** `ls data/pyro`. Only merge it into detector training if it's already images + YOLO labels. Otherwise use it just as eval images. Don't spend over 30 minutes here.

**Step 4: Commit**
```bash
git add scripts/download_data.sh && git commit -m "chore: dataset download script"
```

---

## Day 1: Wednesday

### Task 4: Train the smoke/fire detector (Nano, background, START FIRST)

**Files:**
- Create: `scripts/train_detector.py`

**Step 1: Write the script**

```python
"""Fine-tune YOLO on D-Fire (0=smoke, 1=fire). Run on the Nano."""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dfire")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    root = Path(a.root).resolve()  # Ultralytics resolves relative paths against its own datasets dir
    data_yaml = Path("runs/dfire.yaml")
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        f"path: {root}\ntrain: train/images\nval: test/images\nnames:\n  0: smoke\n  1: fire\n"
    )

    model = YOLO(a.model)
    model.train(data=str(data_yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=0, project="runs", name="smoke", exist_ok=True)
    metrics = model.val()
    print(f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f}")

    Path("models").mkdir(exist_ok=True)
    shutil.copy("runs/smoke/weights/best.pt", "models/smoke_yolo.pt")


if __name__ == "__main__":
    main()
```

**Step 2: Quick 1-epoch dry run to get an ETA**
```bash
tmux new -s train
python scripts/train_detector.py --epochs 1
```
Expected: training completes and prints `mAP50=...`, and `models/smoke_yolo.pt` exists. Note the time per epoch and pick `--epochs` so the full run finishes in ≤ 4 h.

**Step 3: Full run (leave it running)**
```bash
python scripts/train_detector.py --epochs <N>
```
Record final mAP50 and mAP50-95 in `results/detector.md`. This is a benchmark deliverable.

**Step 4: Commit**
```bash
git add scripts/train_detector.py && git commit -m "feat: YOLO detector fine-tuning script"
```

---

### Task 5: Core types (`schema.py`)

**Files:**
- Create: `sentinel/schema.py`
- Test: `tests/test_schema.py`

**Step 1: Write the failing test**

```python
import pytest
from pydantic import ValidationError

from sentinel.schema import CONTEXT_JSON_SCHEMA, ContextResult, Detection, Severity


def test_detection_area():
    assert Detection("smoke", 0.9, (10, 20, 30, 60)).area == 800


def test_severity_ordering():
    assert Severity.ALERT > Severity.MONITOR > Severity.LOG > Severity.IGNORE


def test_context_rejects_unknown_source():
    with pytest.raises(ValidationError):
        ContextResult(source_type="volcano", smoke_color="grey", attended="no",
                      near_structures=False, near_road=False, size_estimate="small",
                      description="x")


def test_json_schema_constrains_source_type():
    enum = CONTEXT_JSON_SCHEMA["properties"]["source_type"]["enum"]
    assert "campfire" in enum and "fog_dust_cloud" in enum
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'sentinel.schema'`

**Step 3: Implement**

```python
"""Shared data types for the sentinel pipeline."""
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(IntEnum):
    IGNORE = 0
    LOG = 1
    MONITOR = 2
    ALERT = 3


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


SourceType = Literal[
    "wildland", "structure", "vehicle", "controlled_burn", "campfire",
    "bbq_chimney", "industrial_stack", "fog_dust_cloud", "unknown",
]


class ContextResult(BaseModel):
    source_type: SourceType
    smoke_color: Literal["white", "grey", "black", "none"]
    attended: Literal["yes", "no", "unclear"]
    near_structures: bool
    near_road: bool
    size_estimate: Literal["small", "medium", "large"]
    description: str = Field(max_length=200)


CONTEXT_JSON_SCHEMA = ContextResult.model_json_schema()
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add sentinel/schema.py tests/test_schema.py && git commit -m "feat: core pipeline types"
```

---

### Task 6: Trend classification (`trend.py`)

**Files:**
- Create: `sentinel/trend.py`
- Test: `tests/test_trend.py`

**Step 1: Write the failing test**

```python
from sentinel.trend import classify_trend


def test_growing():
    assert classify_trend(100, 140) == "growing"


def test_static():
    assert classify_trend(100, 110) == "static"


def test_dissipating():
    assert classify_trend(100, 50) == "dissipating"
    assert classify_trend(100, 0) == "dissipating"


def test_from_zero_baseline():
    assert classify_trend(0, 50) == "growing"
    assert classify_trend(0, 0) == "static"
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_trend.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Growth trend between two observations of the same smoke region."""
from typing import Literal

Trend = Literal["growing", "static", "dissipating"]


def classify_trend(area_before: float, area_after: float,
                   grow: float = 1.3, shrink: float = 0.7) -> Trend:
    if area_before <= 0:
        return "growing" if area_after > 0 else "static"
    ratio = area_after / area_before
    if ratio >= grow:
        return "growing"
    if ratio <= shrink:
        return "dissipating"
    return "static"
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_trend.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add sentinel/trend.py tests/test_trend.py && git commit -m "feat: growth trend classification"
```

---

### Task 7: Severity rules (`severity.py`)

This is the most important logic to defend to judges. Every rule has a test.

**Files:**
- Create: `sentinel/severity.py`
- Test: `tests/test_severity.py`

**Step 1: Write the failing test**

```python
from sentinel.schema import ContextResult, Severity
from sentinel.severity import assess, fallback_severity


def ctx(**kw):
    base = dict(source_type="wildland", smoke_color="grey", attended="no",
                near_structures=False, near_road=False, size_estimate="small",
                description="x")
    base.update(kw)
    return ContextResult(**base)


def test_fog_is_ignored_even_if_growing():
    assert assess(ctx(source_type="fog_dust_cloud"), trend="growing") == Severity.IGNORE


def test_growing_wildland_alerts():
    assert assess(ctx(), trend="growing") == Severity.ALERT


def test_static_small_wildland_is_monitor():
    assert assess(ctx(), trend="static") == Severity.MONITOR


def test_wildland_near_structures_alerts_without_trend():
    assert assess(ctx(near_structures=True), trend=None) == Severity.ALERT


def test_black_smoke_alerts_without_trend():
    assert assess(ctx(smoke_color="black"), trend=None) == Severity.ALERT


def test_attended_static_campfire_is_log():
    assert assess(ctx(source_type="campfire", attended="yes"), trend="static") == Severity.LOG


def test_unattended_campfire_is_monitor():
    assert assess(ctx(source_type="campfire", attended="no"), trend="static") == Severity.MONITOR


def test_campfire_out_of_control_alerts():
    c = ctx(source_type="campfire", attended="yes", size_estimate="medium")
    assert assess(c, trend="growing") == Severity.ALERT


def test_stack_in_benign_zone_is_ignored():
    c = ctx(source_type="industrial_stack", smoke_color="white")
    assert assess(c, trend="static", in_benign_zone=True) == Severity.IGNORE


def test_unknown_in_benign_zone_is_log():
    assert assess(ctx(source_type="unknown"), trend="static", in_benign_zone=True) == Severity.LOG


def test_scheduled_burn_downgrades_to_log():
    assert assess(ctx(), trend="growing", burn_scheduled=True) == Severity.LOG


def test_scheduled_burn_near_structures_still_alerts():
    assert assess(ctx(near_structures=True), trend=None, burn_scheduled=True) == Severity.ALERT


def test_unknown_growing_alerts():
    assert assess(ctx(source_type="unknown"), trend="growing") == Severity.ALERT


def test_fallback_without_context_never_ignores():
    assert fallback_severity(None) == Severity.MONITOR
    assert fallback_severity("static") == Severity.MONITOR
    assert fallback_severity("growing") == Severity.ALERT
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_severity.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Deterministic, auditable severity rules. Rule order matters; see tests."""
from sentinel.schema import ContextResult, Severity

DANGEROUS = {"wildland", "structure", "vehicle"}
BENIGN = {"campfire", "bbq_chimney", "industrial_stack", "controlled_burn"}
ATTENDABLE = {"campfire", "bbq_chimney"}


def assess(ctx: ContextResult, trend: str | None,
           in_benign_zone: bool = False, burn_scheduled: bool = False) -> Severity:
    growing = trend == "growing"
    if ctx.source_type == "fog_dust_cloud":
        return Severity.IGNORE
    if burn_scheduled and not ctx.near_structures:
        return Severity.LOG
    if ctx.source_type in DANGEROUS:
        if growing or ctx.near_structures or ctx.size_estimate == "large" or ctx.smoke_color == "black":
            return Severity.ALERT
        return Severity.MONITOR
    if ctx.source_type in BENIGN or in_benign_zone:
        if growing and ctx.size_estimate != "small":
            return Severity.ALERT
        if ctx.source_type == "industrial_stack" and in_benign_zone:
            return Severity.IGNORE
        if ctx.source_type in ATTENDABLE and ctx.attended == "no":
            return Severity.MONITOR
        return Severity.LOG
    return Severity.ALERT if growing else Severity.MONITOR


def fallback_severity(trend: str | None) -> Severity:
    """Used when the context VLM is unavailable: never silently ignore."""
    return Severity.ALERT if trend == "growing" else Severity.MONITOR
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_severity.py -v`
Expected: 14 passed

**Step 5: Commit**
```bash
git add sentinel/severity.py tests/test_severity.py && git commit -m "feat: severity rule engine"
```

---

### Task 8: Persistence gate (`gate.py`)

This filters single-frame flickers before any VLM call happens (token saving #1).

**Files:**
- Create: `sentinel/gate.py`
- Test: `tests/test_gate.py`

**Step 1: Write the failing test**

```python
from sentinel.gate import PersistenceGate
from sentinel.schema import Detection

D = Detection("smoke", 0.8, (0, 0, 10, 10))
LOW = Detection("smoke", 0.2, (0, 0, 10, 10))


def gate():
    return PersistenceGate(min_conf=0.4, min_frames=3, cooldown_s=60)


def test_fires_after_min_frames():
    g = gate()
    assert g.update("t1", [D], 0) is None
    assert g.update("t1", [D], 1) is None
    assert g.update("t1", [D], 2) == D


def test_low_confidence_resets_streak():
    g = gate()
    g.update("t1", [D], 0)
    g.update("t1", [D], 1)
    assert g.update("t1", [LOW], 2) is None
    assert g.update("t1", [D], 3) is None
    assert g.update("t1", [D], 4) is None
    assert g.update("t1", [D], 5) == D


def test_cooldown_blocks_refire():
    g = gate()
    for t in (0, 1, 2):
        g.update("t1", [D], t)
    for t in (3, 4, 5):
        assert g.update("t1", [D], t) is None
    assert g.update("t1", [D], 62) == D


def test_towers_are_independent():
    g = gate()
    g.update("t1", [D], 0)
    g.update("t1", [D], 1)
    assert g.update("t2", [D], 2) is None


def test_picks_highest_confidence():
    g = PersistenceGate(min_conf=0.4, min_frames=1, cooldown_s=60)
    best = Detection("fire", 0.95, (5, 5, 9, 9))
    assert g.update("t1", [D, best], 0) == best
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_gate.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Require a detection to persist across frames before opening an event."""
from collections import defaultdict

from sentinel.schema import Detection


class PersistenceGate:
    def __init__(self, min_conf: float, min_frames: int, cooldown_s: float):
        self.min_conf = min_conf
        self.min_frames = min_frames
        self.cooldown_s = cooldown_s
        self._streak: dict[str, int] = defaultdict(int)
        self._last_fired: dict[str, float] = {}

    def update(self, tower_id: str, detections: list[Detection], now: float) -> Detection | None:
        strong = [d for d in detections if d.conf >= self.min_conf]
        if not strong:
            self._streak[tower_id] = 0
            return None
        self._streak[tower_id] += 1
        if self._streak[tower_id] < self.min_frames:
            return None
        last = self._last_fired.get(tower_id)
        if last is not None and now - last < self.cooldown_s:
            return None
        self._last_fired[tower_id] = now
        self._streak[tower_id] = 0
        return max(strong, key=lambda d: d.conf)
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_gate.py -v`
Expected: 5 passed

**Step 5: Commit**
```bash
git add sentinel/gate.py tests/test_gate.py && git commit -m "feat: persistence gate"
```

---

### Task 9: Benign zone masks (`zones.py`)

**Files:**
- Create: `sentinel/zones.py`
- Test: `tests/test_zones.py`

**Step 1: Write the failing test**

```python
from sentinel.zones import in_any_zone

SQUARE = [[0, 0], [50, 0], [50, 50], [0, 50]]


def test_box_center_inside_zone():
    assert in_any_zone((10, 10, 30, 30), [SQUARE])


def test_box_center_outside_zone():
    assert not in_any_zone((100, 100, 120, 120), [SQUARE])


def test_no_zones():
    assert not in_any_zone((10, 10, 30, 30), [])
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_zones.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Per-tower polygons marking known benign sources (campground, stack)."""


def _point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def in_any_zone(box: tuple[float, float, float, float], zones: list[list[list[float]]]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(_point_in_polygon(cx, cy, z) for z in zones)
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_zones.py -v`
Expected: 3 passed

**Step 5: Commit**
```bash
git add sentinel/zones.py tests/test_zones.py && git commit -m "feat: benign zone masks"
```

---

### Task 10: Cropping and JPEG (`imaging.py`)

Crop to the detection region with generous padding. The padding keeps the context the VLM needs (fire ring, people, houses) while cutting image tokens (token saving #2).

**Files:**
- Create: `sentinel/imaging.py`
- Test: `tests/test_imaging.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sentinel.imaging import crop_box, to_jpeg

FRAME = np.zeros((480, 640, 3), np.uint8)


def test_crop_adds_padding():
    assert crop_box(FRAME, (100, 100, 200, 200), pad=0.5, max_side=448).shape == (200, 200, 3)


def test_crop_downscales_to_max_side():
    assert crop_box(FRAME, (0, 0, 640, 480), pad=0.0, max_side=448).shape == (336, 448, 3)


def test_crop_clips_to_frame():
    assert crop_box(FRAME, (600, 440, 640, 480), pad=0.5, max_side=448).shape == (60, 60, 3)


def test_to_jpeg_returns_jpeg_bytes():
    assert to_jpeg(FRAME)[:2] == b"\xff\xd8"
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_imaging.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Image helpers: context-preserving crops and JPEG encoding."""
import cv2
import numpy as np


def crop_box(frame: np.ndarray, box: tuple[float, float, float, float],
             pad: float = 0.5, max_side: int = 448) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
    x2, y2 = min(w, int(x2 + pad * bw)), min(h, int(y2 + pad * bh))
    crop = frame[y1:y2, x1:x2]
    scale = max_side / max(crop.shape[:2])
    if scale < 1:
        size = (int(crop.shape[1] * scale), int(crop.shape[0] * scale))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return crop


def to_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_imaging.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add sentinel/imaging.py tests/test_imaging.py && git commit -m "feat: context crops and jpeg encoding"
```

---

### Task 11: Context VLM client (`vlm_client.py`)

It uses guided JSON output (a schema-constrained response), a tight token cap, and a fixed system prompt (so vLLM prefix caching kicks in). It retries once on bad JSON and returns no result immediately on timeout. The teacher labeling and eval scripts reuse the same client.

**Files:**
- Create: `sentinel/vlm_client.py`
- Test: `tests/test_vlm_client.py`

**Step 1: Write the failing test**

```python
import json
from types import SimpleNamespace

from sentinel.vlm_client import ContextVLM

VALID = json.dumps({"source_type": "campfire", "smoke_color": "white", "attended": "yes",
                    "near_structures": False, "near_road": False, "size_estimate": "small",
                    "description": "Small campfire in a fire ring with people nearby."})


class FakeCompletions:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=out))],
            usage=SimpleNamespace(prompt_tokens=300, completion_tokens=40))


def vlm(outputs):
    completions = FakeCompletions(outputs)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return ContextVLM("m", client=client), completions


def test_parses_valid_json():
    v, _ = vlm([VALID])
    ctx, tokens = v.classify(b"\xff\xd8fake")
    assert ctx.source_type == "campfire" and tokens == 340


def test_retries_once_on_bad_json():
    v, _ = vlm(["not json", VALID])
    ctx, tokens = v.classify(b"\xff\xd8fake")
    assert ctx is not None and tokens == 680


def test_gives_up_after_two_bad_outputs():
    v, _ = vlm(["bad", "bad"])
    assert v.classify(b"\xff\xd8fake") == (None, 680)


def test_timeout_returns_none_without_retry():
    v, c = vlm([TimeoutError("slow")])
    assert v.classify(b"\xff\xd8fake") == (None, 0)
    assert len(c.calls) == 1


def test_request_is_token_capped_and_schema_constrained():
    v, c = vlm([VALID])
    v.classify(b"\xff\xd8fake")
    call = c.calls[0]
    assert call["max_tokens"] <= 200
    assert call["temperature"] == 0
    assert call["response_format"]["type"] == "json_schema"
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_vlm_client.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""OpenAI-compatible client for the local vLLM context classifier."""
import base64

from openai import OpenAI
from pydantic import ValidationError

from sentinel.schema import CONTEXT_JSON_SCHEMA, ContextResult

SYSTEM_PROMPT = (
    "You are a wildfire lookout assistant. You see a cropped region from a remote tower "
    "camera where a detector flagged possible smoke or fire. Classify the source and its "
    "context. Benign sources (campfire, BBQ/chimney, industrial stack, controlled burn) and "
    "look-alikes (fog, dust, low cloud) are common, so look for fire rings, people, "
    "buildings, roads and smoke color. Answer ONLY with JSON matching the schema. "
    "description: one factual sentence, at most 25 words."
)
USER_PROMPT = "Classify this scene."


class ContextVLM:
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1",
                 timeout_s: float = 5.0, max_tokens: int = 160, client=None):
        self.model = model
        self.max_tokens = max_tokens
        self.client = client or OpenAI(base_url=base_url, api_key="EMPTY",
                                       timeout=timeout_s, max_retries=0)

    def _messages(self, jpeg: bytes) -> list[dict]:
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]

    def classify(self, jpeg: bytes) -> tuple[ContextResult | None, int]:
        """Returns (context or None, total tokens spent)."""
        tokens = 0
        for _ in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=self._messages(jpeg),
                    max_tokens=self.max_tokens, temperature=0,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "context", "schema": CONTEXT_JSON_SCHEMA}},
                )
            except Exception:
                return None, tokens
            tokens += resp.usage.prompt_tokens + resp.usage.completion_tokens
            try:
                return ContextResult.model_validate_json(resp.choices[0].message.content), tokens
            except ValidationError:
                continue
        return None, tokens
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_vlm_client.py -v`
Expected: 5 passed

**Step 5: Live smoke test on the Nano** (after `git pull`)
```bash
python - <<'EOF'
import cv2, glob, json
from sentinel.imaging import crop_box, to_jpeg
from sentinel.vlm_client import ContextVLM
settings = json.load(open("config/settings.json"))
img = cv2.imread(sorted(glob.glob("data/dfire/test/images/*.jpg"))[0])
print(ContextVLM(settings["vlm_model"], timeout_s=30).classify(to_jpeg(crop_box(img, (0, 0, img.shape[1], img.shape[0]), 0))))
EOF
```
Expected: `(ContextResult(...), <tokens>)`. If you get `(None, 0)`, check that vLLM accepts `response_format` json_schema (upgrade ZRT/vLLM or switch to `extra_body={"guided_json": ...}` for older vLLM).

**Step 6: Commit**
```bash
git add sentinel/vlm_client.py tests/test_vlm_client.py && git commit -m "feat: schema-constrained context VLM client"
```

---

### Task 12: Teacher labeling for distillation (Nano, START BEFORE BED)

A large local teacher VLM labels crops with the context schema, and the LoRA student learns from those labels in T16. No labeling data leaves the device.

**Files:**
- Create: `scripts/teacher_label.py`
- Test: `tests/test_teacher_label.py`

**Step 1: Write the failing test.** Only the YOLO-label parsing is tested. It lives in `sentinel/labels.py` so the test can import it.

```python
# tests/test_teacher_label.py
from sentinel.labels import yolo_boxes


def test_yolo_boxes_to_pixels(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n\n")
    assert yolo_boxes(p, w=100, h=50) == [(40.0, 15.0, 60.0, 35.0)]


def test_missing_label_file_means_no_boxes(tmp_path):
    assert yolo_boxes(tmp_path / "none.txt", 100, 50) == []
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_teacher_label.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'sentinel.labels'`

**Step 3: Implement `sentinel/labels.py`**

```python
"""YOLO label helpers shared by training scripts."""
from pathlib import Path


def yolo_boxes(label_path: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes
```


**Step 4: Run to verify it passes**

Run: `pytest tests/test_teacher_label.py -v`
Expected: 2 passed

**Step 5: Write `scripts/teacher_label.py`**

```python
"""Crop D-Fire + benign images and label them with the local teacher VLM (concurrent)."""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

from sentinel.imaging import crop_box, to_jpeg
from sentinel.labels import yolo_boxes
from sentinel.vlm_client import ContextVLM


def make_crops(images_dir: Path, labels_dir: Path, out_dir: Path, n: int, seed: int = 0) -> list[Path]:
    imgs = sorted(images_dir.glob("*.jpg"))
    random.Random(seed).shuffle(imgs)
    out = []
    for p in imgs[:n]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = yolo_boxes(labels_dir / f"{p.stem}.txt", w, h)
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if boxes else (0, 0, w, h)
        crop = crop_box(img, box, pad=0.5 if boxes else 0.0, max_side=448)
        dst = out_dir / f"{images_dir.parent.name}_{p.stem}.jpg"
        dst.write_bytes(to_jpeg(crop, 90))
        out.append(dst)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/dfire/train/images")
    ap.add_argument("--labels", default="data/dfire/train/labels")
    ap.add_argument("--benign", default="data/benign")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", required=True, help="teacher id from :8001/v1/models")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="data/teacher")
    a = ap.parse_args()

    out = Path(a.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    crops = make_crops(Path(a.images), Path(a.labels), crops_dir, a.n)
    crops += make_crops(Path(a.benign), Path(a.benign) / "_no_labels", crops_dir, 10_000)
    print(f"{len(crops)} crops to label")

    teacher = ContextVLM(a.model, a.base_url, timeout_s=180, max_tokens=200)

    def label(p: Path):
        return p, teacher.classify(p.read_bytes())[0]

    with ThreadPoolExecutor(a.workers) as ex, \
            open(out / "train.jsonl", "w") as ftr, open(out / "heldout.jsonl", "w") as fho:
        for i, (p, ctx) in enumerate(ex.map(label, crops)):
            if ctx is None:
                continue
            held = int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 100 < 15
            (fho if held else ftr).write(json.dumps({"image": str(p), "label": ctx.model_dump()}) + "\n")
            if i % 100 == 0:
                print(f"{i}/{len(crops)}", flush=True)


if __name__ == "__main__":
    main()
```

**Step 6: Serve the teacher and start labeling (Nano, after YOLO training finishes)**
```bash
tmux new -s teacher
zrt pull Qwen/Qwen2.5-VL-32B-Instruct-AWQ   # verify exact repo id on Hugging Face first
zrt serve hf:Qwen/Qwen2.5-VL-32B-Instruct-AWQ --host 0.0.0.0 --port 8001 \
  --max-model-len 8192 --gpu-memory-utilization 0.40
# new tmux window:
python scripts/teacher_label.py --model "$(curl -s localhost:8001/v1/models | python -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["id"])')" --n 200
```
Expected: progress lines. Check that throughput is acceptable at `--n 200`, then rerun with `--n 2000`. Use a 72B-AWQ teacher only if the throughput allows. Also record the **teacher's** seconds per image for the distillation slide.

**Step 7: Commit**
```bash
git add sentinel/labels.py tests/test_teacher_label.py scripts/teacher_label.py
git commit -m "feat: teacher labeling for context distillation"
```

---

### Task 13: Config loading (`config.py`)

**Files:**
- Create: `sentinel/config.py`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

```python
import json

import pytest

from sentinel.config import Settings, load_settings, load_towers


def test_missing_settings_file_gives_defaults(tmp_path):
    assert load_settings(tmp_path / "none.json") == Settings()


def test_settings_override(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"min_frames": 5, "vlm_model": "context"}))
    s = load_settings(p)
    assert s.min_frames == 5 and s.vlm_model == "context"


def test_unknown_setting_is_rejected(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"min_frame": 5}))
    with pytest.raises(ValueError, match="min_frame"):
        load_settings(p)


def test_load_towers(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"id": "t1", "name": "A", "lat": 1.0, "lon": 2.0, "source": "x"}]))
    towers = load_towers(p)
    assert towers["t1"].name == "A" and towers["t1"].benign_zones == []
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
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
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add sentinel/config.py tests/test_config.py && git commit -m "feat: settings and tower config"
```

---

### Task 14: Dispatch report (`report.py`)

Facts are templated by code. The VLM supplies only the one-sentence description (token saving #3).

**Files:**
- Create: `sentinel/report.py`
- Test: `tests/test_report.py`

**Step 1: Write the failing test**

```python
from sentinel.config import Tower
from sentinel.report import build_report, compass, render_text
from sentinel.schema import ContextResult, Severity

TOWER = Tower(id="t1", name="Mt. Umunhum Lookout", lat=37.1606, lon=-121.8983, source="", temp_c=34.0)
CTX = ContextResult(source_type="wildland", smoke_color="grey", attended="no",
                    near_structures=True, near_road=False, size_estimate="medium",
                    description="Grey smoke column rising from a brushy slope above homes.")


def report(ctx=CTX):
    return build_report("e1", TOWER, Severity.ALERT, ctx, "growing", 0.914, now=0.0, thumbnail_b64="")


def test_report_contains_facts():
    r = report()
    assert r["severity"] == "ALERT" and r["lat"] == 37.1606 and r["confidence"] == 0.914
    assert r["detected_at"].startswith("1970-01-01T00:00:00")
    assert r["forecast"] is None


def test_render_text_without_forecast():
    text = render_text(report())
    assert "ALERT" in text and "Wildland" in text and "37.1606" in text and "homes" in text


def test_render_text_with_forecast():
    r = report()
    r["forecast"] = {"temp_c": [34, 36, 38], "wind_mph": [15, 16, 18], "wind_dir_deg": [270, 270, 280]}
    assert "38°C" in render_text(r) and "from W" in render_text(r)


def test_render_text_forecast_pending():
    r = report()
    r["forecast_status"] = "pending"
    assert "Forecast pending" in render_text(r)


def test_report_without_context():
    r = report(ctx=None)
    assert r["source_type"] == "unknown" and "unavailable" in r["description"]


def test_compass():
    assert compass(0) == "N" and compass(270) == "W" and compass(359) == "N" and compass(135) == "SE"
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Dispatch report: facts templated by code, scene description from the VLM."""
from datetime import datetime, timezone

from sentinel.config import Tower
from sentinel.schema import ContextResult, Severity

FALLBACK_DESCRIPTION = "Context model unavailable; detector-only assessment."
_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def compass(deg: float) -> str:
    return _DIRS[int((deg % 360) / 45 + 0.5) % 8]


def build_report(event_id: str, tower: Tower, severity: Severity, ctx: ContextResult | None,
                 trend: str | None, confidence: float, now: float, thumbnail_b64: str) -> dict:
    return {
        "event_id": event_id,
        "tower_id": tower.id,
        "tower_name": tower.name,
        "lat": tower.lat,
        "lon": tower.lon,
        "detected_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "severity": severity.name,
        "confidence": round(confidence, 3),
        "trend": trend,
        "temp_c": tower.temp_c,
        "source_type": ctx.source_type if ctx else "unknown",
        "context": ctx.model_dump(exclude={"description"}) if ctx else None,
        "description": ctx.description if ctx else FALLBACK_DESCRIPTION,
        "forecast": None,
        "forecast_status": "not_requested",
        "thumbnail_jpeg_b64": thumbnail_b64,
    }


def render_text(r: dict) -> str:
    parts = [
        f"[{r['severity']}] {r['source_type'].replace('_', ' ').title()} smoke/fire at "
        f"{r['tower_name']} ({r['lat']:.4f}, {r['lon']:.4f}).",
        f"Confidence {r['confidence']:.2f}. Trend: {r['trend'] or 'unknown'}. Local temp {r['temp_c']:.0f}°C.",
        r["description"],
    ]
    fc = r.get("forecast")
    if fc:
        parts.append(f"Forecast: up to {max(fc['temp_c']):.0f}°C, wind {fc['wind_mph'][0]:.0f} mph "
                     f"from {compass(fc['wind_dir_deg'][0])}.")
    elif r.get("forecast_status") == "pending":
        parts.append("Forecast pending (no connectivity at detection time).")
    return " ".join(parts)
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_report.py -v`
Expected: 6 passed

**Step 5: Commit**
```bash
git add sentinel/report.py tests/test_report.py && git commit -m "feat: templated dispatch report"
```

---

### Task 15: Offline outbox (`outbox.py`)

**Files:**
- Create: `sentinel/outbox.py`
- Test: `tests/test_outbox.py`

**Step 1: Write the failing test**

```python
from sentinel.outbox import Outbox


def test_enqueue_and_due():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {"a": 1}, now=0)
    assert ob.due(now=0) == [("e1", {"a": 1}, 0)]


def test_enqueue_is_idempotent():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {"a": 1}, now=0)
    ob.enqueue("e1", {"a": 2}, now=0)
    assert ob.pending_count() == 1


def test_mark_sent_removes_from_due():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {}, now=0)
    ob.mark_sent("e1", now=1)
    assert ob.due(now=5) == [] and ob.sent_count() == 1


def test_failure_backs_off():
    ob = Outbox(":memory:")
    ob.enqueue("e1", {}, now=0)
    ob.mark_failed("e1", now=0)
    assert ob.due(now=1) == []
    assert ob.due(now=2)[0][2] == 1


def test_survives_restart(tmp_path):
    path = str(tmp_path / "o.db")
    Outbox(path).enqueue("e1", {"x": 1}, now=0)
    assert Outbox(path).pending_count() == 1
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_outbox.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Durable SQLite outbox for ALERT reports awaiting connectivity."""
import json
import sqlite3
import threading

MAX_BACKOFF_S = 300


class Outbox:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS outbox ("
                " event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,"
                " attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL,"
                " sent_at REAL)")
            self.db.commit()

    def enqueue(self, event_id: str, payload: dict, now: float) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO outbox (event_id, payload, next_attempt_at) VALUES (?, ?, ?)",
                            (event_id, json.dumps(payload), now))
            self.db.commit()

    def due(self, now: float) -> list[tuple[str, dict, int]]:
        with self.lock:
            rows = self.db.execute(
                "SELECT event_id, payload, attempts FROM outbox"
                " WHERE sent_at IS NULL AND next_attempt_at <= ? ORDER BY next_attempt_at",
                (now,)).fetchall()
        return [(e, json.loads(p), a) for e, p, a in rows]

    def mark_sent(self, event_id: str, now: float) -> None:
        with self.lock:
            self.db.execute("UPDATE outbox SET sent_at = ? WHERE event_id = ?", (now, event_id))
            self.db.commit()

    def mark_failed(self, event_id: str, now: float) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE outbox SET attempts = attempts + 1,"
                " next_attempt_at = ? + min(?, 1 << (attempts + 1)) WHERE event_id = ?",
                (now, MAX_BACKOFF_S, event_id))
            self.db.commit()

    def pending_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent_at IS NULL").fetchone()[0]

    def sent_count(self) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM outbox WHERE sent_at IS NOT NULL").fetchone()[0]
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_outbox.py -v`
Expected: 5 passed

**Step 5: Run the full suite and commit**
```bash
pytest   # expect all green
git add sentinel/outbox.py tests/test_outbox.py && git commit -m "feat: durable offline outbox"
git push
```

---

## Day 2: Thursday

### Task 16: LoRA-distill the student VLM (Nano, START FIRST)

**Files:**
- Create: `scripts/train_lora.py`

**Step 1: Free GPU memory.** Stop the teacher (`tmux kill-session -t teacher`). Check the label counts:
```bash
wc -l data/teacher/train.jsonl data/teacher/heldout.jsonl
python -c "import json,collections;print(collections.Counter(json.loads(l)['label']['source_type'] for l in open('data/teacher/train.jsonl')))"
```
Expected: ≥ 1,000 train rows and more than one source_type. If a benign class has fewer than 30 rows, add images to `data/benign` and rerun T12 on just those.

**Step 2: Write the script**

```python
"""LoRA-distill teacher context labels into Qwen2.5-VL-7B. Run on the Nano.
Fallback if TRL/transformers versions disagree: adapt HP's Med_VLM_Fine-Tune_vLLM script
(hack guide p.6) to this JSONL; the data format below is the same chat format."""
import json

from datasets import Dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

from sentinel.vlm_client import SYSTEM_PROMPT, USER_PROMPT

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"


def to_example(row: dict) -> dict:
    return {
        "images": [Image.open(row["image"]).convert("RGB")],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": USER_PROMPT}]},
            {"role": "assistant", "content": [{"type": "text", "text": json.dumps(row["label"])}]},
        ],
    }


def main() -> None:
    rows = [json.loads(line) for line in open("data/teacher/train.jsonl")]
    ds = Dataset.from_list([to_example(r) for r in rows])

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(BASE, torch_dtype="bfloat16")
    processor = AutoProcessor.from_pretrained(BASE, max_pixels=448 * 448)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])  # language model only
    args = SFTConfig(output_dir="adapters/context", num_train_epochs=2,
                     per_device_train_batch_size=4, gradient_accumulation_steps=4,
                     learning_rate=1e-4, bf16=True, logging_steps=10, save_strategy="epoch",
                     gradient_checkpointing=True, max_length=None, report_to="none")
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                         processing_class=processor, peft_config=peft_config)
    trainer.train()
    trainer.save_model("adapters/context")


if __name__ == "__main__":
    main()
```

**Step 3: Run**
```bash
tmux new -s lora
python scripts/train_lora.py 2>&1 | tee results/lora_train.log
```
Expected: loss decreasing in the logs, and `adapters/context/adapter_config.json` exists at the end. If it errors within 20 minutes, switch to the HP Med VLM script fallback. Don't debug TRL for more than 45 minutes; the cut line is serving the base model.

**Step 4: Commit**
```bash
git add scripts/train_lora.py && git commit -m "feat: LoRA distillation script"
```

---

### Task 17: Metrics (`metrics.py`)

**Files:**
- Create: `sentinel/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write the failing test**

```python
from sentinel.metrics import Metrics


def test_counters_and_percentiles():
    m = Metrics()
    m.inc("frames")
    m.inc("frames", 2)
    for v in range(1, 101):
        m.time("detect_ms", v)
    s = m.summary()
    assert s["frames"] == 3 and s["detect_ms_p50"] == 50.5 and s["detect_ms_p95"] > 95


def test_merge():
    a, b = Metrics(), Metrics()
    a.inc("x"); b.inc("x", 2); b.time("t", 1.0)
    a.merge(b)
    assert a.counters["x"] == 3 and a.timings["t"] == [1.0]
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Counters and latency samples for benchmarks and the dashboard."""
from collections import Counter, defaultdict

import numpy as np


class Metrics:
    def __init__(self):
        self.counters: Counter = Counter()
        self.timings: dict[str, list[float]] = defaultdict(list)

    def inc(self, key: str, n: int = 1) -> None:
        self.counters[key] += n

    def time(self, key: str, value: float) -> None:
        self.timings[key].append(value)

    def merge(self, other: "Metrics") -> None:
        self.counters.update(other.counters)
        for k, v in other.timings.items():
            self.timings[k].extend(v)

    def summary(self) -> dict:
        out: dict = dict(self.counters)
        for k, v in self.timings.items():
            out[f"{k}_p50"] = float(np.percentile(v, 50))
            out[f"{k}_p95"] = float(np.percentile(v, 95))
        return out
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: 2 passed

**Step 5: Commit**
```bash
git add sentinel/metrics.py tests/test_metrics.py && git commit -m "feat: metrics collection"
```

---

### Task 18: Cloud clients (`cloud.py`)

**Files:**
- Create: `sentinel/cloud.py`
- Test: `tests/test_cloud.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_cloud.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Cloud-side calls, used ONLY by the escalator when the link is up."""
import json
from pathlib import Path

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def parse_open_meteo(data: dict) -> dict:
    h = data["hourly"]
    return {"time": h["time"], "temp_c": h["temperature_2m"],
            "wind_mph": h["wind_speed_10m"], "wind_dir_deg": h["wind_direction_10m"]}


def fetch_forecast(lat: float, lon: float, hours: int = 6, timeout_s: float = 3.0) -> dict:
    params = {"latitude": lat, "longitude": lon,
              "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
              "wind_speed_unit": "mph", "forecast_hours": hours}
    r = httpx.get(OPEN_METEO_URL, params=params, timeout=timeout_s)
    r.raise_for_status()
    return parse_open_meteo(r.json())


def send_dispatch(url: str, payload: dict, timeout_s: float = 5.0) -> None:
    httpx.post(url, json=payload, timeout=timeout_s).raise_for_status()


def fetch_burn_schedule(path: str | Path = "config/burn_schedule.json") -> set[str]:
    """Simulated cloud service: tower ids with a prescribed burn scheduled today."""
    p = Path(path)
    return set(json.loads(p.read_text())) if p.exists() else set()
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_cloud.py -v`
Expected: 2 passed

**Step 5: Live check (laptop, online)**
```bash
python -c "from sentinel.cloud import fetch_forecast; print(fetch_forecast(37.16, -121.90))"
```
Expected: a dict with 6 hourly values.

**Step 6: Commit**
```bash
git add sentinel/cloud.py tests/test_cloud.py && git commit -m "feat: forecast, dispatch and burn-schedule clients"
```

---

### Task 19: Escalation policy (`escalation.py`)

**Files:**
- Create: `sentinel/escalation.py`
- Test: `tests/test_escalation.py`

**Step 1: Write the failing test**

```python
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.schema import Severity

REPORT = {"event_id": "e1", "lat": 37.1, "lon": -121.9, "forecast": None}


def make(online=False, forecast_fn=lambda lat, lon: {"temp_c": [34]}, send_fn=None):
    sent = []
    esc = Escalator(Outbox(":memory:"), Link(online=online), forecast_fn,
                    send_fn or sent.append, Metrics())
    return esc, sent


def test_alert_queues_offline_and_sends_when_online():
    esc, sent = make()
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=1) == 0 and sent == []
    esc.link.online = True
    assert esc.flush(now=2) == 1
    assert sent[0]["forecast"] == {"temp_c": [34]} and sent[0]["forecast_status"] == "ok"
    assert esc.metrics.counters["bytes_up"] > 0


def test_non_alerts_never_reach_the_cloud():
    esc, sent = make(online=True)
    for i, sev in enumerate([Severity.IGNORE, Severity.LOG, Severity.MONITOR]):
        esc.handle({**REPORT, "event_id": f"e{i}"}, sev, now=0)
    assert esc.flush(now=1) == 0 and sent == []
    assert len(esc.local_log) == 2


def test_forecast_failure_still_sends_alert():
    def boom(lat, lon):
        raise TimeoutError
    esc, sent = make(online=True, forecast_fn=boom)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=0) == 1 and sent[0]["forecast_status"] == "pending"


def test_send_failure_is_retried_later():
    def down(payload):
        raise ConnectionError
    esc, _ = make(online=True, send_fn=down)
    esc.handle(dict(REPORT), Severity.ALERT, now=0)
    assert esc.flush(now=0) == 0
    assert esc.outbox.pending_count() == 1 and esc.outbox.due(now=1) == []
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_escalation.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""The explicit edge→cloud escalation policy.

Rule: only ALERT events leave the device, and the cloud is contacted only for what the
edge cannot produce (forecast) plus delivery to dispatch. Video never leaves the device.
"""
import json
from dataclasses import dataclass
from typing import Callable

from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.schema import Severity


@dataclass
class Link:
    online: bool = False


class Escalator:
    def __init__(self, outbox: Outbox, link: Link,
                 forecast_fn: Callable[[float, float], dict],
                 send_fn: Callable[[dict], None], metrics: Metrics):
        self.outbox = outbox
        self.link = link
        self.forecast_fn = forecast_fn
        self.send_fn = send_fn
        self.metrics = metrics
        self.local_log: list[dict] = []

    def handle(self, report: dict, severity: Severity, now: float) -> None:
        if severity == Severity.ALERT:
            self.outbox.enqueue(report["event_id"], report, now)
        elif severity in (Severity.LOG, Severity.MONITOR):
            self.local_log.append(report)

    def flush(self, now: float) -> int:
        if not self.link.online:
            return 0
        sent = 0
        for event_id, payload, _ in self.outbox.due(now):
            if payload.get("forecast") is None:
                try:
                    payload["forecast"] = self.forecast_fn(payload["lat"], payload["lon"])
                    payload["forecast_status"] = "ok"
                    self.metrics.inc("cloud_forecast_calls")
                except Exception:
                    payload["forecast_status"] = "pending"
            try:
                self.send_fn(payload)
            except Exception:
                self.outbox.mark_failed(event_id, now)
                continue
            self.outbox.mark_sent(event_id, now)
            self.metrics.inc("alerts_sent")
            self.metrics.inc("bytes_up", len(json.dumps(payload).encode()))
            sent += 1
        return sent
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_escalation.py -v`
Expected: 4 passed

**Step 5: Commit**
```bash
git add sentinel/escalation.py tests/test_escalation.py && git commit -m "feat: explicit escalation policy"
```

---

### Task 20: Frame replayer and detector wrapper

**Files:**
- Create: `sentinel/replayer.py`, `sentinel/detector.py`
- Test: `tests/test_replayer.py`

**Step 1: Write the failing test**

```python
import itertools

import cv2
import numpy as np

from sentinel.replayer import frames


def test_image_dir_loops(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.full((10, 10, 3), i * 50, np.uint8))
    got = list(itertools.islice(frames(str(tmp_path), fps=2), 4))
    assert len(got) == 4 and abs(int(got[3][0, 0, 0]) - 0) < 5


def test_image_dir_no_loop(tmp_path):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.jpg"), np.zeros((10, 10, 3), np.uint8))
    assert len(list(frames(str(tmp_path), fps=2, loop=False))) == 3
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_replayer.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement `sentinel/replayer.py`**

```python
"""Replays a video file or a folder of time-ordered images as a tower camera."""
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def frames(source: str, fps: float, loop: bool = True) -> Iterator[np.ndarray]:
    path = Path(source)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        while True:
            for f in files:
                img = cv2.imread(str(f))
                if img is not None:
                    yield img
            if not loop:
                return
    cap = cv2.VideoCapture(str(path))
    step = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 30) / fps))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            if not loop:
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        if i % step == 0:
            yield img
        i += 1
```

**Step 4: Implement `sentinel/detector.py`** (integration-tested on the Nano, no unit test)

```python
"""Stage 1: YOLO smoke/fire detector."""
import numpy as np

from sentinel.schema import Detection


class YoloDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO  # imported lazily so laptop tests don't need torch
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [Detection(r.names[int(c)], float(p), tuple(b))
                for b, p, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
```

**Step 5: Run tests, then a Nano smoke test**

Run: `pytest tests/test_replayer.py -v`
Expected: 2 passed

On the Nano:
```bash
python -c "import cv2,glob; from sentinel.detector import YoloDetector; d=YoloDetector('models/smoke_yolo.pt'); print(d(cv2.imread(sorted(glob.glob('data/dfire/test/images/*.jpg'))[0])))"
```
Expected: a list of `Detection(...)`, possibly empty for a negative image.

**Step 6: Commit**
```bash
git add sentinel/replayer.py sentinel/detector.py tests/test_replayer.py
git commit -m "feat: frame replayer and YOLO detector wrapper"
```

---

### Task 21: Pipeline orchestration (`pipeline.py`)

The pipeline makes one VLM call per event, not per frame. Clear ALERTs (dangerous source, near structures or black smoke) and clear IGNOREs finalize immediately, which is the latency win. Everything else waits for a trend re-check.

**Files:**
- Create: `sentinel/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
import numpy as np

from sentinel.config import Settings, Tower
from sentinel.pipeline import Pipeline
from sentinel.schema import ContextResult, Detection, Severity

FRAME = np.zeros((480, 640, 3), np.uint8)
TOWER = Tower(id="t1", name="Test", lat=37.0, lon=-121.0, source="", temp_c=30.0)
SMALL = Detection("smoke", 0.9, (100, 100, 200, 200))
BIG = Detection("smoke", 0.9, (100, 100, 250, 250))


def ctx(**kw):
    base = dict(source_type="wildland", smoke_color="grey", attended="no", near_structures=False,
                near_road=False, size_estimate="small", description="d")
    base.update(kw)
    return ContextResult(**base)


class FakeVLM:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def classify(self, jpeg):
        self.calls += 1
        return self.result, 300


class RecordingEscalator:
    def __init__(self):
        self.handled = []

    def handle(self, report, severity, now):
        self.handled.append((severity, report))


def run(result, schedule):
    """schedule: list of (time, detections)."""
    vlm, esc = FakeVLM(result), RecordingEscalator()
    dets = iter([d for _, d in schedule])
    p = Pipeline({"t1": TOWER}, lambda frame: next(dets), vlm, esc,
                 Settings(min_frames=3, recheck_s=30, max_rechecks=2, cooldown_s=0))
    for t, _ in schedule:
        p.process("t1", FRAME, now=t)
    return p, vlm, esc


def steady(n=3, det=SMALL):
    return [(t, [det]) for t in range(n)]


def test_clear_danger_alerts_immediately():
    _, vlm, esc = run(ctx(near_structures=True), steady())
    assert [s for s, _ in esc.handled] == [Severity.ALERT] and vlm.calls == 1


def test_fog_is_ignored_immediately():
    _, _, esc = run(ctx(source_type="fog_dust_cloud"), steady())
    assert [s for s, _ in esc.handled] == [Severity.IGNORE]


def test_campfire_waits_for_trend_then_logs():
    p, _, esc = run(ctx(source_type="campfire", attended="yes"), steady() + [(32, [SMALL])])
    assert [s for s, _ in esc.handled] == [Severity.LOG]
    assert p.history[0].trend == "static"


def test_small_wildland_growing_on_recheck_alerts():
    _, _, esc = run(ctx(), steady() + [(32, [BIG])])
    assert [s for s, _ in esc.handled] == [Severity.ALERT]


def test_vlm_failure_falls_back_and_still_alerts_when_growing():
    _, _, esc = run(None, steady() + [(32, [BIG])])
    severity, report = esc.handled[0]
    assert severity == Severity.ALERT and report["source_type"] == "unknown"


def test_one_vlm_call_per_event_not_per_frame():
    _, vlm, _ = run(ctx(source_type="campfire", attended="yes"),
                    steady(n=20) + [(32, [SMALL])])
    assert vlm.calls == 1


def test_monitor_finalizes_after_max_rechecks():
    _, _, esc = run(ctx(), steady() + [(32, [SMALL]), (62, [SMALL])])
    assert [s for s, _ in esc.handled] == [Severity.MONITOR]
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement**

```python
"""Stage orchestration: detect → gate → context VLM → trend → severity → escalate."""
import base64
import time
import uuid
from dataclasses import dataclass

import numpy as np

from sentinel.config import Settings, Tower
from sentinel.gate import PersistenceGate
from sentinel.imaging import crop_box, to_jpeg
from sentinel.metrics import Metrics
from sentinel.report import build_report, render_text
from sentinel.schema import ContextResult, Detection, Severity
from sentinel.severity import assess, fallback_severity
from sentinel.trend import classify_trend
from sentinel.zones import in_any_zone


@dataclass
class Event:
    id: str
    tower_id: str
    opened_at: float
    confidence: float
    last_area: float
    thumbnail_b64: str
    in_zone: bool = False
    ctx: ContextResult | None = None
    trend: str | None = None
    severity: Severity | None = None
    rechecks: int = 0
    next_check_at: float = 0.0
    report: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "tower_id": self.tower_id, "opened_at": self.opened_at,
            "severity": self.severity.name if self.severity is not None else None,
            "trend": self.trend, "context": self.ctx.model_dump() if self.ctx else None,
            "report_text": render_text(self.report) if self.report else None,
            "thumbnail_b64": self.thumbnail_b64, "final": self.report is not None,
        }


class Pipeline:
    def __init__(self, towers: dict[str, Tower], detector, vlm, escalator,
                 settings: Settings, metrics: Metrics | None = None):
        self.towers = towers
        self.detector = detector
        self.vlm = vlm
        self.escalator = escalator
        self.s = settings
        self.metrics = metrics or Metrics()
        self.gate = PersistenceGate(settings.min_conf, settings.min_frames, settings.cooldown_s)
        self.active: dict[str, Event] = {}
        self.history: list[Event] = []
        self.latest: dict[str, Detection | None] = {}
        self.last_frame: dict[str, tuple[np.ndarray, Detection | None]] = {}
        self.burn_towers: set[str] = set()

    def process(self, tower_id: str, frame: np.ndarray, now: float) -> None:
        self.metrics.inc("frames")
        t0 = time.perf_counter()
        dets = self.detector(frame)
        self.metrics.time("detect_ms", (time.perf_counter() - t0) * 1000)
        strong = [d for d in dets if d.conf >= self.s.min_conf]
        best = max(strong, key=lambda d: d.conf, default=None)
        self.latest[tower_id] = best
        self.last_frame[tower_id] = (frame, best)

        ev = self.active.get(tower_id)
        if ev is None:
            candidate = self.gate.update(tower_id, dets, now)
            if candidate is not None:
                self._open(tower_id, frame, candidate, now)
        elif now >= ev.next_check_at:
            self._recheck(ev, now)

    def _open(self, tower_id: str, frame: np.ndarray, det: Detection, now: float) -> None:
        self.metrics.inc("candidates")
        tower = self.towers[tower_id]
        if self.s.full_frame:
            h, w = frame.shape[:2]
            crop = crop_box(frame, (0, 0, w, h), 0.0, self.s.full_frame_max_side)
        else:
            crop = crop_box(frame, det.box, self.s.crop_pad, self.s.crop_max_side)
        thumb = base64.b64encode(to_jpeg(crop_box(frame, det.box, 0.5, 256), 70)).decode()
        ev = Event(id=uuid.uuid4().hex[:12], tower_id=tower_id, opened_at=now,
                   confidence=det.conf, last_area=det.area, thumbnail_b64=thumb,
                   in_zone=in_any_zone(det.box, tower.benign_zones))

        t0 = time.perf_counter()
        ev.ctx, tokens = self.vlm.classify(to_jpeg(crop))
        self.metrics.time("vlm_ms", (time.perf_counter() - t0) * 1000)
        self.metrics.inc("vlm_calls")
        self.metrics.inc("vlm_tokens", tokens)
        if ev.ctx is None:
            self.metrics.inc("vlm_failures")

        severity = self._assess(ev, trend=None)
        if severity in (Severity.ALERT, Severity.IGNORE):
            self._finalize(ev, severity, now)
            return
        ev.severity = severity  # provisional until the trend re-check
        ev.next_check_at = now + self.s.recheck_s
        self.active[tower_id] = ev

    def _recheck(self, ev: Event, now: float) -> None:
        current = self.latest.get(ev.tower_id)
        area = current.area if current else 0.0
        ev.trend = classify_trend(ev.last_area, area)
        ev.last_area = area
        ev.rechecks += 1
        severity = self._assess(ev, ev.trend)
        if severity == Severity.MONITOR and ev.rechecks < self.s.max_rechecks:
            ev.severity = severity
            ev.next_check_at = now + self.s.recheck_s
            return
        del self.active[ev.tower_id]
        self._finalize(ev, severity, now)

    def _assess(self, ev: Event, trend: str | None) -> Severity:
        if ev.ctx is None:
            return fallback_severity(trend)
        return assess(ev.ctx, trend, ev.in_zone, ev.tower_id in self.burn_towers)

    def _finalize(self, ev: Event, severity: Severity, now: float) -> None:
        ev.severity = severity
        ev.report = build_report(ev.id, self.towers[ev.tower_id], severity, ev.ctx,
                                 ev.trend, ev.confidence, now, ev.thumbnail_b64)
        self.metrics.inc(f"severity_{severity.name}")
        self.metrics.time("decision_s", now - ev.opened_at)
        self.history.append(ev)
        self.escalator.handle(ev.report, severity, now)
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: 7 passed

**Step 5: Commit**
```bash
git add sentinel/pipeline.py tests/test_pipeline.py && git commit -m "feat: cascaded pipeline orchestration"
```

---

### Task 22: Dashboard API and page (`app.py`)

**Files:**
- Create: `sentinel/app.py`, `sentinel/static/dashboard.html`
- Test: `tests/test_app.py`

**Step 1: Write the failing test**

```python
import numpy as np
from fastapi.testclient import TestClient

from sentinel.app import Runtime, create_app
from sentinel.config import Settings, Tower
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.schema import ContextResult, Detection

CTX = ContextResult(source_type="wildland", smoke_color="black", attended="no", near_structures=False,
                    near_road=False, size_estimate="small", description="d")


class FakeVLM:
    def classify(self, jpeg):
        return CTX, 300


def runtime(tmp_path):
    metrics, link = Metrics(), Link()
    esc = Escalator(Outbox(":memory:"), link, lambda a, b: {}, lambda p: None, metrics)
    det = [Detection("smoke", 0.9, (10, 10, 50, 50))]
    pipe = Pipeline({"t1": Tower("t1", "Test", 1.0, 2.0, "")}, lambda f: det, FakeVLM(), esc,
                    Settings(min_frames=1), metrics)
    return Runtime(pipe, esc, link, feedback_path=tmp_path / "fb.jsonl")


def test_state_and_link_toggle(tmp_path):
    client = TestClient(create_app(runtime(tmp_path)))
    assert client.get("/api/state").json()["online"] is False
    assert client.post("/api/link", json={"online": True}).json() == {"online": True}
    assert client.get("/api/state").json()["online"] is True


def test_frame_event_and_feedback(tmp_path):
    rt = runtime(tmp_path)
    client = TestClient(create_app(rt))
    assert client.get("/api/frame/t1").status_code == 404
    rt.pipeline.process("t1", np.zeros((100, 100, 3), np.uint8), now=0)
    assert client.get("/api/frame/t1").headers["content-type"] == "image/jpeg"
    state = client.get("/api/state").json()
    assert state["events"][0]["severity"] == "ALERT" and state["outbox"]["pending"] == 1
    event_id = state["events"][0]["id"]
    assert client.post(f"/api/feedback/{event_id}", json={"label": "false_alarm"}).status_code == 200
    assert "false_alarm" in rt.feedback_path.read_text()


def test_index_serves_dashboard(tmp_path):
    assert "Wildfire Edge Sentinel" in TestClient(create_app(runtime(tmp_path))).get("/").text
```

**Step 2: Run to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL, `ModuleNotFoundError`

**Step 3: Implement `sentinel/app.py`**

```python
"""Demo dashboard API: live feeds, events, outbox, link toggle, ranger feedback."""
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from sentinel.escalation import Escalator, Link
from sentinel.imaging import to_jpeg
from sentinel.pipeline import Pipeline

DASHBOARD = Path(__file__).parent / "static" / "dashboard.html"


@dataclass
class Runtime:
    pipeline: Pipeline
    escalator: Escalator
    link: Link
    feedback_path: Path = Path("data/feedback.jsonl")
    lock: threading.Lock = field(default_factory=threading.Lock)


class LinkState(BaseModel):
    online: bool


class Feedback(BaseModel):
    label: Literal["correct", "false_alarm"]


def create_app(rt: Runtime) -> FastAPI:
    app = FastAPI(title="Wildfire Edge Sentinel")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return DASHBOARD.read_text()

    @app.get("/api/state")
    def state():
        with rt.lock:
            p = rt.pipeline
            return {
                "online": rt.link.online,
                "towers": [{"id": t.id, "name": t.name} for t in p.towers.values()],
                "active": [e.to_dict() for e in p.active.values()],
                "events": [e.to_dict() for e in reversed(p.history[-30:])],
                "outbox": {"pending": rt.escalator.outbox.pending_count(),
                           "sent": rt.escalator.outbox.sent_count()},
                "metrics": p.metrics.summary(),
            }

    @app.post("/api/link")
    def set_link(s: LinkState):
        rt.link.online = s.online
        return {"online": rt.link.online}

    @app.get("/api/frame/{tower_id}")
    def frame(tower_id: str):
        with rt.lock:
            item = rt.pipeline.last_frame.get(tower_id)
        if item is None:
            raise HTTPException(404, "no frame yet")
        img, det = item[0].copy(), item[1]
        if det is not None:
            x1, y1, x2, y2 = map(int, det.box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{det.cls} {det.conf:.2f}", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return Response(to_jpeg(img, 75), media_type="image/jpeg")

    @app.post("/api/feedback/{event_id}")
    def feedback(event_id: str, fb: Feedback):
        with rt.lock:
            ev = next((e for e in rt.pipeline.history if e.id == event_id), None)
        if ev is None:
            raise HTTPException(404, "unknown event")
        rt.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with rt.feedback_path.open("a") as f:
            f.write(json.dumps({"event_id": event_id, "label": fb.label,
                                "context": ev.ctx.model_dump() if ev.ctx else None,
                                "thumbnail_jpeg_b64": ev.thumbnail_b64}) + "\n")
        return {"ok": True}

    return app
```

**Step 4: Create `sentinel/static/dashboard.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wildfire Edge Sentinel</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2a2e37;
          --alert:#e5484d; --monitor:#f5a524; --log:#3b82f6; --ignore:#6b7280; --ok:#22c55e; }
  body { margin:0; font-family:system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { display:flex; justify-content:space-between; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); }
  h1 { font-size:18px; margin:0; } h2 { font-size:15px; margin:0 0 8px; }
  main { display:grid; grid-template-columns:2fr 1fr; gap:16px; padding:16px; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } }
  .card { background:var(--card); border-radius:8px; padding:12px; margin-bottom:16px; }
  .towers { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:12px; }
  .towers img { width:100%; border-radius:6px; background:#000; min-height:120px; }
  figure { margin:0; } figcaption { color:var(--muted); font-size:13px; }
  .sev { display:inline-block; padding:2px 8px; border-radius:4px; font-weight:600; font-size:12px; }
  .ALERT { background:var(--alert); } .MONITOR { background:var(--monitor); color:#000; }
  .LOG { background:var(--log); } .IGNORE { background:var(--ignore); }
  .event { display:flex; gap:10px; padding:8px 0; border-bottom:1px solid var(--line); }
  .event img { width:96px; height:72px; object-fit:cover; border-radius:4px; flex:none; }
  button { font:inherit; padding:6px 12px; border-radius:6px; border:0; cursor:pointer; }
  #link.online { background:var(--ok); } #link.offline { background:var(--alert); color:#fff; }
  dl { display:grid; grid-template-columns:auto auto; gap:4px 12px; margin:0; font-size:13px; }
  dt { color:var(--muted); } dd { margin:0; }
</style>
</head>
<body>
<header><h1>🔥 Wildfire Edge Sentinel</h1><button id="link">…</button></header>
<main>
  <section>
    <div class="card"><h2>Tower feeds</h2><div class="towers" id="towers"></div></div>
    <div class="card"><h2>Events</h2><div id="events"></div></div>
  </section>
  <aside>
    <div class="card"><h2>Uplink outbox</h2><dl id="outbox"></dl></div>
    <div class="card"><h2>Metrics</h2><dl id="metrics"></dl></div>
  </aside>
</main>
<script>
  let online = false;
  const $ = (id) => document.getElementById(id);
  const post = (url, body) => fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  $("link").onclick = async () => { await post("/api/link", { online: !online }); refresh(); };
  const dl = (el, obj) => el.innerHTML = Object.entries(obj)
    .map(([k, v]) => `<dt>${k}</dt><dd>${typeof v === "number" ? +v.toFixed(2) : v}</dd>`).join("");
  window.feedback = (id, label) => post(`/api/feedback/${id}`, { label });
  const eventRow = (e) => `<div class="event">
      <img src="data:image/jpeg;base64,${e.thumbnail_b64}" alt="">
      <div><span class="sev ${e.severity}">${e.severity ?? "…"}</span> ${e.final ? "" : "<em>re-checking trend…</em>"}
        <div>${e.report_text ?? (e.context ? e.context.source_type : "classifying…")}</div>
        ${e.final ? `<button onclick="feedback('${e.id}','correct')">✓ Correct</button>
                     <button onclick="feedback('${e.id}','false_alarm')">✗ False alarm</button>` : ""}
      </div></div>`;
  async function refresh() {
    const s = await (await fetch("/api/state")).json();
    online = s.online;
    $("link").textContent = online ? "Link: ONLINE (click to cut)" : "Link: OFFLINE (click to restore)";
    $("link").className = online ? "online" : "offline";
    if (!$("towers").children.length) {
      $("towers").innerHTML = s.towers.map((t) =>
        `<figure><img id="f-${t.id}" alt="${t.name}"><figcaption>${t.name}</figcaption></figure>`).join("");
    }
    const ts = Date.now();
    s.towers.forEach((t) => { $("f-" + t.id).src = `/api/frame/${t.id}?t=${ts}`; });
    $("events").innerHTML = [...s.active, ...s.events].map(eventRow).join("") || "<p>No events yet.</p>";
    dl($("outbox"), s.outbox);
    dl($("metrics"), s.metrics);
  }
  setInterval(refresh, 1000);
  refresh();
</script>
</body>
</html>
```

**Step 5: Run to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: 3 passed

**Step 6: Commit**
```bash
git add sentinel/app.py sentinel/static/dashboard.html tests/test_app.py
git commit -m "feat: demo dashboard"
```

---

### Task 23: Runtime entrypoint, dispatch stub, end-to-end on the Nano

**Files:**
- Create: `sentinel/main.py`, `scripts/dispatch_stub.py`, `scripts/run_all.sh`

**Step 1: Implement `sentinel/main.py`**

```python
"""Wires real components and runs the replay loop + dashboard. `python -m sentinel.main`"""
import threading
import time

import uvicorn

from sentinel.app import Runtime, create_app
from sentinel.cloud import fetch_burn_schedule, fetch_forecast, send_dispatch
from sentinel.config import Settings, load_settings, load_towers
from sentinel.detector import YoloDetector
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.replayer import frames
from sentinel.vlm_client import ContextVLM

FLUSH_EVERY_S = 2.0


def run_loop(rt: Runtime, settings: Settings, stop: threading.Event) -> None:
    streams = {tid: frames(t.source, settings.fps) for tid, t in rt.pipeline.towers.items()}
    period = 1.0 / settings.fps
    last_flush = 0.0
    while not stop.is_set():
        tick = time.time()
        for tid, stream in streams.items():
            frame = next(stream)
            with rt.lock:
                rt.pipeline.process(tid, frame, tick)
        if tick - last_flush >= FLUSH_EVERY_S:
            with rt.lock:
                if rt.link.online:
                    rt.pipeline.burn_towers = fetch_burn_schedule()
                rt.escalator.flush(tick)
            last_flush = tick
        time.sleep(max(0.0, period - (time.time() - tick)))


def main() -> None:
    settings = load_settings()
    towers = load_towers()
    metrics, link = Metrics(), Link(online=False)
    escalator = Escalator(Outbox(settings.db_path), link, fetch_forecast,
                          lambda payload: send_dispatch(settings.dispatch_url, payload), metrics)
    pipeline = Pipeline(towers, YoloDetector(settings.detector_weights),
                        ContextVLM(settings.vlm_model, settings.vlm_base_url, settings.vlm_timeout_s),
                        escalator, settings, metrics)
    rt = Runtime(pipeline, escalator, link)
    stop = threading.Event()
    threading.Thread(target=run_loop, args=(rt, settings, stop), daemon=True).start()
    uvicorn.run(create_app(rt), host="0.0.0.0", port=settings.dashboard_port)


if __name__ == "__main__":
    main()
```

**Step 2: Implement `scripts/dispatch_stub.py`** (the simulated cloud dispatch center)

```python
"""Simulated cloud dispatch endpoint. Run: python scripts/dispatch_stub.py"""
import uvicorn
from fastapi import FastAPI

app = FastAPI(title="Dispatch stub (simulated cloud)")
REPORTS: list[dict] = []


@app.post("/dispatch")
def receive(report: dict):
    REPORTS.append(report)
    print(f"[DISPATCH] {report['severity']} {report['tower_name']} {report['event_id']} "
          f"forecast={report.get('forecast_status')}", flush=True)
    return {"received": report["event_id"]}


@app.get("/reports")
def reports():
    return [{k: v for k, v in r.items() if k != "thumbnail_jpeg_b64"} for r in REPORTS]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
```

**Step 3: `scripts/run_all.sh`**

```bash
#!/usr/bin/env bash
# Starts the dispatch stub and the sentinel. Assumes the student VLM is already served on :8000.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/dispatch_stub.py > results/dispatch.log 2>&1 &
python -m sentinel.main
```

**Step 4: Prepare demo sources.** Point `config/towers.json` sources at one FIgLib wildfire sequence (t1) and one benign clip or folder (t2), and add a benign zone polygon on t2 if it contains a stack or campground.

**Step 5: End-to-end run on the Nano**
```bash
chmod +x scripts/*.sh && mkdir -p results && ./scripts/run_all.sh
```
On the laptop, tunnel and open the dashboard:
```bash
ssh -L 8080:localhost:8080 hpX@<nano-ip>    # then open http://localhost:8080
```
Verify this checklist by hand:
- [ ] Both feeds render with boxes
- [ ] The wildfire tower produces an event: "classifying…", then a severity
- [ ] The link is OFFLINE, so the ALERT sits in the outbox (`pending: 1`)
- [ ] Click the link ONLINE: within about 2 s `sent` increments, `results/dispatch.log` shows `forecast=ok`, and the report text includes the forecast
- [ ] The benign tower shows LOG or IGNORE and never reaches dispatch
- [ ] Kill vLLM briefly: events still appear via the fallback (MONITOR/ALERT, "Context model unavailable")

**Step 6: Commit**
```bash
git add sentinel/main.py scripts/dispatch_stub.py scripts/run_all.sh config/towers.json
git commit -m "feat: runtime entrypoint, dispatch stub, end-to-end demo"
```

---

### Task 24: Serve the LoRA adapter and prove distillation

**Files:**
- Create: `scripts/eval_context.py`, `scripts/linear_probe.py`

**Step 1: Re-serve the student with the adapter** (restart the `vlm` tmux session)
```bash
zrt serve hf:Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.30 --enable-prefix-caching \
  --enable-lora --max-lora-rank 16 --lora-modules context=$HOME/sentinel/adapters/context
curl -s localhost:8000/v1/models | python -m json.tool   # expect both base id and "context"
```

**Step 2: Write `scripts/eval_context.py`**

```python
"""Agreement with teacher labels on the held-out split, plus latency and tokens."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sentinel.vlm_client import ContextVLM

GROUP = {"wildland": "danger", "structure": "danger", "vehicle": "danger",
         "controlled_burn": "benign", "campfire": "benign", "bbq_chimney": "benign",
         "industrial_stack": "benign", "fog_dust_cloud": "lookalike", "unknown": "unknown"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--split", default="data/teacher/heldout.jsonl")
    a = ap.parse_args()

    vlm = ContextVLM(a.model, a.base_url, timeout_s=60)
    rows = [json.loads(line) for line in open(a.split)]
    exact = group = fails = 0
    latencies, tokens = [], []
    for r in rows:
        t0 = time.perf_counter()
        ctx, tok = vlm.classify(Path(r["image"]).read_bytes())
        latencies.append((time.perf_counter() - t0) * 1000)
        tokens.append(tok)
        if ctx is None:
            fails += 1
            continue
        want = r["label"]["source_type"]
        exact += ctx.source_type == want
        group += GROUP[ctx.source_type] == GROUP[want]
    out = {"name": a.name, "n": len(rows), "source_type_acc": exact / len(rows),
           "group_acc": group / len(rows), "parse_fail_rate": fails / len(rows),
           "latency_ms_p50": float(np.percentile(latencies, 50)),
           "latency_ms_p95": float(np.percentile(latencies, 95)),
           "tokens_per_call": float(np.mean(tokens))}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/context_{a.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
```

**Step 3: Run base vs adapter (and teacher, if it's still served on :8001)**
```bash
python scripts/eval_context.py --model "hf:Qwen/Qwen2.5-VL-7B-Instruct" --name base7b
python scripts/eval_context.py --model context --name lora7b
# optional: python scripts/eval_context.py --model <teacher id> --base-url http://localhost:8001/v1 --name teacher
```
Expected: `lora7b` group_acc > `base7b` group_acc. That's the headline distillation number. If it's not better, report it honestly and keep the base model.

**Step 4: Write `scripts/linear_probe.py`** (cheap baseline; the first thing to cut)

```python
"""SigLIP embeddings + logistic regression on teacher source_type labels."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoProcessor

MODEL = "google/siglip-base-patch16-224"


def load(path):
    rows = [json.loads(line) for line in open(path)]
    return [r["image"] for r in rows], [r["label"]["source_type"] for r in rows]


def embed(paths, model, proc):
    feats = []
    for i in range(0, len(paths), 64):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + 64]]
        with torch.no_grad():
            f = model.get_image_features(**proc(images=imgs, return_tensors="pt").to("cuda"))
        feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    model = AutoModel.from_pretrained(MODEL).to("cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    xtr, ytr = load("data/teacher/train.jsonl")
    xte, yte = load("data/teacher/heldout.jsonl")
    clf = LogisticRegression(max_iter=2000).fit(embed(xtr, model, proc), ytr)
    t0 = time.perf_counter()
    acc = clf.score(embed(xte, model, proc), yte)
    ms = (time.perf_counter() - t0) * 1000 / len(xte)
    out = {"name": "siglip_linear_probe", "source_type_acc": acc, "latency_ms_per_image": ms}
    Path("results/context_linear_probe.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    main()
```

Run: `python scripts/linear_probe.py`
Expected: accuracy printed. The comparison you want to show is that it's fast but has no reasoning about attendance, structures or description, which is why the VLM earns its cost.

**Step 5: Switch the runtime to the adapter**, if it won:
```json
{"vlm_model": "context"}
```

**Step 6: Commit**
```bash
git add scripts/eval_context.py scripts/linear_probe.py config/settings.json results/context_*.json
git commit -m "feat: distillation eval and linear-probe baseline"
```

---

## Day 3: Friday (submit by 6pm; hard deadline 8pm)

### Task 25: Benchmarks over labeled clips

**Files:**
- Create: `scripts/bench.py`, `data/bench/clips.csv` (not committed, since `data/` is ignored; copy it into `results/clips.csv`)

**Step 1: Build `clips.csv`**, with 20–40 rows: FIgLib and wildfire sequences labeled `alert`, and campfire, fog, stack and BBQ labeled `no_alert`.
```
path,label
data/demo/figlib_seq01,alert
data/demo/benign_campfire01,no_alert
```

**Step 2: Write `scripts/bench.py`**

```python
"""Offline benchmark with a simulated clock (deterministic, no sleeps).
Ablations: --detector-only, --full-frame, --model <base|context>."""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from sentinel.config import Tower, load_settings
from sentinel.detector import YoloDetector
from sentinel.escalation import Escalator, Link
from sentinel.metrics import Metrics
from sentinel.outbox import Outbox
from sentinel.pipeline import Pipeline
from sentinel.replayer import frames
from sentinel.schema import Severity
from sentinel.vlm_client import ContextVLM


class NullVLM:
    def classify(self, jpeg):
        return None, 0


def run_clip(path: str, settings, detector, vlm, metrics: Metrics) -> list:
    tower = Tower(id="bench", name=Path(path).stem, lat=37.0, lon=-121.0, source=path)
    esc = Escalator(Outbox(":memory:"), Link(), lambda *a: {}, lambda p: None, metrics)
    pipe = Pipeline({"bench": tower}, detector, vlm, esc, settings, metrics)
    for i, frame in enumerate(frames(path, settings.fps, loop=False)):
        pipe.process("bench", frame, now=i / settings.fps)
    return [e.severity for e in pipe.history] + [e.severity for e in pipe.active.values()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/bench/clips.csv")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model")
    ap.add_argument("--full-frame", action="store_true")
    ap.add_argument("--detector-only", action="store_true")
    a = ap.parse_args()

    s = load_settings()
    s = replace(s, full_frame=a.full_frame, vlm_model=a.model or s.vlm_model, vlm_timeout_s=30)
    detector = YoloDetector(s.detector_weights)
    vlm = NullVLM() if a.detector_only else ContextVLM(s.vlm_model, s.vlm_base_url, s.vlm_timeout_s)

    total, clips = Metrics(), []
    tp = fp = fn = tn = 0
    for row in csv.DictReader(open(a.clips)):
        m = Metrics()
        severities = run_clip(row["path"], s, detector, vlm, m)
        predicted = m.counters["candidates"] > 0 if a.detector_only else Severity.ALERT in severities
        actual = row["label"] == "alert"
        tp += predicted and actual
        fp += predicted and not actual
        fn += actual and not predicted
        tn += not predicted and not actual
        clips.append({"path": row["path"], "label": row["label"], "predicted_alert": predicted,
                      "severities": [x.name for x in severities if x is not None]})
        total.merge(m)

    summary = total.summary()
    calls = summary.get("vlm_calls", 0)
    out = {"name": a.name, "precision": tp / (tp + fp) if tp + fp else 0.0,
           "recall": tp / (tp + fn) if tp + fn else 0.0, "false_alarms": fp, "missed": fn,
           "tokens_per_vlm_call": summary.get("vlm_tokens", 0) / calls if calls else 0,
           "frames_per_vlm_call": summary.get("frames", 0) / calls if calls else None,
           "metrics": summary, "clips": clips}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/bench_{a.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("name", "precision", "recall", "false_alarms",
                                          "tokens_per_vlm_call", "frames_per_vlm_call")}, indent=2))


if __name__ == "__main__":
    main()
```

**Step 3: Run the ablation matrix** (stop `sentinel.main` first to free the GPU)
```bash
python scripts/bench.py --name detector_only --detector-only
python scripts/bench.py --name base_crop  --model "hf:Qwen/Qwen2.5-VL-7B-Instruct"
python scripts/bench.py --name base_full  --model "hf:Qwen/Qwen2.5-VL-7B-Instruct" --full-frame
python scripts/bench.py --name lora_crop  --model context
```
Expected results table (fill it in from the output):

| Run | Precision | Recall | False alarms | Tokens/VLM call | Frames per VLM call |
|---|---|---|---|---|---|
| detector_only | | | ← expect highest | 0 | — |
| base_full | | | | ← expect highest | |
| base_crop | | | | | |
| lora_crop | | | ← expect lowest | | |

**Step 4: Commit**
```bash
cp data/bench/clips.csv results/clips.csv
git add scripts/bench.py results/ && git commit -m "feat: benchmark harness and results"
```

---

### Task 26: Cost model

**Files:**
- Create: `scripts/cost_model.py`, `config/cost_inputs.json`
- Test: `tests/test_cost_model.py`

**Step 1: Write the failing test**

```python
from scripts.cost_model import compare

P = {"towers": 1, "fps": 1, "tokens_per_full_frame": 1000, "bytes_per_frame": 1_000_000,
     "usd_per_mtok": 1.0, "usd_per_gb": 10.0, "local_vlm_s": 1.0, "detector_s": 0.01,
     "events_per_day": 10, "vlm_calls_per_event": 2, "alerts_per_day": 1, "bytes_per_alert": 50_000}


def test_cloud_every_frame():
    row = compare(P)[0]
    assert row["vlm_calls"] == 86_400 and row["cloud_tokens"] == 86_400_000
    assert row["usd_per_day"] == 86.4 + 864.0


def test_local_every_frame_is_infeasible_at_one_second_per_call():
    row = compare(P)[1]
    assert row["gpu_seconds"] == 86_400 and row["feasible"] is False


def test_cascade():
    row = compare(P)[2]
    assert row["vlm_calls"] == 20 and row["cloud_tokens"] == 0
    assert row["upstream_gb"] == 0.00005 and row["feasible"] is True
```

Also create an empty `scripts/__init__.py` so the test can import it.

**Step 2: Run to verify it fails**

Run: `pytest tests/test_cost_model.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.cost_model'`

**Step 3: Implement**

```python
"""Per-day cost/feasibility: cloud-every-frame vs local-every-frame vs our cascade.
Measured inputs come from results/bench_*.json; prices are filled from current published rates."""
import json
import sys

DAY_S = 86_400


def compare(p: dict) -> list[dict]:
    frames = p["towers"] * p["fps"] * DAY_S
    cloud_tokens = frames * p["tokens_per_full_frame"]
    cloud_gb = frames * p["bytes_per_frame"] / 1e9
    calls = p["events_per_day"] * p["vlm_calls_per_event"]
    cascade_gpu = frames * p["detector_s"] + calls * p["local_vlm_s"]
    cascade_gb = p["alerts_per_day"] * p["bytes_per_alert"] / 1e9
    return [
        {"approach": "cloud VLM every frame", "vlm_calls": frames, "cloud_tokens": cloud_tokens,
         "upstream_gb": cloud_gb, "gpu_seconds": 0, "feasible": True,
         "usd_per_day": round(cloud_tokens / 1e6 * p["usd_per_mtok"] + cloud_gb * p["usd_per_gb"], 4)},
        {"approach": "local VLM every frame", "vlm_calls": frames, "cloud_tokens": 0,
         "upstream_gb": 0.0, "gpu_seconds": frames * p["local_vlm_s"],
         "feasible": frames * p["local_vlm_s"] < DAY_S, "usd_per_day": 0.0},
        {"approach": "cascade (ours)", "vlm_calls": calls, "cloud_tokens": 0,
         "upstream_gb": cascade_gb, "gpu_seconds": cascade_gpu, "feasible": cascade_gpu < DAY_S,
         "usd_per_day": round(cascade_gb * p["usd_per_gb"], 4)},
    ]


if __name__ == "__main__":
    params = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "config/cost_inputs.json"))
    for row in compare(params):
        print(json.dumps(row))
```

**Step 4: Run to verify it passes**

Run: `pytest tests/test_cost_model.py -v`
Expected: 3 passed

**Step 5: Fill in `config/cost_inputs.json` from measurements.** Tokens and latencies come from `results/bench_base_full.json` and `bench_lora_crop.json`. Look up prices from current published pricing and satellite/cellular plan pages; cite the source and date in the README.

```json
{
  "towers": 4, "fps": 2,
  "tokens_per_full_frame": 0, "bytes_per_frame": 0,
  "usd_per_mtok": 0, "usd_per_gb": 0,
  "local_vlm_s": 0, "detector_s": 0,
  "events_per_day": 0, "vlm_calls_per_event": 0,
  "alerts_per_day": 0, "bytes_per_alert": 0
}
```
Then run:
```bash
python scripts/cost_model.py > results/cost_model.jsonl
```

**Step 6: Commit**
```bash
git add scripts/__init__.py scripts/cost_model.py tests/test_cost_model.py config/cost_inputs.json results/cost_model.jsonl
git commit -m "feat: cost and feasibility model"
```

---

### Task 27: README, setup script, reproducibility

**Files:**
- Create: `README.md`, `scripts/setup_nano.sh`

**Step 1: `scripts/setup_nano.sh`**

```bash
#!/usr/bin/env bash
# One-shot setup on a fresh ZGX Nano (the node is wiped after the event).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
  echo "Install CUDA PyTorch for GB10 per the NVIDIA DGX Spark playbook first."; exit 1; }
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt -r requirements-train.txt
command -v zrt >/dev/null || sudo snap install --classic zrt
zrt pull Qwen/Qwen2.5-VL-7B-Instruct
echo "Next: ./scripts/download_data.sh, then see README 'Reproduce'."
```

**Step 2: README sections** (write each one briefly):
1. **Problem & user**: the fire lookout, and why the cloud alone fails (no link, latency, cost).
2. **Architecture diagram**: copy it from the design doc and export a PNG for the deck.
3. **The escalation rule**: one sentence plus the severity/action table.
4. **Results**: detector mAP, the context distillation table (teacher / base / LoRA / linear probe), the benchmark ablation table, the cost model table, and why each metric was chosen.
5. **Reproduce**: setup → data → train detector → teacher label → LoRA → serve → run → bench, with exact commands from T2–T26.
6. **Datasets and licenses**, **models and licenses** (note AGPL for Ultralytics).
7. **Limitations**: simulated connectivity and sensors, teacher labels aren't ground truth, the VLM call blocks the replay loop.

**Step 3: Verify from a clean checkout**
```bash
cd /tmp && git clone <repo> check && cd check && python3 -m venv v && . v/bin/activate \
  && pip install -r requirements-dev.txt && pytest
```
Expected: all tests pass.

**Step 4: Commit**
```bash
git add README.md scripts/setup_nano.sh && git commit -m "docs: README, setup and reproduction" && git push
```

---

### Task 28: Demo, video, presentation, submission (non-code)

- [ ] **5-minute demo script** (record a backup screen capture regardless):
  1. Both towers are live and the link is OFFLINE.
  2. The benign tower shows a campfire, labeled LOG with "attended", and nothing is sent.
  3. The wildfire tower shows smoke, then classifying, then ALERT, and it's queued offline.
  4. Restore the link: the report is delivered with forecast and wind, and the dispatch log prints it.
  5. Show the metrics panel: frames vs VLM calls, tokens per event, bytes sent.
- [ ] **2-minute YouTube video (public)**: problem (15 s) → architecture (20 s) → demo cut (50 s) → numbers (25 s) → impact (10 s).
- [ ] **Interactive presentation** (not static slides; the dashboard itself can be the centerpiece). It needs three required visuals: architecture, benchmarks/evidence, impact.
- [ ] **Pitch lines backed by numbers**: "N frames per VLM call", "X% fewer false alarms than detector-only", "LoRA 7B matches the teacher at Y× speed", "KB instead of GB upstream", "$Z/day vs $W/day".
- [ ] **Submissions**: public GitHub URL, YouTube URL and video file, SJSU Google Drive (brief, video, links, deck, diagrams, photos), and social posts with the required tags.
- [ ] Final `git push` by 6pm. Leave 7–8pm free for upload problems.

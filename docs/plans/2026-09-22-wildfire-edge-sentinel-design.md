# Wildfire Edge Sentinel — Design

**Event:** HP Edge AI SJSUHack · **Deadline:** Fri Sept 25, 8pm · **Hardware:** 1× HP ZGX Nano (NVIDIA GB10, 128 GB unified memory), SSH-only
**Date:** 2026-09-22 · **Status:** Approved design, pre-implementation

---

## 1. Problem & Target User

**User:** Fire lookout / ranger operating remote wildland camera towers.

**Problem:** Towers sit in areas with weak or no connectivity. Streaming video to the cloud for AI analysis is impossible (no link), expensive (satellite/cellular per-MB cost), and slow. Yet early smoke detection is where minutes matter. Naive detectors also flood dispatch with false alarms: campfires, BBQs, chimney smoke, industrial stacks, fog, dust, low cloud.

**Why the cloud alone cannot solve it:** no connectivity, hard latency budget, per-token / per-MB cost ceilings at 24/7 frame rates.

**Solution:** A cascaded edge pipeline on the ZGX Nano that detects smoke/fire, understands *context* (what kind of fire, is it attended, is it growing, is it near structures), assigns severity, writes a dispatch-ready report locally, and escalates to the cloud **only** for data the edge cannot have (forecast, burn schedules) and **only** for confirmed ALERT events.

## 2. Scope

- **In scope:** Outdoor tower cameras (simulated by replaying real wildfire/benign video clips), fixed GPS per tower, simulated local temperature sensor, simulated connectivity toggle, web dashboard.
- **Out of scope:** Indoor sources (lamps, stoves, gas flames), real hardware cameras/radios, multi-Nano coordination, VLM fine-tuning (stretch goal only).

## 3. Architecture

Single Python FastAPI service + local vLLM server (via `zrt serve`). Two processes to keep alive.

```
[Replayer: clips per "tower" + fixed GPS]            (2 fps per tower)
      │
      ▼
Stage 1  YOLO smoke/fire detector (every frame, ms, $0)
         gate: conf ≥ 0.4 AND persists ≥ 3 frames   → else discard
      │ candidate event (grouped by tower + region)
      ▼
Stage 2  Local VLM (Qwen2.5-VL-7B via vLLM), CROPPED region, once per event
         → structured JSON (guided decoding):
           source_type: wildland | structure | vehicle | controlled_burn |
                        campfire | bbq_chimney | industrial_stack |
                        fog_dust_cloud | unknown
           smoke_color: white | grey | black
           attended: yes | no | unclear
           near_structures / near_road: yes | no
           size_estimate: small | medium | large
      │
      ▼
Stage 3  Trend check: re-crop 30–60 s later → growing / static / dissipating
         Severity = deterministic rules(source_type, color, size, growth, zone mask)
         → IGNORE | LOG | MONITOR | ALERT
      │
      ▼
[Report writer]  facts templated by code (GPS, time, confidence, temp);
                 VLM writes only a 1–2 sentence scene description
      │
      ▼
[Escalation policy + SQLite outbox]  →  [Cloud: Open-Meteo forecast, burn schedule,
                                          dispatch webhook (simulated)]
      │
      ▼
[Dashboard: tower feeds, detections, severity, outbox, online/offline toggle, live metrics]
```

**Components**

| Component | Tech | Notes |
|---|---|---|
| Replayer | OpenCV | Loops clips per tower, emits frames at fixed fps |
| Detector | Ultralytics YOLOv8n/s | Fine-tuned on Nano on public wildfire smoke data + benign negatives |
| VLM | Qwen2.5-VL-7B-Instruct via `zrt serve` (vLLM, OpenAI API, :8000) | Guided JSON decoding, prefix caching |
| Severity engine | Plain Python rules | Explicit, auditable, testable |
| Zone masks | JSON per tower | Known benign regions (campground, stack) lower priority |
| Outbox | SQLite | Survives restarts; retry with backoff |
| Cloud clients | Open-Meteo (forecast/wind), burn-schedule stub, dispatch webhook stub | Only called under escalation policy |
| Dashboard | FastAPI + lightweight HTML/JS, exposed via tunneled port | Demo + live metrics |
| Metrics | CSV/SQLite per-stage logs + cost model script | Feeds benchmarks slide |

## 4. Context & Severity Logic

- **Growth over time** is the strongest signal: a campfire stays the same, a wildfire grows.
- **Smoke color:** black suggests fuel/structure/vehicle; white/grey is more ambiguous.
- **Attended:** people or a fire ring nearby suggests campfire/BBQ.
- **Zone masks:** detections inside a known-benign zone are downgraded unless growing.
- **Controlled-burn schedule** (fetched from cloud when online) downgrades matching locations.

| Severity | Examples | Edge action | Cloud action |
|---|---|---|---|
| IGNORE | fog/cloud/dust; known stack in zone, static | Count in metrics | None |
| LOG | attended campfire, BBQ/chimney, static | Store locally | Daily summary when online |
| MONITOR | unclear source, small but new | Re-check every 60 s; escalate on growth | None until upgraded |
| ALERT | growing, uncontrolled, black smoke, near structures/road | Generate & queue report immediately | Fetch forecast + wind + burn schedule; send report + 1 thumbnail |

**Escalation rule (one sentence):** The cloud is used only for data the tower cannot produce (forecast, burn schedules) and only for ALERT events; video never leaves the device.

**Offline ALERT:** report is queued immediately and sent the moment the link returns, marked "forecast pending"; an enriched update follows. Loss of connectivity never delays the alert beyond link restoration.

**Learning loop:** low-confidence crops and ranger corrections (dashboard buttons) are stored locally and used to re-fine-tune the detector on the Nano.

## 5. Economics — Cost, Latency, Tokens

**Frame budget:** 4 towers × 2 fps ≈ 690,000 frames/day.

| Approach | VLM calls/day | Upstream data/day | Paid cloud tokens |
|---|---|---|---|
| Cloud VLM every frame | ~690k | continuous video (tens of GB per camera) | hundreds of millions |
| Local VLM every frame | ~690k | 0 | 0, but GPU saturated |
| **Cascade (this design)** | **~tens (events only)** | **~KB per alert** | **0 for inference** |

Cloud per-token prices and satellite/cellular per-MB rates are **config parameters** in the cost model (filled in from current published rates), not hard-coded assumptions.

**Token & latency optimizations (each measured before/after):**
1. Crop to detection box before VLM → fewer image tokens.
2. One VLM call per event (+ one trend re-check), not per frame.
3. Constrained JSON output with a tight `max_tokens`.
4. vLLM prefix caching for the fixed system prompt.
5. Templated facts; VLM writes only the short description.
6. Upstream payload ≈ 1 KB report + one compressed thumbnail.

## 5a. Models, Fine-Tuning Techniques & Datasets

> Verify each dataset's license and download link before use; record them in the README.

**Models**

| Role | Primary choice | Alternatives | Training technique |
|---|---|---|---|
| Stage 1 detector | YOLO11n/s or YOLOv8n/s (Ultralytics; AGPL-3.0) | RT-DETR (Apache-2.0, if license matters) | Full fine-tune (small model, no adapter needed) |
| Stage 2 context VLM (student) | Qwen2.5-VL-7B-Instruct (known-good in HP's example repo) | Qwen3-VL-8B-Instruct, Gemma 3 12B | LoRA / QLoRA adapter (PEFT + TRL `SFTTrainer`, or Unsloth / LLaMA-Factory) |
| Teacher (offline labeling only) | Largest VL model that fits in FP4 (e.g. Qwen2.5-VL-72B-class) | — | Inference only: generates context-JSON labels |
| Cheap classifier baseline | SigLIP / CLIP image embeddings + logistic regression | — | Linear probe (minutes to train) |

**Techniques and why they matter for the pitch**
- **Teacher→student distillation on-device:** a large local VLM labels thousands of crops with the context JSON schema; a LoRA adapter on the 7–8B student learns it. Smaller model gives faster, cheaper inference with the teacher's quality, and no labeling data leaves the device.
- **LoRA adapters served by vLLM** (`--enable-lora --lora-modules context=./adapters/context`): one base model in memory, swappable task adapters (context classification vs report wording).
- **NVFP4 quantization** of the served student for throughput; benchmark BF16 vs FP4 latency/accuracy.
- **Linear-probe baseline** proves the VLM adds value (or doesn't) — a credible ablation for ML reviewers.

**Datasets**

| Dataset | Content | Use |
|---|---|---|
| D-Fire | ~21k images, YOLO-format fire/smoke boxes, includes negative images | Detector training |
| PyroNear (`pyronear/pyro-sdis` on Hugging Face) | Wildfire smoke from lookout-tower cameras | Detector training/eval — closest to our user |
| FIgLib (HPWREN Fire Ignition Library) | Time-ordered tower image sequences before/after ignition | Growth/trend-check eval, demo replay clips |
| AI For Mankind / HPWREN wildfire smoke (Roboflow) | Bounding-boxed tower smoke images | Small, quick detector sanity set |
| FASDD | Large flame & smoke set, multiple viewpoints | Extra detector data if time allows |
| Benign negatives (campfire, BBQ, chimney, stack, fog, cloud) | Collected from open image sources + D-Fire negatives | VLM context LoRA + false-alarm eval |

## 6. Error Handling

- **VLM down/slow (>5 s timeout):** fall back to YOLO + zone rules; minimum severity MONITOR. Never silently IGNORE.
- **Malformed JSON:** guided decoding prevents it; otherwise retry once, then fallback.
- **Cloud failure:** report stays in SQLite outbox; retry with exponential backoff; no loss on restart.
- **Replayer/clip error:** skip clip, log, continue other towers.

## 7. Testing & Benchmarks

- **Labeled test set:** ~20–40 clips (wildfire + campfire, fog, stack, BBQ). Report ALERT precision/recall and false alarms removed per stage.
- **Latency:** per-stage timings logged; report p50/p95. Target: first smoke frame → queued ALERT < ~10 s.
- **Funnel:** % of frames passing each stage.
- **Tokens:** VLM tokens per event (full frame vs crop).
- **Bandwidth:** bytes upstream per alert vs streaming video.
- **Optional:** energy per event via `nvidia-smi` power sampling.
- **Cost model script:** converts logged counts into cloud-per-frame vs local-per-frame vs cascade comparison.
- **Offline scenario test:** go offline → trigger ALERT → go online → verify delivery then enrichment.
- **Unit tests:** severity rules, escalation policy, outbox retry.

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Benign-class training data scarce | Rely on VLM context classification (zero-shot); YOLO only needs smoke/fire |
| 3-day timeline | Single-service architecture; VLM fine-tune is stretch only |
| Common hackathon theme | Differentiate on context/severity + measured economics + explicit escalation |
| Node wiped at end | Everything in repo: setup script / Dockerfile, model pull commands, dataset download script |
| Shared GPU with teammates | Coordinate training vs serving windows |

## 9. Deliverables Mapping

- **Repo:** code, README, setup script/Dockerfile, dataset + model pull scripts.
- **Metrics:** benchmark CSVs + cost model output + rationale for each metric.
- **Demo (5 min):** dashboard over tunneled port — offline tower, fire grows, ALERT queued, link restored, report + forecast delivered; campfire/fog correctly ignored.
- **Video (2 min):** recorded demo + architecture + numbers.
- **Pitch visuals:** architecture diagram, benchmark/funnel chart, impact (minutes saved, false alarms avoided, bytes/$ saved).

## 10. Next Step

Write the day-by-day implementation plan (writing-plans methodology).

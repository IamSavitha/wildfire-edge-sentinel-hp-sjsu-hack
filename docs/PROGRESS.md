# Wildfire Edge Sentinel — Progress Tracker

**Deadline:** Fri Sept 25, 8pm (internal target 6pm) · **Branch:** `feat/core-pipeline` · **Last updated:** 2026-09-24 02:50 Nano time (ICT) (Nano `spark-d07`, user `hp11`)

Tick a box (`- [x]`) when a step is done. Steps are in the order to run them. Commands and details are in the [README](../README.md) and the [implementation plan](plans/2026-09-22-wildfire-edge-sentinel.md) (task numbers in brackets).

**Where things run:** 💻 laptop · 🖥️ ZGX Nano (over SSH)

---

## Phase 0 — Planning ✅

- [x] Design doc — `docs/plans/2026-09-22-wildfire-edge-sentinel-design.md`
- [x] Implementation plan — `docs/plans/2026-09-22-wildfire-edge-sentinel.md`
- [x] Comparative study (Claude Docs: "Wildfire Edge Sentinel — Comparative Study")
- [x] GitHub repo created: `IamSavitha/wildfire-edge-sentinel-hp-sjsu-hack`

## Phase 1 — Code on the laptop ✅ (144 tests passing)

- [x] Core pipeline: detector gate, context VLM client, severity rules, trend, zones, crops [T5–T11]
- [x] Reports, offline outbox, escalation policy, cloud clients, metrics [T13–T19]
- [x] Replayer, pipeline orchestration, dashboard, runtime entrypoint, dispatch stub [T20–T23]
- [x] Cost model [T26]
- [x] Detector before/after: YOLO-World baseline, `train_detector.py`, `eval_detector.py` [T4, T4b]
- [x] VLM before/after: `teacher_label.py`, `train_lora.py`, `eval_context.py`, `linear_probe.py` [T12, T16, T24, T24b]
- [x] End-to-end: `bench.py`, `compare.py`, data and setup scripts, README [T3, T25, T25b, T27]
- [x] Code review of every batch + final whole-branch review; all Critical/Important issues fixed

## Phase 2 — Repo housekeeping 💻

- [x] Decide: push `feat/core-pipeline` to GitHub, and/or merge into `main` — branch pushed; merge into `main` after results
- [x] (Optional) Live demo warns when `vlm_model` is not served
- [x] (Optional) `run_all.sh`: narrow `trap 'kill 0'` to the stub's pid
- [x] (Optional) `teacher_label.py`: check the server before cutting crops
- [ ] (Optional) Delete or set `sync: false` in `~/Library/Application Support/Ultralytics/settings.json`
- [ ] Check the GitHub "Contributors" sidebar shows only IamSavitha (cache refresh)

## Phase 3 — Nano setup 🖥️ [T2, T3]

- [x] Connect with ZTK; `nvidia-smi` shows GB10
- [x] `git clone` the repo into `~/sentinel`
- [x] `./scripts/setup_nano.sh` → CUDA torch check prints `True`
- [x] `zrt config set proxy.auth.type none && zrt config set proxy.tls.enabled false` (use `sudo` if needed)
- [x] Serve base Qwen2.5-VL-7B on :8000 (tmux `vlm`) — served as model id `base7b` at `--gpu-memory-fraction 0.45`
- [x] `curl localhost:8000/v1/models` → put the exact id in `config/settings.json` as `vlm_model` — `base7b`
- [x] Smoke test: chat returns "OK"; a `json_schema` request returns valid JSON — 7–26 s/call while YOLO trains; base model calls smoke crops "fog/cloud"
- [x] D-Fire in `data/dfire/{train,test}/{images,labels}`; confirm 0 = smoke, 1 = fire — 17,221 train / 4,306 test via HF mirror `badsaarow/d-fire` + `scripts/prepare_dfire.py`
- [x] PyroNear `pyro-sdis` downloaded (optional) — 29,537 train / 4,099 val tower images → `data/pyro_yolo/` via `scripts/prepare_pyro.py`
- [x] 4–6 FIgLib sequences in `data/demo/<name>/` — 5 real ignitions (Rainbow, Beaver, Sorrento, Chia, Brengle), 81 frames each (−40 → +40 min) as `data/demo/fire_*`
- [x] 150–300 benign images (campfire, BBQ, chimney, stack, fog, cloud) in `data/benign/` — 300 real smoke-free lookout-tower images from pyro-sdis (sky/cloud/haze); campfire/BBQ/stack still optional
- [ ] (Optional) ~60 hand-checked crops in `data/gold/gold.jsonl`

## Phase 4 — Detector: before → fine-tune → after 🖥️ [T4, T4b]

- [x] **Before:** `eval_detector.py --weights yolov8s-worldv2.pt --classes smoke,fire --name before_yoloworld` (run while online; caches CLIP)
- [x] `train_detector.py --epochs 1` → note time per epoch → choose N — 2.5 min/epoch, 1 epoch already mAP50 0.085 → N = 50
- [x] `train_detector.py --epochs N` (tmux `train`) → `models/smoke_yolo.pt` — 50 epochs done; final validation mAP50 = 0.78
- [x] **After:** `eval_detector.py --weights models/smoke_yolo.pt --name after_yolo11s` — mAP50 **0.787** (from 0.002), P 0.78, R 0.72, smoke 0.84 / fire 0.73, 3.3 ms/img

- [x] Domain check on lookout-tower smoke (pyro-sdis val, 4,099 images, imgsz 1024): fine-tuned YOLO11s only **mAP50 0.213** (P 0.37, R 0.29) vs 0.787 on D-Fire — small distant plumes are missed (also seen on FIgLib demo frames)
- [x] **Stage-2 tower fine-tune:** from `models/smoke_yolo.pt` on pyro-sdis (29,537 tower images), 960 px, 12 epochs → `models/tower_yolo.pt`
- [x] Evaluate stage 2 on tower val and on D-Fire test — tower smoke mAP50 **0.198 → 0.728**; but D-Fire **0.787 → 0.106** (fire class lost): catastrophic forgetting
- [ ] ⏳ **Stage-3 joint (replay) training:** from stage 1 on D-Fire train + tower train (46,758 images), 960 px, 6 epochs → `models/joint_yolo.pt` (~24 min/epoch, ETA ~05:15)
- [ ] Evaluate stage 3 on both tower val and D-Fire test; pick the deployed detector

## Phase 5 — Context VLM: before → fine-tune → after 🖥️ [T12, T16, T24, T24b]

- [x] Serve the 32B teacher on :8001 (tmux `teacher`) — served as `teacher32b` on the shared zrt proxy (:8000) at 45% memory; base7b paused meanwhile
- [x] `teacher_label.py --n 200` → check progress, several `source_type`s, note seconds per image — 500 crops in 6m50s (~0.8 s/crop, 32 workers); D-Fire smoke/fire crops → wildland/controlled burn/stack, only 6.6% called fog; no-fire crops → mostly fog/cloud ✔
- [x] `teacher_label.py --n 2000` (overnight) → `data/teacher/train.jsonl`, `heldout.jsonl` — 3,100 crops labeled (2,000 D-Fire + 300 benign + 800 pyro-sdis tower smoke)
- [ ] ⏸ (optional, deferred — 32B at ~7.5 s/call ≈ 1 h for 500 crops) **Reference:** `eval_context.py --model <teacher id> --base-url http://localhost:8001/v1 --name ref_teacher32b`
- [x] **Before:** `eval_context.py --model <base id> --name before_base7b` — source-type 45.0%, group 53.0%, 304 tokens/call (never says "unknown": 0/138; controlled burn 1/53)
- [ ] Set `vlm_timeout_s` in `config/settings.json` to ~2× the measured `latency_ms_p95`
- [x] Stop the teacher; `train_lora.py` (tmux `lora`) — done: 326 steps in 1h31m; loss 1.21 → 0.137, answer-token accuracy 70% → 95.1% (`results/lora_training/`); → `adapters/context/adapter_config.json`
- [x] Stop the base `vlm` server; re-serve with `--enable-lora --max-lora-rank 16 --lora-modules context=$HOME/sentinel/adapters/context`
- [x] `/v1/models` lists both the base id and `context` — but the zrt proxy only routes `base7b`; use the backend socket `unix:///opt/hp/zrt/run/vllm-base7b.sock` (fix `ec8358a`). Spot check: LoRA fixes controlled-burn and unknown cases the base model got wrong
- [x] **After:** `eval_context.py --model context --base-url unix:///opt/hp/zrt/run/vllm-base7b.sock --name after_lora7b` — source-type **73.6%** (from 45.0%), group **75.6%** (from 53.0%), 293 tokens/call; unknown 0→123/138, controlled burn 1→34/53, industrial stack 54→38/66 (regression)
- [ ] Baseline: `linear_probe.py`
- [ ] (Optional) Gold set: `--split data/gold/gold.jsonl` for `before_base7b_gold` and `after_lora7b_gold`

## Report artifacts (captured as we go) 📄

- [x] `scripts/collect_artifacts.sh` gathers curves, confusion matrices, sample predictions, logs and label stats into `results/`; synced into the repo
- [x] Detector: `results/detector_{before_yoloworld,after_yolo11s}.json` + `results/detector_training/` (15 plots, `results.csv`)
- [x] Teacher label mix: `results/teacher_labels_summary.json` (2,600 train / 500 held-out)
- [x] Live dashboard snapshots every minute: `results/live/live_metrics.jsonl`
- [x] VLM before/after JSONs + LoRA training log + loss curve; `results/before_after.md` generated
- [ ] End-to-end bench JSONs + `results/before_after.md`

## Phase 6 — End-to-end benchmark and cost 🖥️ [T25, T25b, T26]

- [x] `data/bench/clips.csv` with 20–40 clips (`alert` / `no_alert`) — 25 real tower clips via `scripts/make_bench_clips.py`: 15 alert (5 FIgLib post-ignition + 10 pyro smoke windows), 10 no_alert (smoke-free tower footage)
- [ ] ⏳ `bench.py --name before --detector-weights yolov8s-worldv2.pt --detector-classes smoke,fire --model <base id> --recheck-s 5`
- [ ] `bench.py --name after --detector-weights models/smoke_yolo.pt --model context --recheck-s 5`
- [ ] `bench.py --name base_full --model <base id> --full-frame --recheck-s 5` (token ablation for the cost model)
- [ ] `bench.py --name detector_only --detector-only` (optional ablation)
- [ ] `compare.py` → `results/before_after.md`
- [ ] Fill `config/cost_inputs.json` (measured values + current published prices, with sources) → `cost_model.py`
- [ ] Commit `results/` (JSON + `before_after.md`)

## Phase 7 — Live demo 🖥️ [T23]

- [ ] `config/towers.json` sources point at real `data/demo/...` folders
- [ ] `config/settings.json`: `vlm_model` = `context` (or the base id), `detector_weights` = `models/smoke_yolo.pt`
- [ ] `rm -f data/outbox.db`
- [ ] `./scripts/run_all.sh`; laptop: `ssh -L 8080:localhost:8080 hpX@<nano-ip>` → http://localhost:8080
- [ ] Both feeds show boxes
- [ ] Wildfire tower: "classifying…" → ALERT, held in the outbox while OFFLINE
- [ ] Link ONLINE → sent; `results/dispatch.log` shows `forecast=ok`
- [ ] Benign tower shows LOG/IGNORE and never reaches dispatch
- [ ] Record a backup screen capture of the full demo

## Phase 7b — Demo UI page 💻🖥️

- [ ] Design the demo page: what judges see in 5 minutes (live tower feeds, event timeline, escalation/outbox, online/offline toggle, metrics)
- [ ] Before vs after view: same frame through base vs fine-tuned models, side by side (detector boxes + VLM verdict)
- [ ] Results panel that reads `results/*.json` (detector mAP, VLM accuracy, end-to-end precision/recall, cost model)
- [x] **Live comparative metrics dashboard** — live on the Nano (tmux `monitor`, port 8090): `ssh -N -L 8090:localhost:8090 hp11@100.109.162.35` → http://localhost:8090. Reviewed (202 tests). First reading: teacher32b 839k input / 299k output tokens — while any model runs (detector, base VLM, LoRA VLM, teacher), show live per-model: requests, tokens in/out, tokens/s, latency p50/p95, GPU memory, calls avoided by the cascade, bytes sent upstream, and a running cost comparison (edge vs cloud-per-frame at configurable $/token and $/GB) to summarise the economics
- [ ] Architecture + "why edge" panel (escalation rule, bytes sent vs video, tokens per event)
- [x] **Two-pane demo app** (`python -m sentinel.webapp`, port 8095) — built and reviewed (278 tests; token math verified exact; fair ≤1280 px cloud baseline; honest $0-marginal label): LEFT = pick/upload an image (+ ground truth) and run BEFORE (YOLO-World + base7b) vs AFTER (YOLO11s + LoRA) side by side with boxes, verdict, severity, report, tokens, timings; RIGHT = live usage & economics (tokens actual vs full-frame, calls avoided, bytes, $ edge vs cloud, online vs offline alerts sent/queued, served-model telemetry, benchmark table)
- [x] Deploy the demo app on the Nano after LoRA is served (base7b with `--enable-lora`) — live on :8095 via the backend socket; BEFORE and AFTER both served; first session: AFTER used 191 VLM tokens vs ~4,149 cloud-every-frame estimate (−95%), 0 bytes vs 377 KB uploads
- [ ] Implement, test, and serve it from the Nano dashboard (port 8080, via SSH tunnel)
- [ ] Rehearse the demo flow on the page end to end

## Phase 8 — Deliverables (due Fri 9/25, 8pm) [T28]

- [ ] README "Results" section updated from `results/before_after.md`
- [ ] Comparative study: replace targets with measured numbers; open and verify each source
- [ ] Public GitHub: final push, README renders, no secrets
- [ ] 2-minute video uploaded to YouTube (public), plus the video file
- [ ] Interactive presentation with the 3 required visuals: architecture, benchmarks, impact
- [ ] SJSU Google Drive: project brief, video, links, deck, diagrams, photos
- [ ] Social posts with the required tags
- [ ] Pitch rehearsal: 3-minute pitch + Q&A (escalation rule, before/after numbers, cost model)

---

## Results log

Fill these in as runs finish (they also land in `results/*.json`).

| Run | Key metric | Value | Date |
|---|---|---|---|
| detector `before_yoloworld` | mAP50 | 0.002 (P 0.10, R 0.008, 3.8 ms/img) | 2026-09-23 |
| detector `after_yolo11s` | mAP50 | **0.787** (mAP50-95 0.460, P 0.78, R 0.72, 3.3 ms/img) | 2026-09-23 |
| context `ref_teacher32b` | group accuracy | | |
| context `before_base7b` | group accuracy | 53.0% (source-type 45.0%) | 2026-09-23 |
| context `after_lora7b` | group accuracy | **75.6%** (source-type 73.6%) | 2026-09-24 |
| detector tower smoke (stage 1 → stage 2) | mAP50 | 0.198 → **0.728** | 2026-09-24 |
| detector D-Fire after stage 2 | mAP50 | 0.787 → 0.106 (forgetting) | 2026-09-24 |
| context `linear_probe` | group accuracy | | |
| bench `before` | precision / recall | | |
| bench `after` | precision / recall | | |
| cost model | cascade vs cloud $/day | | |

## Issues / notes

- 2026-09-23: A stuck root `trtllm-serve` (Qwen2-VL, 87 GB) was holding memory; stopped with sudo → 114 GB free.
- 2026-09-23: Nano runs Python 3.12 — fixed a 3.13-only type hint in `sentinel/app.py` (`c8baf53`); 148 tests pass on the Nano.
- 2026-09-23: CUDA PyTorch 2.14 (cu130) installed in `.venv`; transformers 5.17, trl 1.13, peft 0.21, ultralytics 8.4.160.
- 2026-09-23: zrt serves ALL models behind ONE proxy (set to port 8000); models are selected by name. The teacher will NOT be on :8001 — use the default base URL with the teacher's model id.
- 2026-09-23: vLLM at 30% memory failed (no KV cache room while YOLO ran) → use `--gpu-memory-fraction 0.45`.
- 2026-09-23: Runtime `vlm_timeout_s` (15 s) is too low while training shares the GPU — set it from measured p95 after Phase 5.
- 2026-09-23: vLLM per-model metrics (tokens, latency histograms, KV cache) are readable at `/opt/hp/zrt/run/vllm-<label>.sock` → `/metrics`; the live dashboard uses them.
- 2026-09-23: Teacher label mix (500-crop trial): no campfire/BBQ examples in D-Fire or tower data — add some if time allows, or say so in Limitations.
- 2026-09-23 ~20:20–21:39: Nano unreachable over Tailscale (network drop, no reboot); all work intact. Long jobs now run inside tmux.
- 2026-09-23: YOLO-World zero-shot baseline is near zero (mAP50 0.002). Consider more descriptive prompts (e.g. "wildfire smoke plume") so the baseline isn't seen as a strawman.

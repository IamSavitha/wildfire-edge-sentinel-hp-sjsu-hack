# Demo runbook — HP Edge AI SJSU Hack

Version on stage: **`v1.4.0-incident-map`** — models from `v1.3.0-joint` (joint detector at 960 px + LoRA-distilled Qwen2.5-VL-7B) plus Kruthika's incident map, served on the HP ZGX Nano `spark-d07`.

## 1. Before you present (10 minutes)

1. **Link:** `tailscale ping 100.109.162.35` answers.
2. **Start everything on the Nano** (safe to re-run):
   ```bash
   ssh hp11@100.109.162.35 'cd ~/sentinel && ./scripts/start_demo.sh'
   ```
   It must print `before … OK`, `after … OK` and `incident map: HTTP 200`.
3. **Open the tunnel on the laptop** (leave this terminal open):
   ```bash
   ssh -N -L 8095:localhost:8095 -L 8090:localhost:8090 -L 8100:localhost:8100 hp11@100.109.162.35
   ```
4. Open **http://localhost:8095** (demo app), **http://localhost:8090** (live model metrics) and **http://localhost:8100** (incident map).
5. In the demo app, click **Reset session** so the economics start from zero, and enter per-million-token prices only if you have current published rates to cite.
6. In the incident map, open **Demo guide → Load demo scenarios** now: the photos run in seconds, but the two live tower watches (Beaver, Junction) replay for a few minutes. Leave the dispatch link down.
7. **Backup:** have the screen recording of a full run ready in case the link drops.

## 2. The 5-minute story

| Minute | Show | Say |
|---|---|---|
| 0:00 | Title / problem | "Lookout towers sit where the network is weak. Cloud-only AI goes blind when the link drops and pays for every frame. We moved the whole decision onto an HP ZGX Nano at the tower." |
| 0:40 | Demo app, left pane — a **FIgLib tower frame 20 min after ignition** (sample `fb755c1466b0` or `c10714c65583`), both passes | "Same frame, before and after fine-tuning on the device. The off-the-shelf models see nothing. After fine-tuning, the detector finds the plume and the local VLM calls it wildland." |
| 1:30 | A **D-Fire smoke + fire image** (`17a6910bc086`, `125fdbae2090` or `ca734ddea5b5`) | "BEFORE ignores it; AFTER goes straight to ALERT with a dispatch-ready report written on the device." |
| 2:05 | A **no-fire image** (`90ccfd0b38c5` or `2454af4de0a2`) | "No detection, no model call, zero tokens — the cascade only wakes the VLM when it must." |
| 2:25 | Toggle **Offline**, run a fire image again | "Link down: the edge still decides, and the alert waits in the outbox. A cloud-only system would make no decision at all." Toggle **Online**: "It goes out within seconds, about 1 KB, before the forecast." |
| 3:00 | **Incident map** (http://localhost:8100): open the **Junction Fire** incident (triangulated from 2 towers, ALERT) (wind arrow + downwind cone), press **Restore link**, then **Ask Sentinel** "What needs attention now?" | "Same models, a network view. Each tower only knows a direction; where two towers' lines cross is the fire. The wind comes from the tower's own station, so the cone works offline. Restore the link: only the waiting ALERTs go out. And a ranger can ask the on-device model, which answers from this log only." |
| 3:50 | Right pane + http://localhost:8090 | "Tokens used vs a cloud VLM on every frame, bytes sent vs uploading images, live latency per model." |
| 4:15 | Results slide (`results/before_after.md`) | The numbers below. |
| 4:45 | Close | "Trained, served and measured on one ZGX Nano; the cloud only when it adds something the tower can't produce." |

The incident map (triangulation, wind, live watch, Ask Sentinel) was built by Kruthika Virupakshappa.

## 3. The numbers (all measured on the Nano)

| What | Before | After |
|---|---|---|
| Detector, D-Fire test mAP50 | 0.002 (YOLO-World zero-shot) | **0.787** (YOLO11s, stage 1) |
| Detector, lookout-tower smoke mAP50 | 0.198 (stage 1) | **0.718** (joint, deployed) — and D-Fire stays at 0.748 |
| VLM context agreement with the 32B teacher (500 held-out crops) | 45.0% | **73.6%** (LoRA 7B) |
| End-to-end, 25 real tower clips (15 fires) | recall 0/15 | **recall 11/15**, first alert ~5–8 s into the event |
| VLM work | — | **1 call per ~40 frames**, ~220 tokens per call |
| Demo session | — | ~92% fewer VLM tokens than a cloud VLM on every frame; 100% fewer image bytes uploaded |

## 4. Say it accurately

- Incident map, rehearsed on the integrated build (joint detector): all 11 photo scenarios matched their expected severity; Junction triangulated from **2** towers (not 4 as with the older tower-only model); Beaver reached ALERT. With two towers the lines always cross, so don't cite "agreement" as accuracy.

- VLM "accuracy" is **agreement with the 32B teacher** (distillation), not ground truth.
- End-to-end false alarms: **6 of 10** no-fire clips alerted. At least one of those clips contains a real, unlabelled plume (pyro-sdis doesn't label every frame), and one is a genuine low-cloud confusion. The benign set is being hand-checked; don't quote precision as final.
- Cloud-only column: the code is built and fairness-reviewed; **measured cloud numbers need the provider API key**. Until then, cloud costs are estimates from the same images and the provider's token formula.
- Speed is not the headline: on a good link the cloud is about as fast. The edge wins on **bytes and cost, deciding during outages, and slow links**.

## 5. Likely questions (short answers)

- **Why not just the cloud?** No link, no decision; per-frame cost; slow uplinks. The edge decides locally and sends ~1 KB alerts.
- **Why LoRA and a 32B teacher?** No labelled data for "what kind of smoke is this"; the teacher labels on the device, the 7B student serves it cheaply.
- **What went wrong?** Tower domain gap (0.198) → tower fine-tune (0.728) → catastrophic forgetting of D-Fire (0.106) → joint replay training fixed both (0.718 / 0.748).
- **What would you do next?** Hand-checked benign clips, more fog/cloud examples, measured cloud baseline, field trial on a solar tower with the distilled model on a smaller edge device.

## 6. If something breaks

| Symptom | Fix |
|---|---|
| Page won't load | Tunnel dropped: re-run the `ssh -N -L …` command; check `tailscale ping`. |
| AFTER shows "VLM not served" | `ssh hp11@100.109.162.35 'cd ~/sentinel && ./scripts/start_demo.sh'` |
| Everything slow | Another job on the Nano is using the GPU: `ssh hp11@… 'nvidia-smi; tmux ls'` |
| Incident map (8100) won't load or shows no incidents | `ssh hp11@… 'tmux ls; tail -20 ~/sentinel/incidentmap-core.log'` then re-run `start_demo.sh` (it reuses any instance already on 8100). Map blank: `~/sentinel-assets` missing (`scripts/setup_map_assets.sh`, needs internet). Watch "recording not on this device": `scripts/fetch_figlib.sh`. Otherwise skip the segment. |
| Link to the Nano is down | Play the backup recording; talk through `results/before_after.md`. |

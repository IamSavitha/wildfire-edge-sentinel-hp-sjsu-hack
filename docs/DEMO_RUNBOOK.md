# Demo runbook — HP Edge AI SJSU Hack

Version on stage: **`v1.3.0-joint`** (joint detector at 960 px + LoRA-distilled Qwen2.5-VL-7B), served on the HP ZGX Nano `spark-d07`.

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
7. **Live camera + phone** (setup once, rehearse every time; details in §1a): press **Send test alert** on the
   demo app's **Live camera** tab and check that the phone buzzes. Then **Reset live**.
8. **Backup:** have the screen recording of a full run ready in case the link drops.

## 1a. Live camera + phone alerts: setup and rehearsal

**Once, on the phone:** install the **ntfy** app (iOS App Store / Google Play) and subscribe to a *long random*
topic on `ntfy.sh` (anyone who knows the topic can read the alerts), e.g. from `openssl rand -hex 16`.
Allow notifications for ntfy, and make sure no Focus / Do Not Disturb mode silences it during the demo.

**Once, on the Nano** (the topic never goes into the repo, a log or a slide):
```bash
echo 'export NTFY_TOPIC_URL=https://ntfy.sh/<your-long-random-topic>' > ~/.sentinel_alerts.env
chmod 600 ~/.sentinel_alerts.env
tmux kill-session -t webapp 2>/dev/null; ./scripts/start_demo.sh   # a running webapp keeps its old env
```
`start_demo.sh` must print `phone alerts: configured` and, in the health check,
`live camera phone alerts (running app): configured`.

**Camera (laptop):** a Mac with an iPhone nearby shows **Continuity Camera** ("iPhone Camera") in the page's
camera list once the browser has camera permission; any USB/built-in webcam works too. Open the demo app as
**http://localhost:8095** (through the tunnel): browsers only allow the camera on `localhost` or HTTPS, so
`http://<nano-ip>:8095` will not work. On macOS allow the browser in System Settings → Privacy & Security → Camera.

**What to film:** a wildfire video (e.g. a FIgLib tower sequence or a news clip of a smoke plume) full-screen on a
tablet or second monitor, camera ~0.5–1 m away, the plume filling a good part of the frame. **Never a real flame
or smoke indoors.**

**Rehearsal (5 min):**
1. Live camera tab → pick the camera → interval **1.5 s** → **Start**. Boxes appear on the preview.
2. **Send test alert** → phone buzzes (a test push; at most one push per 30 s, else "Suppressed (rate limit)").
3. Play the smoke video: gate dots fill 1/3 → 3/3, the VLM verdict appears once, then ALERT (black/large/near
   structures) or MONITOR (re-checked every 10 s; growing → ALERT). Banner: **Sent to phone ✓**; phone buzzes
   with the snapshot and the dispatch text.
4. **Reset live**, toggle **Offline**, replay → banner **Queued — link offline**; toggle **Online** → the banner
   turns **Sent to phone ✓** within seconds and the phone buzzes. Wait 30 s between pushes (rate limit).
5. **Reset live** before the real demo (clears the open event and the one-ALERT-per-fire latch).

## 2. The 5-minute story

| Minute | Show | Say |
|---|---|---|
| 0:00 | Title / problem | "Lookout towers sit where the network is weak. Cloud-only AI goes blind when the link drops and pays for every frame. We moved the whole decision onto an HP ZGX Nano at the tower." |
| 0:40 | Demo app, left pane — a **FIgLib tower frame 20 min after ignition** (sample `fb755c1466b0` or `c10714c65583`), both passes | "Same frame, before and after fine-tuning on the device. The off-the-shelf models see nothing. After fine-tuning, the detector finds the plume and the local VLM calls it wildland." |
| 1:30 | A **D-Fire smoke + fire image** (`17a6910bc086`, `125fdbae2090` or `ca734ddea5b5`) | "BEFORE ignores it; AFTER goes straight to ALERT with a dispatch-ready report written on the device." |
| 2:05 | A **no-fire image** (`90ccfd0b38c5` or `2454af4de0a2`) | "No detection, no model call, zero tokens — the cascade only wakes the VLM when it must." |
| 2:25 | **Live camera** tab (45 s): smoke video on the tablet in front of the camera → boxes on the preview → gate "smoke seen in 3/3 frames" → VLM verdict → ALERT → **phone buzzes** (hold it up). **Reset live**, toggle **Offline**, smoke again → "Queued — link offline"; toggle **Online** → phone buzzes again | "A real camera, live, on the Nano. Smoke must persist for three frames before we spend one VLM call; one alert per fire, not one per frame. Link down: the edge still decides and the alert waits in the outbox — a cloud-only system would make no decision at all. Link back: it goes out within seconds, about 1 KB plus one snapshot." |
| 3:10 | **Incident map** (http://localhost:8100): open the triangulated **Junction Fire** incident (wind arrow + downwind cone), press **Restore link**, then **Ask Sentinel** "What needs attention now?" | "Same models, a network view. Each tower only knows a direction; where two towers' lines cross is the fire. The wind comes from the tower's own station, so the cone works offline. Restore the link: only the waiting ALERTs go out. And a ranger can ask the on-device model, which answers from this log only." |
| 3:55 | Right pane + http://localhost:8090 | "Tokens used vs a cloud VLM on every frame, bytes sent vs uploading images, live latency per model." |
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
| Live camera: "not a secure origin" / Start disabled | The page was opened by IP. Use **http://localhost:8095** through the tunnel. |
| Live camera: permission denied / no camera listed | Browser site settings → allow Camera; macOS System Settings → Privacy & Security → Camera → enable the browser, reload. iPhone missing: unlock it, same Apple ID, Wi-Fi + Bluetooth on, bring it close; or pick the built-in webcam. "In use by another app": quit FaceTime/Zoom/Photo Booth. |
| Live camera: boxes but never reaches 3/3 | Plume too small or flickering: move closer, full-screen the video, pause on a dense-smoke frame. Still nothing: use the single-image tab with a tower sample. |
| Live camera: "Frame rejected: live detector weights not found" | The app was started with other weights: re-run `start_demo.sh` after `tmux kill-session -t webapp`. |
| Live camera: ALERT already raised, nothing new | One ALERT per fire (latch): press **Reset live**. |
| Phone does not buzz; banner says "Delivered (simulated…)" | The app has no topic: check `start_demo.sh` prints `configured`; if the webapp was already running, `tmux kill-session -t webapp` and re-run. |
| Banner "Delivery failed — retrying" | ntfy unreachable from the Nano (`curl -sI https://ntfy.sh` on the Nano). The alert stays in the outbox and goes out when it answers; keep talking, it retries every few seconds. |
| Banner "Suppressed (rate limit)" | Less than 30 s since the last push (a test push counts). Wait, **Reset live**, repeat. |
| Phone gets pushes but silently | ntfy app: check the topic's notification sound/priority settings; turn off Focus / Do Not Disturb. |

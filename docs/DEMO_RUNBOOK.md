# Demo runbook — HP Edge AI SJSU Hack

Version on stage: **`v2.2.0-submission`**. It is one server, the Sentinel Console on port 8080, with the forest-officer console as its home page. The models are unchanged from `v1.3.0-joint`: joint YOLO11s at 960 px, plus Qwen2.5-VL-7B with the `context` LoRA. Everything runs on the HP ZGX Nano. The runbook for the earlier three-port demo is kept in the `v1.5.0-live-camera` tag.

## 1. Before you present (30 minutes)

1. **Start the console on the Nano** (safe to re-run; it also serves the VLM if it is not up):
   ```bash
   ssh hp11@<nano-ip>
   cd ~/sentinel-console            # or ~/sentinel on main
   tmux kill-session -t console     # only if one is running
   RESULTS=~/sentinel/results ./scripts/start_console.sh
   ```
   The health check must show `Detector ok`, `Context VLM ok` and **`Phone alerts ok`**. If phone alerts are `off`, `~/.sentinel_alerts.env` (`NTFY_TOPIC_URL=...`, chmod 600) is missing.
2. **Open the tunnel on the laptop** and leave the terminal open:
   ```bash
   ssh -N -L 8080:localhost:8080 hp11@<nano-ip>
   ```
   Then open **http://localhost:8080**.
3. **System → Send test push.** The team phone, subscribed to the topic in the ntfy app, should buzz.
4. **Edge vs Cloud → Prices:** enter the cloud provider's current published rates, so the $ figures show.
5. **Mobile camera:** allow the camera. Pick the iPhone (Continuity Camera) or the webcam, and have a wildfire video ready on a tablet. Flames with a large plume go straight to ALERT; a small static plume stays MONITOR.
6. The **link must be up** (header: *Dispatch link up*). Do one full rehearsal of section 2.

## 2. The demo (about 2 min 45 s)

| Step | Do | Say |
|---|---|---|
| 1 | **Edge vs Cloud → Send a fire frame** | "Both pipelines finish. The edge decided in about 5.4 s and the frame never left the device; cloud-only had to upload it first." |
| 2 | **Header → Simulate outage**, then **Send a fire frame** | Pause. "The edge goes fully green and queues the ALERT. Cloud-only stops at the network line: blind during the outage." |
| 3 | **Mobile camera → Start camera** on the tablet video | "Same models on a patrol unit's camera: smoke 1, 2, 3 frames, the VLM classifies it, ALERT, still with no network." |
| 4 | **Header → Restore link** | "Queued, restored, sent." Hold up the team phone as the ntfy notification arrives. |
| 5 | **Incidents**, then **Ask Sentinel → What needs attention now?** | "The assistant also runs on the device: zero bytes to the cloud." |

Shortcut: **http://localhost:8080/#tab=camera&start** opens straight into the live camera.

## 3. If something goes wrong

| Symptom | Fix |
|---|---|
| Page does not load | The tunnel dropped: rerun the `ssh -N -L 8080…` command and refresh. |
| Camera blocked | The console must be on `localhost` (secure origin); allow the camera in the browser and in macOS Privacy settings. |
| iPhone not in the camera list | Same Apple ID, Wi-Fi and Bluetooth on, Continuity Camera on in iPhone Settings; check it shows in Photo Booth, then reload. |
| No phone buzz | System tab: Phone alerts must be *ok*; Send test push; check the phone is subscribed to the same topic. |
| Anything else | Keep talking and use **Edge vs Cloud → Replay**, or the screenshots in `results/console/` (`race-*.png`, `officer-*.png`). |

## 4. After the demo

Leave the console running, or stop it with `tmux kill-session -t console`. Incidents and the alert outbox persist in `~/sentinel-state/console`. The Docker alternative is `sudo docker compose up -d --build` from the repository root; see the README.

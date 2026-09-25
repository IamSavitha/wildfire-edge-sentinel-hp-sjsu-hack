# Sentinel Console (v2.0.0): design

**Date:** 2026-09-24 · **Branch:** `feat/console` · **Tag when done:** `v2.0.0-console`

## Problem

The system runs as four separate servers, each with its own screen and its own copy of the models:

| Port | Screen | Owns |
|---|---|---|
| 8095 | Demo app (`sentinel/webapp.py`) | image upload, live camera, before/after, economics panel |
| 8090 | Live metrics (`sentinel/monitor.py`) | vLLM tokens, cost, GPU |
| 8100 | Officer console (`sentinel/incident_map.py`, Kruthika) | incident map, tower watch, Ask Sentinel, demo guide |
| n/a | Old dashboard (`sentinel/app.py`) | tower replay |

Their data doesn't connect. A fire seen by the live camera never becomes an incident on the map. The map only simulates escalation, while the live camera pushes to a real phone. Each server has its own online/offline toggle. Two of them define the same `/api/config`, `/api/samples` and `/api/network`. It reads as a demo, not as a system.

## Goal

One product: the **Sentinel Console**. It is one process on one port (8080), with one screen and one set of shared services. It looks professional and minimalist, shows details on demand, and uses no "demo" wording. The console must also be the place where **edge vs cloud-only** is shown, since that is the hackathon's main argument.

## Architecture

```
                         ┌──────────────── sentinel/console.py (port 8080) ────────────────┐
 tower recordings ──┐    │  Services (built once)                                           │
 mobile camera   ───┼──▶ │   detector cache + GPU lock · VLM client · served-model list    │
 field photos    ───┘    │   Link (one uplink) · Delivery (outbox → ntfy / dispatch)        │
                         │   IncidentStore · CloudShadow ledger · metrics monitor           │
                         │                                                                  │
                         │  /ops/*   incident map app (create_map_app, shared services)     │
                         │  /lab/*   web app (create_web_app: compare models, live camera)  │
                         │  /api/*   console: uplink, alerts, edge-vs-cloud, models, system │
                         │  /        console UI (static/console/)  · /assets, /tiles        │
                         └──────────────────────────────────────────────────────────────────┘
```

- **Services (`sentinel/services.py`).** These are built once and passed into the existing app factories as optional arguments. When no services are passed, the factories behave exactly as they do today, so v1.x entry points and tags are untouched.
- **Mounting.** The map app is mounted at `/ops` and the web app at `/lab`. Their own routes keep working as `/ops/api/state`, `/lab/api/analyze` and so on, so route names can't collide. Starlette does not run a mounted app's lifespan, so the console's lifespan enters each sub-app's lifespan context itself.
- **One uplink.** A single `Link` is shared. The map's `/api/network`, the web app's `/api/network` and the console's `/api/uplink` all flip it, and bringing it back up flushes the one outbox.
- **One delivery path.** When the map escalates an ALERT, it goes through the shared `Delivery` (durable outbox, then ntfy push and optional dispatch POST) instead of the simulated `escalate`. The same happens for tower, mobile-camera and field-report ALERTs. One fire gives one incident, one outbox entry and one phone push (the per-fire latch and the incident match stop duplicates).
- **Mobile camera → incident.** When a `LiveCamera` event reaches MONITOR or ALERT, the map records an observation for a registered camera `mobile-1`. Its location comes from settings, or from browser geolocation when allowed. The event then shows up on the map, in the incident list and in the alert log.

## Offline-first

The system has to work with **no network and no internet at all**. Only the upstream alert waits for the link.

| Needs | Without internet |
|---|---|
| Detection, VLM context, severity, incidents, Ask Sentinel | Run on the Nano; unaffected |
| Console UI | Served by the Nano. No CDN, web fonts or remote scripts: system font stack, inline SVG icons and charts, MapLibre/PMTiles from local `/assets`. A test fails if any console file references an external URL |
| Map | Local PMTiles basemap |
| Wind / forecast | On-site sensor, or the last cached forecast, labelled "cached"; never blocks a decision |
| Alert to dispatch / phone (ntfy.sh) | Held in the durable outbox and sent when the link returns (the existing Delivery backoff) |
| Alert to the on-site operator | **Immediate and local:** the console raises a banner, a sound and a browser notification on every new ALERT, whatever the uplink state |
| Access | `--host 0.0.0.0` makes the console reachable on the local network (tower LAN / Wi-Fi) with no tunnel. The mobile camera needs a secure context, so it runs on `localhost` (on the Nano, or through an SSH tunnel over the LAN) |

The header's uplink pill shows **Online** or **Offline, N alerts queued**. No page shows an error state just because the internet is gone.

## Edge vs Cloud

`sentinel/shadow.py`, `CloudShadow`: for every frame the edge node really processes, it records what a cloud-only design would have done with the same frame.

| Quantity | Edge | Cloud-only |
|---|---|---|
| Frames / bytes uplinked | ALERT payloads only (measured) | every frame, whole, ≤1280 px JPEG (modeled from the real frame size) |
| Tokens / VLM calls | measured per event | `cloud_frame_tokens(w, h)` + prompt, per frame |
| $ | 0 marginal | tokens × price + GB × price (prices are user inputs, labelled as assumptions) |
| Time to decision | measured detect + VLM | model inference (edge VLM ms as proxy, or measured) + `netprofile.transfer_s` for the chosen link |
| During outage | decides; ALERT queued | blind: the frame counts as undecided |

- **Link profiles:** fiber, LTE, rural cell, GEO satellite and down, taken from `netprofile.py`. They only change the modeled cloud latency, never the real network.
- **Measured mode:** this is optional and off unless `~/.cloud_vlm.env` exists. It sends a sample (each incident's opening frame, capped per hour) to the hosted model through `CachedVLM`, and shows agreement, real latency, tokens and cost. Without a key every cloud number is labelled *modeled*.
- **Benchmark card:** reads `results/bench_after*.json`, `bench_cloud_*.json` and `outage_*.json` when they exist (`webapp_logic.load_benchmarks`).
- **Fleet projection:** towers × fps × days fed to `scripts/cost_model.compare`.

## Console UI

A single page with the sidebar **Operations · Cameras · Alerts · Edge vs Cloud · Models · System**, plus an **Ask Sentinel** side panel that can be opened from anywhere.

| Page | Visible at a glance | On demand (drawer / expand / hover) |
|---|---|---|
| Operations | full-bleed map, incident list (severity dot, name, age), "+ Field report" | incident drawer: frames, timeline, report text, wind, bearings |
| Cameras | tower tiles with live thumbnail and status; Mobile camera tile; **Run drill** | per-camera detections and gate state |
| Alerts | delivery log: time, incident, channel, state (sent / queued / expired) | payload, retries, recipient (topic redacted); send test |
| Edge vs Cloud | 4 big numbers (cost, bytes, time to decision, frames decided offline), link-profile control, one chart | method and assumptions, per-source breakdown, benchmark, fleet projection, measured samples |
| Models | deployed detector + VLM cards with headline score | eval tables, version history, compare on an image (before/after) |
| System | health dots (detector, VLM, outbox, uplink), GPU/memory sparkline, **Uplink** with *Test outage* | process details, served models, logs path |

**Header, always visible:** node name · uplink state · active incidents · last alert · *vs cloud-only today: $ saved · MB not uplinked*.

**Visual rules (minimalist, professional):**
- **Tokens and colors:** a dark neutral base with a light theme too. Colors are design tokens on `:root`, with ember for fire and severity, blue for edge, and grey for cloud. There's one accent per meaning.
- **Typography:** the system font stack (no web fonts, so it works offline), with a clear hierarchy that puts big numbers before labels and labels before text.
- **Text:** a page shows at most one line of helper text. Everything else goes behind an "i" tooltip or a Details drawer. There are no emoji.
- **Content:** status is shown with dots and pills rather than sentences. Tables are collapsed by default. The same card component is used everywhere.
- **Motion:** subtle count-up on numbers and a pulse on new ALERTs.
- **Libraries:** vanilla JS modules, no build step, nothing loaded from the internet. MapLibre and PMTiles come from the existing `/assets/lib`, and charts are inline SVG.

## Demo features folded into real ones

- **Network toggle:** becomes System, then Uplink, then *Test outage*.
- **Before/after:** moves to Models, then *Compare on an image*.
- **Sample images:** become *+ Field report*, a photo from a ranger or a citizen.
- **Demo guide:** becomes Cameras, then *Run drill*, which replays a recorded fire through the towers.

## Models deployed

- **Detector:** v1.3.0-joint at 960 px, with the D-Fire model for close-up field photos.
- **VLM:** Qwen2.5-VL-7B with the `context` LoRA (v1). LoRA v2 (`context_v2`, +0.8 pt source-type and +1.4 pt group on 500 held-out crops) can be picked on the Models page, and is only promoted if the end-to-end benchmark agrees.
- **Not included:** YOLO11m (0.658 vs 0.730 mAP50) and the unreviewed detector-boosts branch.

## Testing

- **Unit tests:** `Services` (one detector load, shared lock), `CloudShadow` arithmetic (matches `cloud_frame_tokens` and `netprofile`), and measured mode (no key means no call; the hourly cap holds).
- **Integration (TestClient, with fake detector, VLM and notifier):**
  - A mobile-camera ALERT gives one incident, one outbox entry and one push.
  - A tower ALERT goes through the same outbox.
  - A test outage queues alerts from every source, and restoring the uplink flushes them once.
  - Route prefixes don't collide.
  - Sub-app lifespans run.
- **UI:** `node --check` on every JS module, and a test that every `/api/...` path the UI fetches exists in the console app.
- **Regression:** all existing tests still pass.
- **Nano:** smoke test through `ssh -N -L 8080:localhost:8080`.

## Risks

- **UI size:** the UI is the largest piece. If time runs short, Operations, Cameras, Alerts, Edge vs Cloud and System ship first, and Models reuses the existing compare panel.
- **Map assets:** they live in `~/sentinel-assets` on the Nano (Kruthika's). The console only reads them and never writes into her folders.

# Sentinel Console (v2.0.0): implementation plan

Design: [2026-09-24-sentinel-console-design.md](2026-09-24-sentinel-console-design.md) · Branch `feat/console` · Tag `v2.0.0-console`

**Rules for every task**
- **TDD:** write the test first.
- **Run the tests:** `python -m pytest -q` must pass; there are 530+ tests on this branch today.
- **Commits:** one commit per task, a Conventional Commit message, and **no attribution or co-author lines**.
- **Existing entry points** (`python -m sentinel.webapp`, `sentinel.incident_map`, `sentinel.monitor`, `scripts/start_demo.sh`) must behave exactly as before whenever no `services` is passed.
- **Secrets:** never log the ntfy topic or cloud keys.
- **Dependencies:** no new Python dependencies. The UI uses vanilla ES modules with no build step.

---

## Backend

### Task 1: `sentinel/services.py`: shared services
The `Services` dataclass is built once by the console:
- **Detectors:**
  - `gpu_lock: threading.Lock` is one lock for every model call.
  - `detector(weights: str, imgsz: int | None = None)` loads each weights file once (a cache keyed by path plus mtime) through an injectable `detector_factory(weights, imgsz)`. The default is `YoloDetector(weights, conf=0.1, imgsz=imgsz or imgsz_for(weights))`. Loading is guarded by a cache lock. Callers hold `gpu_lock` while running a model.
- **VLM:**
  - `vlm(model: str)` caches one client per model through an injectable `vlm_factory`. The default is `ContextVLM(model, vlm_base_url, timeout_s)`, which supports `unix://`.
  - `served_models() -> tuple[list[str], str | None]` keeps a 10 s TTL cache over an injectable `model_lister` (`list_served_models`).
- **Uplink and alerts:**
  - `link: Link` is the one uplink.
  - `delivery: Delivery` wraps the one durable outbox (`Outbox(outbox_path)`), `notifier` (`NtfyNotifier.from_env()`) and `dispatch_fn`.
  - `set_online(online: bool, now) -> int` flips the link and, on a down→up change, flushes delivery. It returns the number sent and calls every callback registered with `on_link_change(cb)`, which the map app uses to refresh its incidents' escalation state.
- **Shared state:**
  - `shadow: CloudShadow` (Task 2).
  - `clock`.
  - `started_at`.

**Tests:**
- One load per weights file even with 8 concurrent callers.
- `vlm()` is cached.
- The served-models TTL holds.
- `set_online(False)` then `set_online(True)` flushes a queued ALERT exactly once and calls the callbacks.
- Fakes for everything (no GPU, no network).

### Task 2: `sentinel/shadow.py`: `CloudShadow` ledger
This is thread-safe and has no I/O.
- **`record(source: str, width: int, height: int, frame_bytes: int, *, edge_vlm_called: bool, edge_tokens: int, edge_bytes_up: int, edge_decide_ms: float | None, online: bool, vlm_ms: float | None = None)`** records one frame:
  - `source` is one of `tower`, `mobile` or `field`.
  - Cloud-only for this frame uploads a whole JPEG at ≤1280 px. Its bytes are estimated as `frame_bytes` scaled by `(min(1, 1280/max(w,h)))**2`.
  - Its input tokens are `webapp_logic.cloud_frame_tokens(w, h)` plus the prompt tokens (constant `CLOUD_PROMPT_TOKENS`). Look up the real prompt size used in `scripts/bench_cloud.py`, or fall back to 220 with a comment.
  - Its output tokens use the constant `CLOUD_OUTPUT_TOKENS = 80`.
  - When `online` is False, the frame counts as **cloud-blind** (no decision is possible), while the edge still decided.
- **`summary(prices: dict, profile: str) -> dict`** returns:
  - totals per side: frames, bytes up, VLM calls, tokens in/out, $, and the p50 time to decision.
  - `cloud_blind_frames` and `edge_offline_decisions`.
  - `savings` (the $, bytes, % and × figures).
  - `by_source`.
  - `method` (short strings that explain each number, shown on demand in the UI).
  - `modeled: True`.

  Cloud time to decision is `vlm_ms_p50` (the edge VLM measured, used as the model-compute proxy, since it is the same Qwen2.5-VL-7B) plus `netprofile.transfer_s(bytes, get_profile(profile), CLOUD_REQUEST_RTTS)`. Edge time to decision is the measured `edge_decide_ms`.
- **Prices:** these come from `config/cost_inputs.json` through `live_metrics.effective_prices`. When every price is 0, return `prices_set: False` and let the UI show "set prices" rather than $0.
- **`profiles()`** returns the netprofile names with their rtt, uplink and loss values.
- **`fleet(towers: int, fps: float, days: int, prices) -> dict`** calls `scripts/cost_model.compare`. Move `compare` into `sentinel/costmodel.py` and keep a re-export in `scripts/cost_model.py` so its tests still pass. It uses the measured averages from this ledger (bytes per frame, tokens per frame, VLM calls per event) when there is data, and the defaults otherwise.
- `reset()`.

**Tests:**
- Arithmetic matches `cloud_frame_tokens` and `netprofile.transfer_s`.
- Offline frames count as blind.
- Prices set to 0 give `prices_set` False.
- The satellite profile gives a larger cloud time to decision than fiber.
- `fleet` matches `compare`.
- Concurrent `record` calls are safe.

### Task 3: map app on shared services
Add `services: Services | None = None` to `create_map_app`. When it is given:
- **Models:** detectors and the VLM come from `services` (the same `gpu_lock` and caches). `served_models` comes from `services`.
- **Uplink:** online/offline is `services.link.online`. The map's `state["online"]` is no longer used, and `/api/network` calls `services.set_online`.
- **Escalation:** `do_escalate` pushes the ALERT through `services.delivery`. Make sure the report has `event_id` (the incident id or the report id) and `detected_at` (ISO); `Delivery._send` relies on both. Then `delivery.handle(report, Severity.ALERT, now)`, then flush if online.
  - Return the same dict shape as today: `{"decision": "sent" | "queued", "bytes_up", "at", "note", "event_id"}`, where `bytes_up` is the JSON size of the payload if it was sent.
  - `flush_outbox` / `on_link_change` update the incidents whose escalation is `queued` from `delivery.event(event_id)["state"]` (sent / expired) and `bump` the stats.
- **Shadow:** every processed frame (watch step, report upload) calls `services.shadow.record(...)` with source `tower` or `field`.
- **External sightings:** new `app.state.ingest_external(obs: dict, now: float) -> dict`. It records an observation from another source (the mobile camera, Task 5) **without escalating**, because the caller already delivered it. It builds the observation with the existing `observation()` / `store.record()` helpers for a camera entry passed in `obs`, and returns the incident.
- **App state:** expose `app.state.stats`, `app.state.cameras` and `app.state.start_watch` (the function behind `POST /api/watch`, used by Run drill).
- **Watch step lock:** `watch_step` must not hold `inc_lock` while waiting on the shared `gpu_lock` longer than it does today, so keep the existing lock order (inc_lock, then gpu_lock).

**Tests** (new `tests/test_incident_map_services.py`, using fakes and TestClient):
- A tower ALERT while offline gives one outbox entry. After `set_online(True)`, the notifier is called once and the incident escalation reads `sent`.
- A report upload records one shadow frame with source `field`.
- `ingest_external` creates an incident without an outbox entry.
- `test_incident_map.py` still passes untouched.

### Task 4: web app on shared services
Add `services: Services | None = None` to `create_web_app`. When it is given:
- **Shared models:** detector loads go through `services.detector(spec["weights"], spec.get("imgsz"))`, and the VLM through `services.vlm`. The web app uses `services.gpu_lock`, `services.link`, `services.delivery` and `services.served_models`, and does not start its own retry loop (the console runs one, Task 5).
- **Shadow:**
  - `/api/analyze` records every image to `services.shadow` with source `field`, but only for the `after` pipeline so compare runs aren't double counted.
  - `/api/live/frame` records each frame with source `mobile`.
- **Event hook:** new `on_live_event: Callable[[dict, np.ndarray], None] | None` (a `create_web_app` argument). `LiveCamera` calls it for each new MONITOR or ALERT event, passing the event view and the frame, after the pipeline returns and outside its lock. It is used by the console to open an incident.

**Tests** (`tests/test_webapp_services.py`):
- A live ALERT calls `on_live_event` once and uses the shared outbox.
- `/api/network` on the web app flips the shared link.
- `/api/analyze` records shadow frames for `after` only.
- The existing webapp tests still pass.

### Task 5: `sentinel/console.py`: the one server
`create_console_app(*, services=None, cameras=None, assets_dir, data_dir, state_dir, results_dir="results", prices_path="config/cost_inputs.json", mobile_camera: dict | None = None, map_kwargs=None, web_kwargs=None, monitor=None, cloud_sampler=None, clock=time.time) -> FastAPI`.

**Composition:**
- Build `Services`, `map_app = create_map_app(..., services=services)` and `web_app = create_web_app(..., services=services, on_live_event=...)`.
- Mount `/ops` → map_app and `/lab` → web_app.
- Mount the static `/console` files, plus `/assets` and `/tiles` exactly as incident_map serves them today.
- `GET /` returns `static/console/index.html`.

**Lifespan:**
- Enter both sub-apps' `router.lifespan_context(app)`, because Starlette does not run them for mounts.
- Start `monitor`.
- Run one outbox retry thread (`services.delivery.flush` every `FLUSH_EVERY_S` while online and pending).

**Mobile camera → incident:** `on_live_event(ev, frame)` calls `map_app.state.ingest_external(...)` for camera `mobile-1`, which has:
- a name ("Mobile unit 1")
- lat and lon, from `mobile_camera` or `settings.json`, with the web app's `live_lat`/`live_lon` defaults
- azimuth 0 and hfov 60

One live event maps to one incident: key it by the event id, and later events for the same fire update that incident (`find_match` already merges by place and time).

**Console API** (all JSON; small payloads; details behind `?full=1`):
- `GET /api/overview` returns the header data: node name, uplink state, active incidents, the last alert (time, incident, state) and a savings chip (`shadow.summary`).
- `GET|POST /api/uplink` `{online}` calls `services.set_online`.
- `GET /api/alerts` returns the delivery log: merge `delivery.summary()["events"]` with incident names. `POST /api/alerts/test` calls `notifier.send_test()`, with its existing 30 s rate limit. The topic is always redacted (`notify.redact_topic_url`).
- `GET /api/edge-cloud?profile=lte` returns `shadow.summary`, `profiles`, and `benchmarks` (`webapp_logic.load_benchmarks(results_dir)`), plus `measured` (Task 6). `POST /api/edge-cloud/prices` saves prices the same way `webapp` does. `GET /api/edge-cloud/fleet?towers=&fps=&days=` returns the fleet projection.
- `GET /api/models` returns:
  - the deployed detector and VLM (weights path, version tag from `docs/versions/`, headline scores read from `results/detector_*.json` and `results/context_after_lora7b*.json`)
  - the served VLM models
  - the active VLM model, switchable with `POST /api/models/vlm {model}` to any served model; this changes both the map's and the web app's model by setting a shared `services.active_vlm`
- `GET /api/system` returns health for detector, vlm, outbox, uplink, notifier and dispatch (each `ok`/`warn`/`down` plus one short line), `live_metrics.system_stats()` (GPU/memory), uptime, and served models.

`main()`: `--port 8080`, `--host 127.0.0.1`, and the same model, asset and state flags that `start_demo.sh` passes to the two apps today.

**Tests** (`tests/test_console.py`, with fakes):
- `/`, `/ops/api/state`, `/lab/api/config` and `/api/overview` all answer.
- The sub-app lifespans ran.
- A mobile-camera ALERT gives one incident plus one outbox entry plus one push.
- A tower ALERT and a mobile ALERT while offline both queue; `POST /api/uplink {online:true}` sends each once.
- Switching the VLM model applies to both apps.
- `/api/alerts` never contains the topic URL.

### Task 6: measured cloud samples (optional)
`sentinel/cloud_sampler.py` `CloudSampler(vlm, cap_per_hour=20, clock)`:
- `maybe_sample(incident_id, jpeg, edge_ctx, edge_ms)` runs in a background thread and calls the hosted model at most once per incident, within the hourly cap.
- It is built only when the `CLOUD_VLM_*` env vars are set: `cloud_vlm_from_env` wrapped in `CachedVLM`, loading `~/.cloud_vlm.env` if present.
- It stores `{incident_id, cloud_ms, cloud_tokens, cloud_source_type, edge_source_type, agree, cached}`.
- `summary()` returns n, agreement %, p50 cloud ms and tokens, and `enabled`.
- The map app calls it on incident open (a hook passed through `services.on_incident_opened`).

**Tests:**
- No env means no sampler and `measured.enabled` is False.
- The cap is enforced.
- Once per incident.
- A cached hit is flagged.

---

## Console UI (`sentinel/static/console/`)

**Look:** professional and minimalist. Details appear only when asked for.

**Design tokens** in `console.css` `:root`, with a light theme under `[data-theme=light]`:
- **Base:** dark neutral (`--bg #0b0d10`, `--panel #12151a`, `--line #1f242c`, `--text #e6e8eb`, `--muted #8a93a0`).
- **Severity:** `--alert #ff5a3c`, `--monitor #f5a524`, `--log #6b7684`.
- **Sides:** `--edge #3b82f6`, `--cloud #9aa4b2`, `--ok #22c55e`.
- **Shape:** radius 10px, 8-pt spacing grid.
- **Font:** Inter from Google Fonts, falling back to the system font. Numbers use `font-variant-numeric: tabular-nums`.

**Components** (`ui.js`): `card`, `kpi` (big number, small label, optional delta), `pill`/`dot` (status), `drawer` (right slide-in for details), `info` ("i" tooltip), `segmented` control, `sparkline` (inline SVG), `table` (collapsed by default, "Show all"), `toast`, and `countUp`.

**Rules:**
- ≤ 4 KPIs per page.
- At most one helper line per page.
- No emoji; inline SVG icons (Lucide paths).
- Every secondary fact sits behind a drawer, an "i" or an expand.
- Keyboard focus is visible.
- Works at 1280 px and down to 390 px width.

### Task 7: shell
- **`index.html`:** sidebar (logo mark "Sentinel", nav with icons) and a header bar. The header shows the node name, an uplink pill, active incidents, the last alert, and a savings chip that opens a drawer. It also holds a content outlet and an Ask Sentinel toggle, which opens a right-hand panel.
- **`app.js`:** a hash router (`#/operations` etc., Operations by default) that loads page modules (`pages/*.js`, each exporting `mount(el, api)` / `unmount()`).
- **`api.js`:** a small fetch wrapper with polling helpers that pause when the tab is hidden, and one shared `/api/overview` poll every 2 s.
- **Ask Sentinel:** the panel posts to `/ops/api/ask` and shows answers with a minimal chat look.
- **Tests:** a Python test that runs `node --check` on every JS file (skipped if node is missing), and a contract test that extracts every `'/api/...'`, `'/ops/api/...'` and `'/lab/api/...'` literal from the JS and asserts the console app has a matching route.

### Task 8: Operations page
- **Map:** full-height MapLibre map with the PMTiles base, reusing the setup code in `static/incident_map.html` (`/assets/lib/*`, `/tiles/sierra-pacific.pmtiles`). It shows cameras as small markers and incidents as severity-colored pulsing dots, with the spread cone and bearings drawn only for the selected incident.
- **Incident list:** a narrow left overlay list of incidents (dot, name, age, source type). Clicking one opens the drawer with the thumbnail and frames strip, the severity/trend timeline, the latest update, the report text (collapsed), wind, location source and escalation state, plus status actions (PATCH).
- **"+ Field report":** opens a small modal to upload a photo with optional lat/lon, and posts to `/ops/api/report`.
- **Empty state:** "All clear" with the number of cameras watching.

### Task 9: Cameras page
- **Tiles:** a grid of camera tiles (thumbnail from `/ops/api/watch/{sid}/frame/{camera_id}` when a watch is running, otherwise the last frame; name; status dot).
- **Mobile camera tile:** opens a camera view (`getUserMedia`, device picker, start/stop). It posts frames to `/lab/api/live/frame` exactly as `static/webapp.html` does (port that logic: 1 frame per 1–2 s, one in flight, ≤1280 px JPEG). The overlay shows boxes, a gate progress ring and the event severity; the details drawer shows timings, tokens and delivery.
- **Run drill:** a button and a menu of recordings from `/ops/api/recordings` that starts `/ops/api/watch`, with a Stop button.

### Task 10: Alerts and System pages
- **Alerts:** a list of deliveries (time, incident, severity pill, channel icons phone/dispatch, state pill sent/queued/retrying/expired). A row opens the payload and attempts in the drawer. There's a "Send test alert" button, and the recipient is shown redacted.
- **System:**
  - health grid (dot and one word each)
  - GPU/memory sparklines
  - served models
  - **Uplink** card with a large switch "Test outage" and the explainer "Edge keeps deciding; alerts queue and send when the link returns" behind an "i"
  - process details in a drawer

### Task 11: Edge vs Cloud page
- **Top:** a segmented link-profile control (Fiber · LTE · Rural cell · Satellite · Down).
- **4 KPIs:** cost saved (or "set prices"), data not uplinked (MB and ×), time to decision edge vs cloud, and frames decided while offline. Each has an "i" that shows its method line from `shadow.summary().method`.
- **Chart:** one hero chart, horizontal bars for Edge vs Cloud-only across bytes, tokens, $ and time, with log scale where needed. Chart.js from cdnjs is acceptable, or inline SVG.
- **Below, collapsed:**
  - "Outage timeline", the latest incident's detect → queued → delivered markers vs cloud-only blind
  - "Benchmark", from results
  - "Fleet projection", sliders for towers, fps and days that call `/api/edge-cloud/fleet`
  - "Measured cloud samples", shown only when enabled
  - "Assumptions & prices", a form that posts to `/api/edge-cloud/prices`
- **Labels:** a *Modeled* or *Measured* tag always sits next to the cloud numbers.

### Task 12: Models page
- **Cards:** two cards (Detector, Context VLM) with name, version tag and one headline score. A "Details" drawer holds the eval table, per-class counts and the SHA/tag.
- **VLM selector:** chooses among the served models, for example `context` or `context_v2`, and posts to `/api/models/vlm`.
- **Compare on an image:** upload or pick a sample, then run the before and after pipelines through `/lab/api/analyze`. Shows two images with boxes and a one-line verdict each. Port the rendering from `static/webapp.html`.

---

## Release

### Task 13: run, docs and tracker
- **`scripts/start_console.sh`:**
  - Reuses the VLM check and start from `start_demo.sh` (zrt, base7b plus the `context` and `context_v2` LoRA adapters).
  - Starts one tmux session `console`: `python -m sentinel.console --port 8080 ...` with the same weights, assets and state flags, loading `~/.sentinel_alerts.env` (`set -a`) and `~/.cloud_vlm.env` if present.
  - Stops nothing that belongs to others.
  - Prints the tunnel line `ssh -N -L 8080:localhost:8080 hp11@<nano-ip>`.
- **README:** a "Sentinel Console" section (one screenshot placeholder, run command, page list).
- **`docs/versions/v2.0.0-console.md`:** what changed, models (unchanged: v1.3.0-joint, LoRA v1 with v2 selectable), how to run, verification, restore.
- **`docs/PROGRESS.md`:** a "Sentinel Console (v2.0.0)" section with one line per task, each ticked.

### Task 14: Nano verification (controller)
1. Before touching `~/sentinel`, check its branch, status and reflog (shared host).
2. Pull `feat/console` with `--ff-only` into our own clone, or into `~/sentinel` if it is on our branch.
3. Run `start_console.sh` and open the tunnel.
4. Smoke-test each page. Test outage → mobile-camera ALERT queued → uplink restored → phone buzzes once → incident on the map → Edge vs Cloud counts move.
5. Take screenshots into `results/console/`.
6. Tag `v2.0.0-console`, merge to `main` and push.

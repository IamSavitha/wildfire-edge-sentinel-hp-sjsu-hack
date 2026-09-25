// Cameras: tower feeds, a recorded-fire drill, and the mobile camera (laptop webcam or phone) through the same
// pipeline. The camera needs a secure context: open the console on localhost (on the node or via a tunnel).
import {del, get, poll, post} from "../api.js";
import {$, dot, empty, esc, icon, ms, openDrawer, pill, title, toast} from "../ui.js";

const INTERVAL_MS = 1500;
let live = null;

export function mount(el, ctx) {
  el.innerHTML = `<div class="page">
    <div class="row wrap" style="margin-bottom:18px">
      <p class="lead grow" style="margin:0">Every camera runs through the same detector, gate and VLM on this device.</p>
      <div class="row" id="drill-ctl"></div>
    </div>
    <div class="grid" style="grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);align-items:start" id="top">
      <div class="card" style="padding:0;overflow:hidden">
        <div class="live-view" id="lv"><video id="video" playsinline muted></video><canvas id="overlay"></canvas>
          <div class="idle" id="idle"><div>${icon("mobile")}<div style="margin-top:6px;color:var(--text);font-weight:600">Mobile unit 1</div>
            <div class="small">Laptop webcam or an iPhone (Continuity Camera)</div></div></div>
          <div class="hud" id="hud" hidden></div></div>
        <div class="row" style="padding:12px 14px">
          <select class="input" id="cam" style="max-width:260px" aria-label="Camera"><option value="">Default camera</option></select>
          <span class="grow small muted" id="msg"></span>
          <button class="btn sm ghost" id="details" type="button">Details</button>
          <button class="btn primary" id="start" type="button">${icon("play")}Start</button>
        </div>
      </div>
      <div class="card"><h3>${icon("mobile")}Mobile unit <span class="spacer"></span><span id="mstate"></span></h3><div id="mcard"></div></div>
    </div>
    <div class="section-title">Towers</div>
    <div class="tiles" id="tiles"></div>
  </div>`;
  if (window.matchMedia("(max-width: 900px)").matches) $("#top", el).style.gridTemplateColumns = "1fr";

  // ---------------------------------------------------------------- towers + drill
  let cams = [], sessions = [], recordings = [];
  async function refreshTowers() {
    const [st, w] = await Promise.all([get("/ops/api/state?hours=2"), get("/ops/api/watch")]);
    cams = st.cameras.filter(c => !c.mobile);
    sessions = w.sessions.filter(s => s.state === "running");
    const byCam = {};
    for (const s of sessions) for (const c of s.cams) byCam[c.camera_id] = {sid: s.id, ...c};
    const incByCam = {};
    for (const i of st.incidents) if (i.status === "active") incByCam[i.camera_id] = i;
    $("#tiles", el).innerHTML = cams.map(c => {
      const w = byCam[c.id], inc = incByCam[c.id];
      const img = w && w.has_frame ? `<img alt="${esc(c.name)}" src="/ops/api/watch/${w.sid}/frame/${encodeURIComponent(c.id)}?t=${Date.now()}">`
        : `<div class="faint">${icon("tower")}</div>`;
      const status = inc ? pill(inc.severity, inc.severity) : w ? pill("Watching", "ok") : pill("Idle", "tag");
      return `<div class="tile"><div class="img">${img}</div><div class="cap">${dot(inc ? inc.severity : w ? "ok live" : "")}
        <div class="grow"><div class="name ellipsis">${esc(c.name)}</div><div class="small muted ellipsis">${esc(c.site || "")}</div></div>${status}</div></div>`;
    }).join("") || empty("No tower cameras configured");
    drawDrill();
  }
  function drawDrill() {
    const ctl = $("#drill-ctl", el);
    if (sessions.length) {
      const s = sessions[0];
      ctl.innerHTML = `<span class="small muted">${esc(s.label)} · ${s.idx}/${s.total} frames</span>
        <button class="btn sm" id="drill-stop" type="button">${icon("stop")}Stop drill</button>`;
      $("#drill-stop", el).onclick = async () => { await Promise.all(sessions.map(x => del(`/ops/api/watch/${x.id}`).catch(() => {}))); refreshTowers(); };
    } else if (!recordings.length) {
      ctl.innerHTML = "";
    } else {
      ctl.innerHTML = `<select class="input" id="rec" style="max-width:240px" aria-label="Recorded fire">
        ${recordings.map(r => `<option value="${esc(r.key)}">${esc(r.fire)}${r.date ? " · " + esc(r.date) : ""}</option>`).join("")}
        ${recordings.length > 1 ? `<option value="all">All recordings</option>` : ""}</select>
        <button class="btn" id="drill" type="button">${icon("play")}Run drill</button>`;
      $("#drill", el).onclick = async () => {
        try {
          await post("/ops/api/watch", {recording: $("#rec", el).value, interval_s: 1, batch_frames: 5, time_mode: "live", start_offset_s: -300});
          toast("Drill started: a recorded fire is replayed through the tower cameras.");
          refreshTowers();
        } catch (e) { toast(e.message, true); }
      };
    }
  }
  get("/ops/api/recordings").then(r => { recordings = r.recordings; drawDrill(); }).catch(() => {});
  const stopTowers = poll(refreshTowers, 2000);

  // ---------------------------------------------------------------- mobile camera
  live = {running: false, inflight: false, stream: "s" + Math.random().toString(36).slice(2, 10), media: null,
                  sent: 0, last: null, timer: null};
  const video = $("#video", el), canvas = $("#overlay", el), capture = document.createElement("canvas");
  const msg = (t, err = false) => { const m = $("#msg", el); m.textContent = t; m.style.color = err ? "var(--alert)" : ""; };

  async function listCams(keep) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    let devs = [];
    try { devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput" && d.deviceId); } catch (e) { return; }
    const sel = $("#cam", el);
    const want = keep || sel.value;
    sel.innerHTML = `<option value="">Default camera</option>` + devs.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || "Camera " + (i + 1))}</option>`).join("");
    if (want && [...sel.options].some(o => o.value === want)) sel.value = want;
  }
  function camError(e) {
    const n = e && e.name;
    if (n === "NotAllowedError" || n === "SecurityError") return "Camera permission denied: allow it for this site (and in macOS Privacy settings).";
    if (n === "NotFoundError" || n === "OverconstrainedError") return "No such camera. Pick another one.";
    if (n === "NotReadableError") return "The camera is in use by another app.";
    return "Could not open the camera: " + ((e && e.message) || n);
  }
  async function openCam() {
    const id = $("#cam", el).value;
    const v = {width: {ideal: 1280}, height: {ideal: 720}};
    if (id) v.deviceId = {exact: id};
    const media = await navigator.mediaDevices.getUserMedia({video: v, audio: false});
    if (live.media) live.media.getTracks().forEach(t => t.stop());
    live.media = media;
    attach();
    const track = media.getVideoTracks()[0];
    await listCams(track && track.getSettings ? track.getSettings().deviceId : "");
    return track ? track.label : "";
  }
  function attach() {
    if (!live.media) return;
    video.srcObject = live.media;
    video.play().catch(() => {});
    $("#idle", el).hidden = true;
    $("#hud", el).hidden = false;
    $("#start", el).innerHTML = `${icon("stop")}Stop`;
    $("#start", el).className = "btn";
  }
  async function start() {
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      msg(window.isSecureContext === false ? "The browser only allows the camera on localhost (open the console through a tunnel or on the node)." :
        "This browser has no camera API.", true);
      return;
    }
    try {
      msg("Opening the camera…");
      const label = await openCam();
      live.running = true;
      msg(label ? `Streaming from ${label}` : "Streaming");
      await post("/lab/api/live/reset").catch(() => {});
      schedule(0);
    } catch (e) { msg(camError(e), true); }
  }
  function stop() {
    live.running = false;
    clearTimeout(live.timer);
    if (live.media) live.media.getTracks().forEach(t => t.stop());
    live.media = null;
    video.srcObject = null;
    $("#idle", el).hidden = false;
    $("#hud", el).hidden = true;
    $("#start", el).innerHTML = `${icon("play")}Start`;
    $("#start", el).className = "btn primary";
    paint(null);
    msg("Stopped.");
  }
  const schedule = d => { clearTimeout(live.timer); if (live.running) live.timer = setTimeout(tick, Math.max(0, d)); };
  function grab() {
    const v = video;
    if (!v.videoWidth) return Promise.resolve(null);
    const s = Math.min(1, 1280 / Math.max(v.videoWidth, v.videoHeight));
    capture.width = Math.round(v.videoWidth * s); capture.height = Math.round(v.videoHeight * s);
    capture.getContext("2d").drawImage(v, 0, 0, capture.width, capture.height);
    return new Promise(r => capture.toBlob(r, "image/jpeg", 0.85));
  }
  async function tick() {
    if (!live.running || live.inflight) return;
    const t0 = performance.now();
    live.inflight = true;
    try {
      const blob = await grab();
      if (!blob) return;
      const r = await fetch(`/lab/api/live/frame?stream=${live.stream}`, {method: "POST", headers: {"Content-Type": "image/jpeg"}, body: blob, cache: "no-store"});
      if (r.status === 202) return;
      const d = await r.json().catch(() => ({}));
      if (!r.ok) { msg(typeof d.detail === "string" ? d.detail : `Frame rejected (${r.status})`, true); return; }
      live.sent++;
      live.last = d;
      if (d.new_alert) { toast("ALERT raised by the mobile unit"); ctx.refresh(); }
      paint(d);
    } catch (e) { msg("The node is not answering: " + e.message, true); }
    finally { live.inflight = false; schedule(INTERVAL_MS - (performance.now() - t0)); }
  }
  function paint(d) {
    const cw = canvas.clientWidth, ch = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (cw && ch) {
      canvas.width = Math.round(cw * dpr); canvas.height = Math.round(ch * dpr);
      const g = canvas.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, cw, ch);
      if (d && d.frame && live.running) {
        const s = Math.min(cw / d.frame.width, ch / d.frame.height), ox = (cw - d.frame.width * s) / 2, oy = (ch - d.frame.height * s) / 2;
        g.font = "600 12px -apple-system, system-ui, sans-serif";
        for (const det of (d.detections || []).slice(0, 20)) {
          const [x1, y1, x2, y2] = det.box, x = ox + x1 * s, y = oy + y1 * s;
          g.lineWidth = 2.5; g.strokeStyle = det.above_gate ? "#ff5a3c" : "rgba(255,255,255,.45)"; g.setLineDash(det.above_gate ? [] : [5, 4]);
          g.strokeRect(x, y, Math.max(1, (x2 - x1) * s), Math.max(1, (y2 - y1) * s));
          g.setLineDash([]); g.fillStyle = det.above_gate ? "#ff5a3c" : "rgba(0,0,0,.6)";
          const label = `${det.cls} ${det.conf.toFixed(2)}`, tw = g.measureText(label).width + 10;
          g.fillRect(x, y - 20 < 0 ? y : y - 20, tw, 20); g.fillStyle = "#fff"; g.fillText(label, x + 5, (y - 20 < 0 ? y : y - 20) + 14);
        }
      }
    }
    const hud = $("#hud", el), card = $("#mcard", el), mstate = $("#mstate", el);
    if (!d) { hud.innerHTML = ""; card.innerHTML = `<p class="muted small" style="margin:0">Point the camera at smoke (a wildfire video on a tablet works). Three frames in a row with smoke open an event, the VLM classifies it once, and an ALERT reaches dispatch.</p>`; mstate.innerHTML = live.running ? pill("Starting", "tag") : pill("Off", "tag"); return; }
    const g = d.gate || {streak: 0, needed: 3}, ev = d.event, latched = d.latched || {};
    const gate = `<span class="gate">${Array.from({length: g.needed}, (_, i) => `<i class="${i < g.streak || ev || latched.active ? "on" : ""}"></i>`).join("")}</span>`;
    const status = ev ? (ev.status === "classifying" ? "Classifying" : `Monitoring · ${title(ev.source_type)}`) : latched.active ? "Alerted" : g.streak ? "Smoke seen" : "Watching";
    hud.innerHTML = `<span class="chip">${gate}${esc(status)}</span>${d.timings ? `<span class="chip num">${ms(d.timings.total_ms)}</span>` : ""}
      ${d.online === false ? `<span class="chip off">${icon("wifiOff")}Offline: deciding locally</span>` : ""}`;
    const last = (d.events || [])[0];
    mstate.innerHTML = ev && ev.severity ? pill(ev.severity, ev.severity) : latched.active ? pill("ALERT", "ALERT") : pill("Watching", "ok");
    card.innerHTML = last ? `<div class="row" style="margin-bottom:8px">${pill(last.severity || "…", last.severity || "")}<strong class="grow ellipsis">${esc(title(last.source_type))}</strong></div>
      <div class="muted small">${esc(last.description || "")}</div>
      ${last.delivery ? `<div class="row small" style="margin-top:10px">${icon("send")}<span class="grow">Dispatch</span>${pill(title(last.delivery.state), last.delivery.state)}</div>` : ""}`
      : `<p class="muted small" style="margin:0">No event yet · ${live.sent} frames analysed on the device.</p>`;
  }
  $("#start", el).onclick = () => live.running ? stop() : start();
  $("#cam", el).onchange = async () => { if (live.running) { try { await openCam(); } catch (e) { stop(); msg(camError(e), true); } } };
  $("#details", el).onclick = () => {
    const d = live.last;
    openDrawer("Mobile unit · last frame", d ? `<dl class="kv">
      <dt>Frame</dt><dd>${d.frame.width} × ${d.frame.height}</dd>
      <dt>Detector</dt><dd>${ms(d.timings.detect_ms)}</dd>
      <dt>VLM</dt><dd>${d.vlm.status ? `${esc(d.vlm.status)} · ${ms(d.timings.vlm_ms)} · ${d.vlm.tokens} tokens` : "not needed for this frame"}</dd>
      <dt>Gate</dt><dd>${d.gate.streak}/${d.gate.needed} frames ≥ ${d.gate.min_conf}</dd>
      <dt>Detections</dt><dd>${(d.detections || []).map(x => `${esc(x.cls)} ${x.conf.toFixed(2)}`).join(", ") || "none"}</dd>
      <dt>Outbox</dt><dd>${d.delivery.pending} queued · ${d.delivery.delivered} delivered</dd>
      <dt>Frames sent</dt><dd>${live.sent}</dd></dl>` : `<p class="muted">Start the camera to see per-frame details.</p>`);
  };
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) navigator.mediaDevices.addEventListener("devicechange", () => listCams());
  listCams();
  paint(null);
  const onResize = () => paint(live.last);
  window.addEventListener("resize", onResize);
  return () => { stopTowers(); window.removeEventListener("resize", onResize); if (live.running) stop(); };
}

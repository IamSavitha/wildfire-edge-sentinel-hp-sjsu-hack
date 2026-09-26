// Officer console add-on: the parts of the earlier apps the forest-officer console did not have, as extra tabs
// in its own style: Edge vs Cloud (economics and no-network operation), the mobile camera, models (before/after)
// and system (health, live model metrics, alert delivery). Loaded after the page's own script, whose helpers
// (api, post, esc, toast, showTab, drawStats, refresh, STATE) it reuses. Everything is served by this device.
(() => {
  const q = (s, el = document) => el.querySelector(s);
  const qa = (s, el = document) => [...el.querySelectorAll(s)];
  const fmt = {
    n: (v, d = 0) => v == null || !isFinite(v) ? "–" : Number(v).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}),
    k: v => v == null || !isFinite(v) ? "–" : Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(v),
    b(v) { if (v == null || !isFinite(v)) return "–"; const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
      while (Math.abs(v) >= 1000 && i < u.length - 1) { v /= 1000; i++; } return `${v.toFixed(v < 10 && i ? 1 : 0)} ${u[i]}`; },
    ms: v => v == null || !isFinite(v) ? "–" : v < 1000 ? `${Math.round(v)} ms` : v < 60000 ? `${(v / 1000).toFixed(1)} s` : `${(v / 60000).toFixed(1)} min`,
    usd: v => v == null || !isFinite(v) ? "–" : v >= 100 ? `$${fmt.n(v)}` : v >= 1 ? `$${v.toFixed(2)}` : v >= 0.0001 ? `$${v.toFixed(4)}` : v > 0 ? "< $0.0001" : "$0",
    x: v => v == null || !isFinite(v) ? "–" : v >= 100 ? `${fmt.k(v)}×` : `${v.toFixed(v < 10 ? 1 : 0)}×`,
    pct: v => v == null || !isFinite(v) ? "–" : `${(v * 100).toFixed(1)}%`,
    words: s => String(s || "unknown").replace(/_/g, " "),
  };
  const help = t => `<span class="x-help" title="${esc(t)}">?</span>`;
  const kpi = (v, k, s = "", cls = "") => `<div class="x-kpi ${cls}"><div class="v">${v}</div><div class="k">${esc(k)}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`;
  const bar = (who, v, max, label) => `<div class="x-bar ${who}"><span class="who">${who === "edge" ? "Edge" : "Cloud-only"}</span>
    <div class="tr"><div class="fl" style="width:${v > 0 && max > 0 ? Math.max(1, 100 * v / max).toFixed(1) : 0}%"></div></div><span class="val">${label}</span></div>`;

  const PROFILES = [["fiber", "Fiber"], ["lte", "LTE"], ["rural_cellular", "Rural cell"], ["satellite_geo", "Satellite"], ["outage", "No link"]];
  let PROFILE = "rural_cellular", EC = null, FLEET = null, fleetArgs = {towers: 20, fps: 1, days: 30}, built = false;

  // ---------------------------------------------------------------- tabs
  const TABS = [["edge", "Edge vs Cloud"], ["camera", "Mobile camera"], ["models", "Models"], ["system", "System"]];
  const tabs = q("aside .tabs"), aside = q("aside");
  for (const [id, label] of TABS) {
    const b = document.createElement("button");
    b.className = "tab"; b.dataset.tab = id; b.textContent = label;
    b.onclick = () => showTab(id);
    tabs.appendChild(b);
    const p = document.createElement("div");
    p.className = "pane"; p.id = "pane-" + id;
    p.innerHTML = `<div class="scroll" id="x-${id}"></div>`;
    aside.appendChild(p);
  }
  const DRAW = {edge: drawEdge, models: drawModels, system: drawSystem};
  let active = null;
  const baseShowTab = showTab;
  showTab = function (name) {                         // eslint-disable-line no-global-assign
    baseShowTab(name);
    active = name;
    q("main").classList.toggle("wide", TABS.some(([id]) => id === name));
    if (name === "edge" && q("#x-race-home") && !LIVE.running) raceTo(q("#x-race-home"), false);
    if (DRAW[name]) DRAW[name]();
    if (name === "camera") drawCamera();
    const stage = q("#x-stage");
    if (stage) stage.classList.toggle("pip", name !== "camera");   // the camera keeps running in a corner
  };
  setInterval(() => { if (!document.hidden && DRAW[active] && active !== "models") DRAW[active](); }, 3000);
  const linked = (location.hash.match(/^#tab=([a-z]+)/) || [])[1];      // e.g. /#tab=edge opens that tab
  if (linked && q(`.tab[data-tab="${linked}"]`)) setTimeout(() => showTab(linked), 0);
  // /#tab=camera&start opens straight into the live camera (one click fewer on stage)
  if (linked === "camera" && /[&?]start\b/.test(location.hash)) setTimeout(() => startCam(), 400);

  // ---------------------------------------------------------------- stats strip: edge vs cloud-only
  let OV = null;
  const baseDrawStats = drawStats;
  drawStats = function () {                           // eslint-disable-line no-global-assign
    baseDrawStats();
    if (!OV || !OV.ec) return;
    const s = OV.ec, e = s.edge, c = s.cloud;
    const items = [
      ["Data kept local", c.bytes_up ? fmt.b(Math.max(0, c.bytes_up - e.bytes_up)) : "–", "good"],
      ["VLM calls avoided", fmt.n(s.savings.vlm_calls_avoided), "good"],
      ["Tokens avoided", s.savings.tokens_pct == null ? "–" : `${s.savings.tokens_pct.toFixed(0)}%`, "good"],
      ["Decided with no link", fmt.n(s.edge_offline_decisions), s.edge_offline_decisions ? "queued" : ""],
      [s.prices_set ? "Cloud cost avoided" : "Cloud cost (set prices)", s.prices_set ? fmt.usd(s.savings.usd) : "–", "good"]];
    q("#stats").insertAdjacentHTML("beforeend", `<div class="group" id="x-vs-group" title="Compared with a cloud-only design that sends every frame to a hosted VLM. Open the Edge vs Cloud tab for details.">
      <div class="gl">vs Cloud</div>${items.map(([k, v, cls]) => `<div class="stat ${cls}"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("")}</div>`);
    q("#x-vs-group").onclick = () => showTab("edge");
  };
  async function pollOverview() {
    try {
      const [o, ec] = await Promise.all([api("/api/overview"), api("/api/edge-cloud?profile=" + PROFILE)]);
      OV = {...o, ec: ec.summary};
      EC = ec;
      if (STATE) drawStats();
      onPipelinePoll();
    } catch (e) { /* the page's own banner reports a lost service */ }
  }
  setInterval(() => { if (!document.hidden) pollOverview(); }, 3000);
  pollOverview();

  // ---------------------------------------------------------------- Edge vs Cloud

  async function drawEdge() {
    try { EC = await api("/api/edge-cloud?profile=" + PROFILE); } catch (e) { return; }
    const box = q("#x-edge");
    if (!built) {
      built = true;
      box.innerHTML = `
        <div class="x-sec"><div class="note"><b>Same frames, two designs.</b> Every frame this device analysed is compared with a cloud-only
          system that uploads each frame to a hosted VLM (same Qwen2.5-VL-7B). Edge numbers are measured; cloud-only is modelled from the real frames.</div>
          <div class="actions" id="x-prof">${PROFILES.map(([v, l]) => `<button class="chip" data-p="${v}">${l}</button>`).join("")}</div></div>
        <div id="x-race-home"></div>
        <div class="x-sec"><div class="x-kpis" id="x-ekpi"></div></div>
        <div class="x-sec"><h4>Edge vs cloud-only<span class="sp"></span><span id="x-tag"></span></h4><div class="x-vs" id="x-bars"></div></div>
        <div class="x-sec" id="x-outage"></div>
        <div class="x-sec"><h4>Benchmark on labelled tower clips</h4><div id="x-bench"></div></div>
        <div class="x-sec"><h4>Fleet projection ${help("Scale the measured per-frame numbers to a network of towers. Cloud-only runs a VLM on every frame; local-every-frame shows why one GPU cannot run the VLM on everything; the cascade runs the detector on every frame and the VLM only on persistent smoke.")}</h4>
          <div class="two"><div class="field"><span>Towers <b id="x-f-towers"></b></span><input type="range" min="1" max="500" value="${fleetArgs.towers}" data-f="towers"></div>
            <div class="field"><span>Days <b id="x-f-days"></b></span><input type="range" min="1" max="365" value="${fleetArgs.days}" data-f="days"></div></div>
          <div class="field"><span>Frames per second per camera <b id="x-f-fps"></b></span><input type="range" min="0.2" max="5" step="0.2" value="${fleetArgs.fps}" data-f="fps"></div>
          <div id="x-fleet"></div></div>
        <div class="x-sec"><details><summary>Prices (cloud API and data), your inputs</summary>
          <div class="two" style="margin-top:8px">${["usd_per_mtok_in:$ / 1M input tokens", "usd_per_mtok_out:$ / 1M output tokens", "usd_per_gb:$ / GB uplinked"].map(x => {
            const [k, l] = x.split(":"); return `<div class="field"><span>${l}</span><input type="number" min="0" step="any" data-price="${k}" value="${EC.prices[k] || ""}" placeholder="0"></div>`; }).join("")}</div>
          <div class="actions" style="margin-top:8px"><button class="btn primary" id="x-save">Apply</button><span class="note">Use the provider's current published rates.</span></div></details></div>
        <div class="x-sec"><details><summary>How each number is computed</summary><div id="x-method" class="note" style="margin-top:8px"></div>
          <div class="actions" style="margin-top:8px"><button class="btn" id="x-reset">Reset the comparison</button></div></details></div>
        <div class="x-sec" id="x-measured" hidden></div>`;
      q("#x-prof").onclick = e => { const b = e.target.closest("[data-p]"); if (!b) return; PROFILE = b.dataset.p; drawEdge(); pollOverview(); };
      qa("input[data-f]", box).forEach(r => r.oninput = () => { fleetArgs[r.dataset.f] = +r.value; loadFleet(); });
      q("#x-save").onclick = async () => {
        const body = {}; qa("input[data-price]", box).forEach(i => { if (i.value !== "") body[i.dataset.price] = +i.value; });
        try { await post("/api/edge-cloud/prices", body); toast("Prices applied"); drawEdge(); pollOverview(); } catch (e) { toast(e.message); }
      };
      q("#x-reset").onclick = async () => { await post("/api/edge-cloud/reset", {}); drawEdge(); pollOverview(); };
      loadFleet();
      raceTo(q("#x-race-home"), false);
      const lf = EC.summary.last_fire;
      if (lf && !LIVE.running) { seenFire = lf.id; race(Promise.resolve(fireFrom(lf)), lf.online, lf.cloud_upload_ms, RACE_REPLAY); }
    }
    qa("#x-prof .chip").forEach(b => b.classList.toggle("on", b.dataset.p === PROFILE));
    const s = EC.summary, e = s.edge, c = s.cloud, sv = s.savings, m = s.method;
    q("#x-tag").innerHTML = EC.measured.enabled && EC.measured.n ? `<span class="x-pill m">${EC.measured.n} measured</span>` : `<span class="x-pill">Cloud modelled</span>`;
    const cloudTime = s.link_down ? "no decision" : fmt.ms(c.decide_ms_p50);
    q("#x-ekpi").innerHTML = [
      kpi(sv.bytes_x ? fmt.x(sv.bytes_x) : c.bytes_up ? fmt.b(sv.bytes) : "–", sv.bytes_x ? "less data sent upstream" : "not uploaded by the edge",
        `Edge ${fmt.b(e.bytes_up)} · cloud-only ${fmt.b(c.bytes_up)}`, "edge"),
      kpi(s.prices_set ? fmt.usd(sv.usd) : "Set prices", "cloud API + data cost avoided",
        s.prices_set ? `Cloud-only ${fmt.usd(c.usd)} · edge ${fmt.usd(e.usd)}` : `Tokens avoided: ${sv.tokens_pct == null ? "–" : sv.tokens_pct.toFixed(0) + "%"}`, "good"),
      kpi(fmt.ms(e.decide_ms_p50), "edge time to classify a plume", `Cloud-only on ${PROFILES.find(p => p[0] === PROFILE)[1]}: ${cloudTime}`),
      kpi(fmt.n(s.edge_offline_decisions), "frames decided with no link",
        s.cloud_blind_frames ? `Cloud-only was blind for ${fmt.n(s.cloud_blind_frames)}` : "Try “Simulate outage” in the header", s.edge_offline_decisions ? "hot" : ""),
    ].join("");
    const rows = [["Data uplinked", e.bytes_up, c.bytes_up, fmt.b, m.cloud_bytes], ["VLM calls", e.vlm_calls, c.vlm_calls, v => fmt.n(v), m.edge_tokens],
      ["Tokens", e.tokens, c.tokens, fmt.k, m.cloud_tokens],
      ["Time to decision", e.decide_ms_p50, s.link_down ? 0 : c.decide_ms_p50, fmt.ms, m.cloud_time, s.link_down ? "no decision" : null]];
    if (s.prices_set) rows.push(["Cost", e.usd, c.usd, fmt.usd, m.cloud_usd]);
    q("#x-bars").innerHTML = e.frames ? rows.map(([l, ev, cv, f, tip, cl]) => {
      const max = Math.max(ev || 0, cv || 0);
      return `<div class="x-row"><div class="l">${esc(l)} ${help(tip)}</div>${bar("edge", ev || 0, max, f(ev))}${bar("cloud", cv || 0, max, cl || f(cv))}</div>`;
    }).join("") : `<div class="empty">No frames yet. Start <b>Live watch</b>, the <b>Mobile camera</b>, or report a sighting: every frame the device analyses is compared here.</div>`;
    q("#x-outage").innerHTML = outage();
    q("#x-bench").innerHTML = bench();
    q("#x-method").innerHTML = Object.entries(m).map(([k, v]) => `<p style="margin:4px 0"><b>${esc(fmt.words(k))}:</b> ${esc(v)}</p>`).join("");
    const ms = EC.measured, mbox = q("#x-measured");
    mbox.hidden = !ms.enabled;
    if (ms.enabled) mbox.innerHTML = `<h4>Measured cloud samples</h4><div class="x-kpis">${kpi(fmt.n(ms.n), "frames sent to " + esc(ms.host))}
      ${kpi(ms.agreement_pct == null ? "–" : ms.agreement_pct.toFixed(0) + "%", "same source type as the edge")}${kpi(fmt.ms(ms.cloud_ms_p50), "cloud latency (p50)")}
      ${kpi(fmt.k(ms.cloud_tokens_p50), "cloud tokens per frame")}</div><div class="note">One frame per incident, at most ${ms.cap_per_hour} an hour, cached.</div>`;
    drawFleet();
  }
  function outage() {
    const s = EC.summary, blind = s.cloud_blind_frames;
    return `<h4>When the network is gone ${help("During an outage a cloud-only system cannot see or decide anything. The edge keeps detecting and classifying, keeps the incident map current, and holds ALERTs in a durable outbox until the link returns.")}</h4>
      <div class="x-kpis">${kpi(fmt.n(s.edge_offline_decisions), "frames the edge decided offline", "", s.edge_offline_decisions ? "hot" : "")}
        ${kpi(fmt.n(blind), "frames cloud-only could not decide")}${kpi(fmt.n(OV ? OV.uplink.queued : 0), "ALERTs waiting in the outbox")}
        ${kpi(OV && OV.uplink.online ? "Up" : "Down", "dispatch link now", OV && !OV.uplink.online ? "Detection continues on the device" : "")}</div>`;
  }
  function bench() {
    const b = EC.benchmarks, ba = b.before_after || {}, ed = b.edge, cl = b.cloud;
    const pr = v => v == null ? "–" : `${(v * 100).toFixed(0)}%`;
    let html = "";
    if (ed) html += `<table class="x-t"><tr><th></th><th>Edge cascade</th><th>Cloud-only</th></tr>
      ${[["Precision", "precision", pr], ["Recall", "recall", pr], ["False alarms", "false_alarms", v => fmt.n(v)],
         ["Frames per VLM call", "frames_per_vlm_call", v => fmt.n(v, 1)], ["Time to decision (p50)", "time_to_decision_s_p50", v => v == null ? "–" : `${fmt.n(v, 1)} s`]]
        .map(([l, k, f]) => `<tr><td>${l}</td><td>${f(ed[k])}</td><td>${cl ? f(cl[k]) : '<span class="note">needs a provider key</span>'}</td></tr>`).join("")}</table>
      <div class="note">${fmt.n(ed.n_clips)} labelled tower clips (results/bench_after.json).</div>`;
    const d = ba.detector || {}, v = ba.context || {};
    if (d.before || d.after || v.before || v.after) html += `<table class="x-t" style="margin-top:8px"><tr><th>Before → after fine-tuning</th><th>Before</th><th>After</th></tr>
      <tr><td>Detector mAP50 (D-Fire)</td><td>${d.before ? d.before.map50.toFixed(3) : "–"}</td><td class="win">${d.after ? d.after.map50.toFixed(3) : "–"}</td></tr>
      <tr><td>VLM source-type agreement</td><td>${v.before ? pr(v.before.source_type_acc) : "–"}</td><td class="win">${v.after ? pr(v.after.source_type_acc) : "–"}</td></tr>
      <tr><td>VLM tokens per call</td><td>${v.before ? fmt.n(v.before.tokens_per_call) : "–"}</td><td>${v.after ? fmt.n(v.after.tokens_per_call) : "–"}</td></tr></table>`;
    return html || `<div class="note">No benchmark results on this device.</div>`;
  }
  let fleetT = null;
  function loadFleet() {
    clearTimeout(fleetT);
    fleetT = setTimeout(async () => {
      try { FLEET = await api(`/api/edge-cloud/fleet?towers=${fleetArgs.towers}&fps=${fleetArgs.fps}&days=${fleetArgs.days}`); drawFleet(); } catch (e) { /* keep */ }
    }, 150);
  }
  function drawFleet() {
    for (const k of ["towers", "fps", "days"]) { const el = q("#x-f-" + k); if (el) el.textContent = fleetArgs[k]; }
    const box = q("#x-fleet");
    if (!box || !FLEET || !EC) return;
    const names = ["Cloud VLM on every frame", "Local VLM on every frame", "Edge cascade (this system)"];
    box.innerHTML = `<table class="x-t"><tr><th></th><th>VLM calls</th><th>Uplinked</th><th>Cost</th><th>Fits 1 GPU</th></tr>
      ${FLEET.rows.map((r, i) => `<tr><td>${names[i]}</td><td>${fmt.k(r.vlm_calls_total)}</td><td>${fmt.b(r.upstream_gb_total * 1e9)}</td>
        <td${i === 2 ? ' class="win"' : ""}>${EC.summary.prices_set ? fmt.usd(r.usd_per_day_total) : "–"}</td><td>${r.feasible ? "yes" : "no"}</td></tr>`).join("")}</table>
      <div class="note">Measured on this device: ${FLEET.measured.length ? esc(FLEET.measured.join(", ").replace(/_/g, " ")) : "nothing yet (stated defaults)"}.</div>`;
  }

  // ---------------------------------------------------------------- live pipeline race
  // One fire frame through both designs, phase by phase, with the real timings of that frame: the edge ran
  // detector + VLM here; cloud-only would upload the whole frame over the selected link and run the same model.
  const STEPS = ["Captured", "Detector", "Upload", "VLM", "Decision", "Alert"];
  const RACE_LIVE = 1, RACE_REPLAY = 0.45;      // live: real time; replays: sped up, same proportions
  let seenFire, raceToken = 0, pendingRestore = null, lastOnline = null;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const fireFrom = f => ({detect_ms: f.detect_ms, vlm_ms: f.vlm_ms, severity: f.severity,
                          decision: f.severity === "ALERT" ? (f.online ? "sent" : "queued") : "logged"});

  const RACE = Object.assign(document.createElement("div"), {className: "x-sec x-race", id: "x-race"});
  let raceBuilt = false;
  function buildRace() {
    if (raceBuilt) return;
    raceBuilt = true;
    const box = RACE;
    const cell = (lane, i) => `<div class="st${i ? "" : " first"}" id="x-${lane}${i}" style="grid-row:${lane === "e" ? 2 : 3};grid-column:${i + 2}">
      <span class="o">·</span><small></small></div>`;
    box.innerHTML = `<h4><span id="x-race-title">Live pipeline: one fire frame, two designs</span> <span class="sp"></span>
        <span class="actions" id="x-race-btns"><button class="btn ember" id="x-fire">Send a fire frame</button><button class="btn" id="x-replay">Replay</button></span></h4>
      <div class="x-rg" id="x-rg">
        <div style="grid-row:1;grid-column:1"></div>${STEPS.map((n, i) => `<div class="hd" style="grid-row:1;grid-column:${i + 2}">${n}</div>`).join("")}
        <div class="lane-bg" id="x-lane-e" style="grid-row:2;grid-column:1/-1"></div>
        <div class="lane-bg" id="x-lane-c" style="grid-row:3;grid-column:1/-1"></div>
        <div class="ln" style="grid-row:2;grid-column:1"><b>Edge</b><small id="x-t-e">on this device</small></div>
        <div class="ln" style="grid-row:3;grid-column:1"><b>Cloud-only</b><small id="x-t-c">hosted VLM</small></div>
        ${STEPS.map((_, i) => cell("e", i)).join("")}${STEPS.map((_, i) => cell("c", i)).join("")}
        <div class="nb" id="x-nb" style="left:calc(74px + (100% - 74px) / 3)"><span>network</span></div>
      </div>
      <div class="x-restore" id="x-restore" hidden></div>
      <div class="x-race-msg" id="x-race-msg"><span class="note">Waiting for a fire frame. Press <b>Send a fire frame</b>, start <b>Live watch</b>, or use the <b>Mobile camera</b>. Try it again after <b>Simulate outage</b>.</span></div>`;
    q("#x-fire", box).onclick = sendFire;
    q("#x-replay", box).onclick = () => { const lf = EC && EC.summary.last_fire; if (lf) race(Promise.resolve(fireFrom(lf)), lf.online, lf.cloud_upload_ms, RACE_REPLAY); };
  }
  function raceTo(slot, live) {                      // show the one race panel in the Edge tab or beside the camera
    buildRace();
    if (!slot) return;
    if (RACE.parentNode !== slot) slot.appendChild(RACE);
    q("#x-race-title").textContent = live ? "Live: mobile camera → edge vs cloud" : "Live pipeline: one fire frame, two designs";
    q("#x-race-btns").hidden = !!live;
    RACE.classList.toggle("live", !!live);
  }
  function st(lane, i, state, text, tone = "") {
    const el = q(`#x-${lane}${i}`); if (!el) return;
    const o = q(".o", el);
    o.className = "o " + state;
    o.textContent = {ok: "✓", run: "", blocked: "✕", queued: "‖", skip: "–", na: "–", dash: "–", idle: "·"}[state] ?? "·";
    q("small", el).textContent = text || "";
    el.className = el.className.replace(/ (good|bad|amber|lit)/g, "") + (tone ? " " + tone : "") + (state === "ok" || state === "skip" || state === "queued" ? " lit" : "");
  }
  function resetRace() {
    for (const lane of ["e", "c"]) STEPS.forEach((_, i) => st(lane, i, "idle", ""));
    qa("#x-lane-e, #x-lane-c").forEach(l => l.classList.remove("done", "stopped"));
    const b = q("#x-blind"); if (b) b.remove();
    q("#x-restore").hidden = true;
    q("#x-t-e").textContent = "on this device"; q("#x-t-c").textContent = "hosted VLM";
  }
  // fireP resolves to {detect_ms, vlm_ms, severity, decision}; online and uploadMs are known at the start
  async function race(fireP, online, uploadMs, scale) {
    if (!q("#x-rg")) return;
    const token = ++raceToken, alive = () => token === raceToken;
    resetRace();
    pendingRestore = null;
    q("#x-nb").classList.toggle("down", !online);
    q("#x-nb span").textContent = online ? "network" : "no network";
    q("#x-race-msg").innerHTML = `<span class="note">A fire frame enters both pipelines${online ? "" : " — <b>the network is down</b>"}…</span>`;
    let fire = null;
    fireP.then(f => { fire = f; }, () => {});
    const d = ms => Math.max(350, (ms || 0) * scale);
    const t0 = performance.now();

    const edge = (async () => {
      st("e", 0, "run"); await sleep(300); if (!alive()) return; st("e", 0, "ok", "frame");
      st("e", 1, "run"); await sleep(d(fire ? fire.detect_ms : 60)); if (!alive()) return;
      st("e", 1, "ok", fire && fire.detect_ms ? fmt.ms(fire.detect_ms) : "smoke found", "good");
      st("e", 2, "skip", "0 B · stays here");
      st("e", 3, "run", "on device");
      const vStart = performance.now();
      const f = await fireP.catch(() => null); if (!alive()) return;
      if (!f) { st("e", 3, "blocked", "failed", "bad"); return; }
      const left = d(f.vlm_ms) - (performance.now() - vStart);
      if (scale < 1 && left > 0) await sleep(left);
      if (!alive()) return;
      st("e", 3, "ok", `local · ${fmt.ms(f.vlm_ms)}`, "good");
      st("e", 4, "run"); await sleep(300); if (!alive()) return;
      st("e", 4, "ok", f.severity || "decided", "good");
      if (f.decision === "queued") {
        st("e", 5, "queued", "queued locally", "amber"); pendingRestore = {t: Date.now()}; showRestore("queued");
        if (OV && OV.uplink.online) setTimeout(restore, 1200);         // the link is already back: play the delivery
      }
      else if (f.severity === "ALERT") { st("e", 5, "run"); await sleep(300); if (!alive()) return; st("e", 5, "ok", "sent", "good"); }
      else st("e", 5, "ok", "none needed", "good");
      q("#x-lane-e").classList.add("done");
      q("#x-t-e").textContent = `decided in ${fmt.ms((f.detect_ms || 0) + (f.vlm_ms || 0))}`;
      return f;
    })();

    const cloud = (async () => {
      st("c", 0, "run"); await sleep(300); if (!alive()) return; st("c", 0, "ok", "frame");
      st("c", 1, "na", "no model on site");
      st("c", 2, "run", online ? "uploading" : "");
      if (!online) {
        await sleep(700); if (!alive()) return;
        st("c", 2, "blocked", "no network", "bad");
        for (const i of [3, 4, 5]) st("c", i, "dash", "—");
        q("#x-lane-c").classList.add("stopped");
        q("#x-rg").insertAdjacentHTML("beforeend", `<div class="x-blind" id="x-blind" style="grid-row:3;grid-column:5/8">BLIND DURING OUTAGE</div>`);
        q("#x-t-c").textContent = "no decision";
        return null;
      }
      const up = uploadMs == null ? 1500 : uploadMs;
      await sleep(d(up)); if (!alive()) return;
      const bytes = EC && EC.summary.last_fire ? EC.summary.last_fire.cloud_bytes : null;
      st("c", 2, "ok", `${bytes ? fmt.b(bytes) + " · " : ""}${fmt.ms(up)}`);
      st("c", 3, "run", "in the cloud");
      const vStart = performance.now();
      const f = await fireP.catch(() => null); if (!alive() || !f) return;
      const left = d(f.vlm_ms) - (performance.now() - vStart);    // same model: same compute time, after the upload
      if (left > 0) await sleep(left);
      if (!alive()) return;
      st("c", 3, "ok", `cloud · ${fmt.ms(f.vlm_ms)}`);
      st("c", 4, "run"); await sleep(300); if (!alive()) return;
      st("c", 4, "ok", f.severity || "decided");
      if (f.severity === "ALERT") { st("c", 5, "run"); await sleep(300); if (!alive()) return; st("c", 5, "ok", "sent after reply"); }
      else st("c", 5, "ok", "none needed");
      q("#x-t-c").textContent = `decided in ${fmt.ms((f.vlm_ms || 0) + up)}`;
      return f;
    })();

    const [ef, cf] = await Promise.all([edge, cloud]);
    if (!alive() || !ef) return;
    const edgeMs = (ef.detect_ms || 0) + (ef.vlm_ms || 0);
    q("#x-race-msg").innerHTML = online
      ? `<b class="g">Both decided.</b> Edge: ${fmt.ms(edgeMs)}, and the frame never left the device${ef.severity === "ALERT" ? " (only the small ALERT report did)" : ""}. Cloud-only: ${cf ? fmt.ms((cf.vlm_ms || 0) + (uploadMs || 0)) : "–"} after uploading the whole frame${EC && EC.summary.last_fire ? ` (${fmt.b(EC.summary.last_fire.cloud_bytes)})` : ""}.`
      : `<b class="g">Edge decided in ${fmt.ms(edgeMs)} with no network</b>${ef.decision === "queued" ? " and queued the ALERT locally" : ""}. <b class="r">Cloud-only stopped at the network boundary: blind during the outage.</b>`;
  }
  function showRestore(stage) {
    const box = q("#x-restore"); if (!box) return;
    box.hidden = false;
    const c = (cls, t) => `<span class="c ${cls}">${t}</span>`;
    box.innerHTML = [c("amber", "‖ Queued locally"), "→", c(stage === "restoring" ? "run" : stage === "sent" ? "ok" : "", stage === "queued" ? "Network restored" : "✓ Network restored"),
      "→", c(stage === "sent" ? "ok" : "", stage === "sent" ? "✓ Alert sent" : "Alert sent")].join(" ");
  }
  async function restore() {                                        // the link is back: the queued ALERT goes out
    if (!pendingRestore) return;
    pendingRestore = null;
    const token = raceToken;
    showRestore("restoring"); await sleep(900); if (token !== raceToken) return;
    showRestore("sent");
    st("e", 5, "ok", "sent on restore", "good");
    q("#x-race-msg").innerHTML = `<b class="g">Link restored: the queued ALERT went out.</b> <span class="note">Cloud-only never saw this fire: nothing was decided during the outage.</span>`;
  }
  async function onPipelinePoll() {
    const online = OV && OV.uplink.online;
    if (pendingRestore && online) restore();
    lastOnline = online;
    if (LR && LR.eventId && OV && OV.last_alert && OV.last_alert.event_id === LR.eventId) liveDelivery(OV.last_alert);
    const lf = EC && EC.summary.last_fire;
    if (!lf || lf.id === seenFire) return;
    if (skipNext) { skipNext = false; seenFire = lf.id; return; }
    const first = seenFire === undefined;
    seenFire = lf.id;
    if (!first && active === "edge" && !LIVE.running && q("#x-rg") && !sending) race(Promise.resolve(fireFrom(lf)), lf.online, lf.cloud_upload_ms, RACE_REPLAY);
  }
  let sending = false, skipNext = false;
  async function sendFire() {
    if (sending) return;
    let photos = [];
    try { photos = (await api("/api/samples")).samples.filter(s => s.kind === "photo"); } catch (e) { /* none */ }
    if (!photos.length) { toast("No sample fire photos on this device"); return; }
    const pick = photos[Math.floor(Math.random() * photos.length)];
    const online = !!(OV && OV.uplink.online);
    const lf = EC && EC.summary.last_fire;
    const fd = new FormData();
    fd.append("sample_id", pick.id); fd.append("lat", "32.84"); fd.append("lon", "-116.53"); fd.append("name", "Pipeline check");
    sending = true;
    const fireP = api("/api/report", {method: "POST", body: fd}).then(r => ({
      detect_ms: r.result.detect_ms, vlm_ms: r.result.vlm_ms, severity: r.result.severity,
      decision: r.incident ? ((r.incident.escalation || {}).decision || "logged") : "logged"}));
    fireP.then(f => { if (f.vlm_ms != null) skipNext = true; }, e => toast(e.message))     // already animated live
      .finally(() => { sending = false; pollOverview(); refresh(); });
    race(fireP, online, lf ? lf.cloud_upload_ms : null, RACE_LIVE);
  }

  // ---------------------------------------------------------------- Mobile camera
  const LIVE = {running: false, inflight: false, media: null, timer: null, sent: 0, last: null,
                stream: "o" + Math.random().toString(36).slice(2, 10)};
  const capture = document.createElement("canvas");
  let camBuilt = false;
  function drawCamera() {
    const box = q("#x-camera");
    if (camBuilt) { if (LIVE.running) raceTo(q("#x-race-cam"), true); return; }
    camBuilt = true;
    box.innerHTML = `
      <div class="x-sec"><div class="note">A patrol unit's camera (laptop webcam, or an iPhone through Continuity Camera) runs through the <b>same
        detector, gate and VLM</b> as the towers, on this device. The video opens on the right; the pipeline and the team phone show here.</div>
        <div id="x-live-home"><div class="x-live" id="x-live"><video id="x-video" playsinline muted></video><canvas id="x-ov"></canvas>
          <div class="idle" id="x-idle">Camera off. Point it at smoke (a wildfire video on a tablet works).</div></div></div>
        <div class="actions"><select id="x-cam" class="btn" style="flex:1;min-width:0"><option value="">Default camera</option></select>
          <button class="btn ember" id="x-start">Start camera</button></div>
        <div class="x-status" id="x-stat"><span class="note">Not streaming.</span></div></div>
      <div id="x-race-cam"></div>
      <div class="x-sec" id="x-phone-sec"><h4>Team phone (ntfy) ${help("The real ALERT push: when the edge decides ALERT and the link is up, the report goes to the team's phone through ntfy. With no link it waits in the outbox on this device and is pushed when the link returns.")}</h4>
        <div class="x-phone-row"><div class="x-phone"><div class="x-ph-notch"></div><div class="x-ph-screen" id="x-ph"></div></div><div class="note" id="x-ph-note"></div></div></div>
      <div class="x-sec" id="x-ev"></div>`;
    q("#x-start").onclick = () => LIVE.running ? stopCam() : startCam();
    q("#x-cam").onchange = async () => { if (LIVE.running) { try { await openCam(); } catch (e) { stopCam(camErr(e)); } } };
    listCams();
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) navigator.mediaDevices.addEventListener("devicechange", () => listCams());
    phone("idle");
    paintCam(null);
  }
  function stage() {                                   // the big camera view over the map
    let el = q("#x-stage");
    if (el) return el;
    el = document.createElement("div");
    el.className = "x-stage"; el.id = "x-stage"; el.hidden = true;
    el.innerHTML = `<div class="x-stage-h"><span class="x-live-dot"></span><b>Mobile unit 1</b><span class="note" id="x-stage-stat">live</span>
        <span class="sp"></span><button class="btn" id="x-pip">Minimize</button><button class="btn" id="x-stop2">Stop camera</button></div>
      <div class="x-stage-body" id="x-stage-body"></div><div class="x-stage-alert" id="x-stage-alert" hidden></div>`;
    q("#mapwrap").appendChild(el);
    q("#x-pip").onclick = () => { const pip = el.classList.toggle("pip"); q("#x-pip").textContent = pip ? "Expand" : "Minimize"; };
    q("#x-stop2").onclick = () => stopCam();
    return el;
  }
  function phone(state, info = {}) {
    const scr = q("#x-ph"), note = q("#x-ph-note");
    if (!scr) return;
    const t = new Date(), now = `${t.getHours() % 12 || 12}:${String(t.getMinutes()).padStart(2, "0")}`;
    const lock = `<div class="x-ph-clock">${now}</div>`;
    if (state === "sent") {
      scr.innerHTML = lock + `<div class="x-ntfy buzz"><div class="x-ntfy-h"><span class="x-ntfy-ic">ntfy</span><span>ntfy · now</span></div>
        <b>🔥 WILDFIRE ALERT — ${esc(info.where || "Mobile unit 1")}</b><div>${esc(fmt.words(info.source || "wildland"))} smoke/fire${info.lat != null ? ` at ${info.lat.toFixed(3)}, ${info.lon.toFixed(3)}` : ""}. ${esc(info.desc || "")}</div></div>`;
      note.innerHTML = `<b style="color:var(--ok)">Delivered to the team's phone</b> by ntfy${info.host ? ` (${esc(info.host)})` : ""}.`;
    } else if (state === "queued") {
      scr.innerHTML = lock + `<div class="x-ph-empty">No signal from the tower yet</div>`;
      note.innerHTML = `<b style="color:#c05600">Link down:</b> the ALERT waits in the outbox on this device and is pushed the moment the link returns.`;
    } else if (state === "sending") {
      scr.innerHTML = lock + `<div class="x-ph-empty">…</div>`;
      note.textContent = "Sending the ALERT to the team's phone…";
    } else if (state === "off") {
      scr.innerHTML = lock + `<div class="x-ph-empty">Phone push is off on this node</div>`;
      note.innerHTML = `The ALERT was delivered to dispatch only: phone push needs <code>NTFY_TOPIC_URL</code> on the device (it is off in rehearsal mode).`;
    } else {
      scr.innerHTML = lock + `<div class="x-ph-empty">No alerts</div>`;
      note.textContent = "Waiting for an ALERT from the mobile unit.";
    }
  }

  // ---- the pipeline, driven by the live camera's real frames
  let LR = null;          // {token, online, stage: gate|vlm|decided|alert, eventId, vlmT, sev}
  function liveWaiting() {
    raceToken++; resetRace(); LR = null; pendingRestore = null;
    q("#x-race-msg").innerHTML = `<span class="note">Watching. Point the camera at smoke: the pipeline starts on the first frame with smoke.</span>`;
    stage().querySelector("#x-stage-alert").hidden = true;
  }
  function liveBegin(d) {
    const online = d.online !== false;
    raceToken++; resetRace(); pendingRestore = null;
    LR = {token: raceToken, online, stage: "gate", eventId: null, sev: null};
    RACE.classList.remove("pop"); void RACE.offsetWidth; RACE.classList.add("pop");
    q("#x-nb").classList.toggle("down", !online);
    q("#x-nb span").textContent = online ? "network" : "no network";
    st("e", 0, "ok", "frame"); st("e", 1, "run", "smoke 1/3"); st("e", 2, "skip", "0 B · stays here");
    st("c", 0, "ok", "frame"); st("c", 1, "na", "no model on site");
    q("#x-race-msg").innerHTML = `<span class="note">Smoke in view${online ? "" : " — <b>no network</b>"}. The edge confirms it over 3 frames before calling the VLM.</span>`;
    const token = raceToken, alive = () => LR && LR.token === token;
    (async () => {                                   // cloud-only: every frame goes up and through a hosted VLM
      st("c", 2, "run", online ? "uploading" : "");
      if (!online) {
        await sleep(700); if (!alive()) return;
        st("c", 2, "blocked", "no network", "bad");
        for (const i of [3, 4, 5]) st("c", i, "dash", "—");
        q("#x-lane-c").classList.add("stopped");
        q("#x-rg").insertAdjacentHTML("beforeend", `<div class="x-blind" id="x-blind" style="grid-row:3;grid-column:5/8">BLIND DURING OUTAGE</div>`);
        q("#x-t-c").textContent = "no decision";
        return;
      }
      const lf = EC && EC.summary.last_fire, up = lf && lf.cloud_upload_ms != null ? lf.cloud_upload_ms : 1500;
      await sleep(Math.max(400, up)); if (!alive()) return;
      st("c", 2, "ok", `${lf ? fmt.b(lf.cloud_bytes) + " · " : ""}${fmt.ms(up)}`);
      st("c", 3, "run", "in the cloud");
      const vlm = lf && lf.vlm_ms ? lf.vlm_ms : 5400;
      await sleep(vlm); if (!alive()) return;
      st("c", 3, "ok", `cloud · ${fmt.ms(vlm)}`);
      while (alive() && !LR.sev) await sleep(250);     // same model, same verdict as the edge
      if (!alive()) return;
      st("c", 4, "ok", LR.sev);
      st("c", 5, LR.sev === "ALERT" ? "ok" : "run", LR.sev === "ALERT" ? "sent after reply" : "re-checking");
      q("#x-t-c").textContent = `decided in ${fmt.ms(up + vlm)}`;
      LR.cloudMs = up + vlm;
    })();
  }
  function liveVlmStart() {                           // the frame now in flight completes the gate: the VLM runs
    if (!LR || LR.stage !== "gate") return;
    LR.stage = "vlm"; LR.vlmT = performance.now();
    st("e", 1, "ok", "smoke 3/3", "good");
    st("e", 3, "run", "on device");
    q("#x-race-msg").innerHTML = `<span class="note">Smoke held for 3 frames: the VLM on this device is classifying it…</span>`;
  }
  function liveDecided(sev, source, vlmMs) {
    if (!LR || LR.sev) return;
    LR.sev = sev; LR.stage = "decided"; LR.source = source;
    st("e", 1, "ok", "smoke 3/3", "good");
    st("e", 3, "ok", `local · ${fmt.ms(vlmMs)}`, "good");
    st("e", 4, "ok", `${sev}${source ? " · " + fmt.words(source) : ""}`, "good");
    if (sev !== "ALERT") {
      st("e", 5, "run", "re-checking growth");
      q("#x-race-msg").innerHTML = `<span class="note"><b>${esc(sev)}</b> (${esc(fmt.words(source))}): the edge keeps watching and re-checks whether it grows.</span>`;
      stageAlert(sev, source, "watching for growth");
    }
  }
  function liveAlert(ev) {
    if (!LR) return;
    if (!LR.sev || LR.sev !== "ALERT") { LR.sev = null; liveDecided("ALERT", ev.source_type, LIVE.last && LIVE.last.timings ? LIVE.last.timings.vlm_ms : null); }
    LR.stage = "alert"; LR.eventId = ev.id; LR.desc = ev.description;
    liveDelivery(ev.delivery ? {...ev.delivery, event_id: ev.id} : {state: "sending"});
    q("#x-lane-e").classList.add("done");
    q("#x-t-e").textContent = "ALERT raised";
  }
  function liveDelivery(dv) {                          // from the frame response, then from the overview poll
    if (!LR || !dv) return;
    const where = (dv.incident && dv.incident.name) || "Mobile unit 1";
    const cfg = OV && OV.last_alert;
    const info = {where, source: LR.source, desc: LR.desc || "", host: null};
    if (dv.phone === "expired") { st("e", 5, "blocked", "expired", "bad"); return; }
    if (dv.state === "queued") {
      if (LR.delivered !== "queued") { LR.delivered = "queued"; st("e", 5, "queued", "queued locally", "amber"); pendingRestore = {t: Date.now()}; showRestore("queued"); phone("queued"); }
      stageAlert("ALERT", LR.source, "queued on the edge: no link");
    } else if (dv.state === "sent" || dv.delivered) {
      if (LR.delivered === "sent") return;
      LR.delivered = "sent";
      st("e", 5, "ok", dv.phone === "sent" ? "sent to phone" : "sent to dispatch", "good");
      if (q("#x-restore") && !q("#x-restore").hidden) showRestore("sent");
      if (dv.phone === "sent") phone("sent", info); else if (dv.phone === "off") phone("off");
      stageAlert("ALERT", LR.source, dv.phone === "sent" ? "sent to the team's phone" : "reported to dispatch");
      q("#x-race-msg").innerHTML = `<b class="g">ALERT decided on the edge and ${dv.phone === "sent" ? "pushed to the team's phone" : "reported to dispatch"}.</b>` +
        (LR.online ? ` <span class="note">The frame itself never left the device.</span>` : ` <span class="note">Cloud-only was blind during the outage.</span>`);
    } else { st("e", 5, "run", "sending"); phone("sending"); }
  }
  function stageAlert(sev, source, what) {
    const a = q("#x-stage-alert"); if (!a) return;
    a.hidden = false;
    a.className = "x-stage-alert " + sev;
    a.innerHTML = `<b>${esc(sev)}</b> ${esc(fmt.words(source))} · ${esc(what)}`;
  }
  function liveFrame(d) {                              // advance the pipeline from one live-frame response
    const g = d.gate || {streak: 0, needed: 3}, e = d.event, lat = d.latched || {};
    // a new plume after the last one was decided and its latch cleared: a fresh run through both pipelines
    if (LR && (LR.stage === "decided" || LR.stage === "alert") && !e && !lat.active && !d.new_alert && g.streak > 0) LR = null;
    if (!LR && (g.streak > 0 || (e && e.severity))) liveBegin(d);
    if (!LR) return;
    if (LR.stage === "gate") {
      if (g.streak > 0) st("e", 1, "run", `smoke ${g.streak}/${g.needed}`);
      else if (!e && !lat.active && !d.new_alert) { liveWaiting(); return; }     // the smoke went away before the gate
    }
    const done = d.new_alert || (d.new_events || []).find(x => x.severity) || (e && e.severity ? e : null);
    if (done && !LR.sev) liveDecided(done.severity, done.source_type, d.timings && d.timings.vlm_ms);
    if (d.new_alert) liveAlert(d.new_alert);
  }

  async function listCams(keep) {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    let devs = [];
    try { devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput" && d.deviceId); } catch (e) { return; }
    const sel = q("#x-cam"), want = keep || sel.value;
    sel.innerHTML = `<option value="">Default camera</option>` + devs.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || "Camera " + (i + 1))}</option>`).join("");
    if (want && [...sel.options].some(o => o.value === want)) sel.value = want;
  }
  function camErr(e) {
    const n = e && e.name;
    if (n === "NotAllowedError" || n === "SecurityError") return "Camera permission denied: allow it for this page (and in macOS Privacy settings).";
    if (n === "NotFoundError" || n === "OverconstrainedError") return "No such camera: pick another one.";
    if (n === "NotReadableError") return "The camera is in use by another app.";
    return "Could not open the camera: " + ((e && e.message) || n);
  }
  async function openCam() {
    const id = q("#x-cam").value, v = {width: {ideal: 1280}, height: {ideal: 720}};
    if (id) v.deviceId = {exact: id};
    const media = await navigator.mediaDevices.getUserMedia({video: v, audio: false});
    if (LIVE.media) LIVE.media.getTracks().forEach(t => t.stop());
    LIVE.media = media;
    const video = q("#x-video"); video.srcObject = media; await video.play().catch(() => {});
    q("#x-idle").hidden = true;
    const tr = media.getVideoTracks()[0];
    await listCams(tr && tr.getSettings ? tr.getSettings().deviceId : "");
    return tr ? tr.label : "";
  }
  async function startCam() {
    if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
      q("#x-stat").innerHTML = `<span class="note">${window.isSecureContext === false ? "Browsers only allow the camera on localhost: open this console through the SSH tunnel or on the device itself." : "This browser has no camera API."}</span>`;
      return;
    }
    try {
      q("#x-stat").innerHTML = `<span class="note">Opening the camera…</span>`;
      await openCam();
      LIVE.running = true;
      q("#x-start").textContent = "Stop camera"; q("#x-start").className = "btn";
      const stg = stage();
      q("#x-stage-body").appendChild(q("#x-live"));
      stg.hidden = false; stg.classList.remove("pip"); q("#x-pip").textContent = "Minimize";
      raceTo(q("#x-race-cam"), true);
      liveWaiting(); phone("idle");
      await post("/lab/api/live/reset", {}).catch(() => {});
      tick();
    } catch (e) { stopCam(camErr(e)); }
  }
  function stopCam(msg = "Stopped.") {
    LIVE.running = false; clearTimeout(LIVE.timer);
    if (LIVE.media) LIVE.media.getTracks().forEach(t => t.stop());
    LIVE.media = null;
    const v = q("#x-video"); if (v) v.srcObject = null;
    const idle = q("#x-idle"); if (idle) idle.hidden = false;
    const b = q("#x-start"); if (b) { b.textContent = "Start camera"; b.className = "btn ember"; }
    const home = q("#x-live-home"), lv = q("#x-live");
    if (home && lv && lv.parentNode !== home) home.appendChild(lv);
    const stg = q("#x-stage"); if (stg) stg.hidden = true;
    LR = null; raceToken++;
    if (q("#x-race-home")) raceTo(q("#x-race-home"), false);
    paintCam(null);
    const st = q("#x-stat"); if (st) st.innerHTML = `<span class="note">${esc(msg)}</span>`;
  }
  async function tick() {
    if (!LIVE.running || LIVE.inflight) return;
    const t0 = performance.now(), v = q("#x-video");
    LIVE.inflight = true;
    try {
      if (!v.videoWidth) return;
      const s = Math.min(1, 1280 / Math.max(v.videoWidth, v.videoHeight));
      capture.width = Math.round(v.videoWidth * s); capture.height = Math.round(v.videoHeight * s);
      capture.getContext("2d").drawImage(v, 0, 0, capture.width, capture.height);
      const blob = await new Promise(r => capture.toBlob(r, "image/jpeg", 0.85));
      const prev = LIVE.last, pg = prev && prev.gate;
      if (LR && pg && !prev.event && !(prev.latched || {}).active && pg.streak === pg.needed - 1) liveVlmStart();
      const r = await fetch(`/lab/api/live/frame?stream=${LIVE.stream}`, {method: "POST", headers: {"Content-Type": "image/jpeg"}, body: blob, cache: "no-store"});
      if (r.status === 202) return;
      const d = await r.json().catch(() => ({}));
      if (!r.ok) { q("#x-stat").innerHTML = `<span class="note">${esc(typeof d.detail === "string" ? d.detail : "Frame rejected")}</span>`; return; }
      LIVE.sent++; LIVE.last = d;
      try { liveFrame(d); } catch (err) { console.error(err); }
      const ss = q("#x-stage-stat"); if (ss) ss.textContent = `live · ${fmt.ms(d.timings.total_ms)} per frame · ${d.online === false ? "no link: deciding on the device" : "online"}`;
      if (d.new_alert) { toast("ALERT from the mobile unit: see Incidents"); refresh(); }
      else if ((d.new_events || []).length || (d.event && d.event.severity)) refresh();
      paintCam(d);
    } catch (e) { q("#x-stat").innerHTML = `<span class="note">The device is not answering: ${esc(e.message)}</span>`; }
    finally { LIVE.inflight = false; if (LIVE.running) LIVE.timer = setTimeout(tick, Math.max(0, 1500 - (performance.now() - t0))); }
  }
  function paintCam(d) {
    const cv = q("#x-ov");
    if (cv) {
      const cw = cv.clientWidth, ch = cv.clientHeight, dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
      const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, cw, ch);
      if (d && d.frame && LIVE.running) {
        const s = Math.min(cw / d.frame.width, ch / d.frame.height), ox = (cw - d.frame.width * s) / 2, oy = (ch - d.frame.height * s) / 2;
        g.font = "600 12px system-ui, sans-serif";
        for (const det of (d.detections || []).slice(0, 20)) {
          const [x1, y1, x2, y2] = det.box, x = ox + x1 * s, y = oy + y1 * s;
          g.lineWidth = 2.5; g.strokeStyle = det.above_gate ? "#d9480f" : "rgba(255,255,255,.5)"; g.setLineDash(det.above_gate ? [] : [5, 4]);
          g.strokeRect(x, y, Math.max(1, (x2 - x1) * s), Math.max(1, (y2 - y1) * s)); g.setLineDash([]);
          const lab = `${det.cls} ${det.conf.toFixed(2)}`, ty = y > 18 ? y - 18 : y;
          g.fillStyle = det.above_gate ? "#d9480f" : "rgba(0,0,0,.6)"; g.fillRect(x, ty, g.measureText(lab).width + 10, 18);
          g.fillStyle = "#fff"; g.fillText(lab, x + 5, ty + 13);
        }
      }
    }
    const st = q("#x-stat"), ev = q("#x-ev");
    if (!st || !d) { if (ev) ev.innerHTML = `<h4>What happens</h4><div class="note">1. The detector checks every frame (~40 ms).<br>2. Smoke in 3 frames in a row opens an event.<br>
      3. The VLM classifies it once (~5 s) and severity rules decide.<br>4. MONITOR: re-checked for growth. ALERT: queued in the outbox and pushed to dispatch; with no link it waits and goes out when the link returns.</div>`; return; }
    const g = d.gate || {streak: 0, needed: 3}, e = d.event, lat = d.latched || {};
    const dots = `<span class="x-gate">${Array.from({length: g.needed}, (_, i) => `<i class="${i < g.streak || e || lat.active ? "on" : ""}"></i>`).join("")}</span>`;
    const status = e ? (e.severity ? `<span class="sev ${e.severity}">${e.severity}</span> ${esc(fmt.words(e.source_type))}` : "Classifying…") : lat.active ? '<span class="sev ALERT">ALERT</span> sent for this fire' : g.streak ? "Smoke seen" : "Watching";
    st.innerHTML = `${dots}<span>${status}</span><span class="note">${fmt.ms(d.timings.total_ms)} · ${LIVE.sent} frames${d.online === false ? " · no link: deciding on the device" : ""}</span>`;
    const last = (d.events || [])[0];
    ev.innerHTML = last ? `<h4>Last decision</h4><div class="x-status"><span class="sev ${esc(last.severity || "")}">${esc(last.severity || "…")}</span>
      <b>${esc(fmt.words(last.source_type))}</b>${last.delivery ? `<span class="tag ${last.delivery.state === "sent" ? "sent" : "queued"}">${last.delivery.state === "sent" ? "Reported to dispatch" : "Alert waiting (link down)"}</span>` : ""}</div>
      <div class="note">${esc(last.description || "")}</div><div class="actions"><button class="btn" onclick="showTab('incidents')">Open incidents</button></div>`
      : ev.innerHTML;
  }

  // ---------------------------------------------------------------- Models
  let SAMPLES_LAB = null;
  async function drawModels() {
    let m;
    try { m = await api("/api/models"); } catch (e) { return; }
    if (!SAMPLES_LAB) { try { SAMPLES_LAB = (await api("/lab/api/samples")).samples || []; } catch (e) { SAMPLES_LAB = []; } }
    const d = m.detector, v = m.vlm, dr = d.results || {}, vr = v.results || {};
    const vc = (k, key) => vr[k] && vr[k][key] != null ? fmt.pct(vr[k][key]) : "–";
    const perSrc = Object.keys((vr.context || {}).per_source || {});
    q("#x-models").innerHTML = `
      <div class="x-sec"><h4>Deployed on this device</h4><div class="x-kpis">
        ${kpi(d.headline.value != null ? d.headline.value.toFixed(3) : "–", "Detector mAP50 (tower val)", `${esc(d.name)} · ${esc(d.version)} · ${dr.tower && dr.tower.ms_per_image ? fmt.ms(dr.tower.ms_per_image) + " per frame" : ""}`, "edge")}
        ${kpi(vr[v.active] ? fmt.pct(vr[v.active].source_type_acc) : "–", "VLM source-type agreement", `${esc(v.name)} · adapter <b>${esc(v.active)}</b>`, "edge")}</div>
        ${v.options.length > 1 ? `<div class="actions" id="x-adapt"><span class="note" style="align-self:center">VLM adapter for every camera:</span>${v.options.map(o => `<button class="chip ${o === v.active ? "on" : ""}" data-a="${esc(o)}">${esc(o)}</button>`).join("")}</div>` : ""}</div>
      <div class="x-sec"><h4>Before → after fine-tuning ${help("Detector: YOLO-World zero-shot vs YOLO11s fine-tuned (joint D-Fire + tower). VLM: Qwen2.5-VL-7B base vs LoRA distilled from the 32B teacher, on 500 held-out crops.")}</h4>
        <table class="x-t"><tr><th></th><th>Before</th><th>LoRA v1</th><th>LoRA v2</th></tr>
          <tr><td>VLM source type</td><td>${vc("before", "source_type_acc")}</td><td class="win">${vc("context", "source_type_acc")}</td><td>${vc("context_v2", "source_type_acc")}</td></tr>
          <tr><td>Danger / benign / look-alike</td><td>${vc("before", "group_acc")}</td><td class="win">${vc("context", "group_acc")}</td><td>${vc("context_v2", "group_acc")}</td></tr>
          <tr><td>Tokens per call</td><td>${vr.before ? fmt.n(vr.before.tokens_per_call) : "–"}</td><td>${vr.context ? fmt.n(vr.context.tokens_per_call) : "–"}</td><td>${vr.context_v2 ? fmt.n(vr.context_v2.tokens_per_call) : "–"}</td></tr></table>
        <table class="x-t"><tr><th>Detector (mAP50)</th><th>Before</th><th>After</th></tr>
          <tr><td>D-Fire</td><td>${dr.before ? dr.before.map50.toFixed(3) : "–"}</td><td class="win">${dr.dfire ? dr.dfire.map50.toFixed(3) : "–"}</td></tr>
          <tr><td>Tower (HPWREN)</td><td>–</td><td class="win">${dr.tower ? dr.tower.map50.toFixed(3) : "–"}</td></tr></table>
        ${perSrc.length ? `<details><summary>Per source type (correct / crops)</summary><table class="x-t"><tr><th></th><th>Before</th><th>v1</th><th>v2</th></tr>
          ${perSrc.map(s => `<tr><td>${esc(fmt.words(s))}</td>${["before", "context", "context_v2"].map(k => { const x = ((vr[k] || {}).per_source || {})[s]; return `<td>${x ? `${x.correct}/${x.n}` : "–"}</td>`; }).join("")}</tr>`).join("")}</table></details>` : ""}</div>
      <div class="x-sec"><h4>Compare on an image</h4>
        <div class="two"><div class="field"><span>Sample</span><select id="x-smp"><option value="">Choose a sample…</option>${SAMPLES_LAB.slice(0, 200).map(s => `<option value="${esc(s.id)}">${esc(s.group || "")} · ${esc(s.name)}</option>`).join("")}</select></div>
          <div class="field"><span>…or upload</span><input type="file" id="x-up" accept="image/*"></div></div>
        <div class="x-cmp" id="x-cmp"></div></div>`;
    const ad = q("#x-adapt");
    if (ad) ad.onclick = async e => {
      const b = e.target.closest("[data-a]"); if (!b || b.classList.contains("on")) return;
      try { await post("/api/models/vlm", {model: b.dataset.a}); toast(`Every camera now uses ${b.dataset.a}`); drawModels(); } catch (err) { toast(err.message); }
    };
    q("#x-smp").onchange = e => e.target.value && compare({sample_id: e.target.value}, `/lab/api/sample/${encodeURIComponent(e.target.value)}`);
    q("#x-up").onchange = e => { const f = e.target.files[0]; if (f) compare({image: f}, URL.createObjectURL(f)); };
  }
  async function compare(input, src) {
    const box = q("#x-cmp");
    box.innerHTML = ["Before", "After"].map(l => `<div><div class="im"><img alt="" src="${src}"></div><div class="cap note">${l}: running on the device…</div></div>`).join("");
    const fd = new FormData();
    if (input.image) fd.append("image", input.image); else fd.append("sample_id", input.sample_id);
    fd.append("pipelines", "before,after");
    let r;
    try { r = await api("/lab/api/analyze", {method: "POST", body: fd}); } catch (e) { box.innerHTML = `<div class="note">${esc(e.message)}</div>`; return; }
    const W = r.image.width, H = r.image.height;
    box.innerHTML = r.results.map(p => {
      if (p.error) return `<div class="note">${esc(p.label)}: ${esc(p.error)}</div>`;
      const boxes = (p.detections || []).slice(0, 20).map(d => { const [x1, y1, x2, y2] = d.box;
        return `<rect x="${x1}" y="${y1}" width="${Math.max(1, x2 - x1)}" height="${Math.max(1, y2 - y1)}" fill="none" stroke="${d.above_gate ? "#d9480f" : "#fff"}" stroke-width="${Math.max(2, W / 250)}"${d.above_gate ? "" : ` stroke-dasharray="${W / 90}"`}/>`; }).join("");
      const c = p.context || {};
      return `<div><div class="im"><img alt="" src="${src}"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">${boxes}</svg></div>
        <div class="cap"><b>${p.pipeline === "after" ? "After" : "Before"}</b> ${p.severity ? `<span class="sev ${esc(p.severity)}">${esc(p.severity)}</span>` : ""} ${esc(fmt.words(c.source_type || (p.severity === "IGNORE" ? "no smoke" : "")))}</div>
        <div class="note">${esc(p.detector_label)} + ${esc(p.vlm_label)} · ${fmt.ms((p.detect_ms || 0) + (p.vlm_ms || 0))}${p.tokens ? ` · ${fmt.n(p.tokens)} tokens` : ""}</div>
        ${c.description ? `<div class="note">${esc(c.description)}</div>` : ""}</div>`;
    }).join("");
  }

  // ---------------------------------------------------------------- System
  async function drawSystem() {
    let s, a;
    try { [s, a] = await Promise.all([api("/api/system"), api("/api/alerts")]); } catch (e) { return; }
    const st = s.stats || {}, mon = s.monitor || {}, models = mon.models || [];
    const up = h => h < 3600 ? `${Math.round(h / 60)} min` : `${(h / 3600).toFixed(1)} h`;
    const mem = st.mem_total_gb ? 100 * st.mem_used_gb / st.mem_total_gb : null;
    q("#x-system").innerHTML = `
      <div class="x-sec"><h4>${esc(s.node)}<span class="sp"></span><span class="note">up ${up(s.uptime_s)}</span></h4>
        <div class="x-health">${s.health.map(h => `<div class="x-h"><span class="d ${h.state}"></span><div><b>${esc(h.label)}</b><small>${esc(h.note)}</small></div></div>`).join("")}</div></div>
      <div class="x-sec"><h4>Resources</h4><div class="x-kpis">
        ${kpi(st.gpu_util_pct != null ? Math.round(st.gpu_util_pct) + "%" : "–", "GPU")}
        ${kpi(st.mem_used_gb != null ? `${st.mem_used_gb.toFixed(0)} / ${st.mem_total_gb.toFixed(0)} GB` : "–", "Unified memory")}</div>
        ${mem != null ? `<div class="x-meter"><div style="width:${mem.toFixed(0)}%"></div></div>` : ""}</div>
      <div class="x-sec"><h4>Live model metrics (vLLM on this device) ${help("Read from the model server's own metrics socket every few seconds: requests, tokens and latency per served model. No internet involved.")}</h4>
        ${models.length ? `<table class="x-t"><tr><th>Model</th><th>Requests</th><th>Tokens in / out</th><th>Latency p50 / p95</th><th>Tok/s</th></tr>
          ${models.map(x => `<tr><td>${esc(x.label)}</td><td>${fmt.k(x.requests)}</td><td>${fmt.k(x.prompt_tokens)} / ${fmt.k(x.generation_tokens)}</td>
            <td>${x.latency_p50_s != null ? x.latency_p50_s.toFixed(1) + " s" : "–"} / ${x.latency_p95_s != null ? x.latency_p95_s.toFixed(1) + " s" : "–"}</td>
            <td>${fmt.n((x.prompt_tok_per_s || 0) + (x.gen_tok_per_s || 0), 0)}</td></tr>`).join("")}</table>` : `<div class="note">Model server metrics not available.</div>`}
        <div class="note">Served models: ${s.served_models.map(esc).join(", ") || "none"}</div></div>
      <div class="x-sec"><h4>Alert delivery<span class="sp"></span><button class="btn" id="x-test" ${a.phone.configured && a.online ? "" : "disabled"}>Send test push</button></h4>
        <div class="x-kpis">${kpi(fmt.n(a.delivered), "delivered", "", "good")}${kpi(fmt.n(a.queued), "waiting for the link", "", a.queued ? "hot" : "")}</div>
        <div class="note">Phone push: ${a.phone.configured ? `on (${esc(a.phone.host)})` : "off (set NTFY_TOPIC_URL on the device)"} · Dispatch endpoint: ${a.dispatch_configured ? "on" : "phone only"}</div>
        <div class="x-list">${a.alerts.slice(0, 12).map(x => `<div class="it"><span class="sev ${esc(x.severity || "ALERT")}">${esc(x.severity || "ALERT")}</span>
          <span>${esc(x.incident ? x.incident.name : x.tower_name)} <span class="note">· ${esc(x.source)} · ${x.detected_at ? new Date(x.detected_at).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : ""}</span></span>
          <span class="tag ${x.state === "sent" ? "sent" : "queued"}">${x.phone === "expired" ? "expired" : x.state === "sent" ? "delivered" : "waiting"}</span></div>`).join("") || '<div class="note">No ALERTs yet. Only ALERTs leave the device.</div>'}</div></div>`;
    const t = q("#x-test");
    if (t) t.onclick = async () => { try { const r = await post("/api/alerts/test", {}); toast(r.status === "sent" ? "Test push sent" : "Test suppressed (one per 30 s)"); } catch (e) { toast(e.message); } };
  }
})();

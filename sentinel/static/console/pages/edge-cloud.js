// Edge vs Cloud: for the frames this node really processed, what a cloud-only design would have uploaded,
// spent and waited, and what it could not decide at all while the link was down.
import {get, poll, post} from "../api.js";
import {$, bytes, compact, empty, esc, icon, info, isoTime, kpi, more, ms, num, pct, pill, seg, times, toast, usd} from "../ui.js";

const PROFILES = [["fiber", "Fiber"], ["lte", "LTE"], ["rural_cellular", "Rural cell"], ["satellite_geo", "Satellite"], ["outage", "Down"]];
let profile = "lte";
let fleetArgs = {towers: 20, fps: 1, days: 30};

export function mount(el, ctx) {
  el.innerHTML = `<div class="page">
    <div class="row wrap" style="margin-bottom:18px">
      <div id="prof"></div><span class="grow"></span>
      <div class="legend"><span class="e">Edge (measured)</span><span class="c">Cloud-only (modeled)</span></div>
    </div>
    <div class="grid g4" id="kpis"></div>
    <div class="card" style="margin-top:14px"><h3>Same frames, two architectures<span class="spacer"></span><span id="tag"></span></h3><div class="versus" id="vs"></div></div>
    <div class="card" style="margin-top:14px;padding-top:4px;padding-bottom:4px" id="more"></div>
  </div>`;
  let d = null, alerts = [], fleet = null;

  function drawProfiles() {
    $("#prof", el).innerHTML = seg(PROFILES, profile);
    $("#prof", el).onclick = e => { const b = e.target.closest("button"); if (!b) return; profile = b.dataset.v; drawProfiles(); refresh(); };
  }

  function bar(kind, v, max, label) {
    const w = v > 0 && max > 0 ? Math.max(0.8, 100 * v / max) : 0;       // linear: the gap is the point
    return `<div class="bar ${kind}"><div class="track"><div class="fill" style="width:${w.toFixed(1)}%"></div></div><div class="val">${label}</div></div>`;
  }
  function row(label, tip, e, c, fmt, cLabel) {
    const max = Math.max(e || 0, c || 0);
    return `<div class="vs-row"><div class="lbl row">${esc(label)}${tip ? info(tip) : ""}</div><div class="bars">
      ${bar("edge", e || 0, max, fmt(e))}${bar("cloud", c || 0, max, cLabel ?? fmt(c))}</div></div>`;
  }

  function draw() {
    const s = d.summary, e = s.edge, c = s.cloud, sv = s.savings, m = s.method;
    $("#tag", el).innerHTML = d.measured.enabled && d.measured.n ? pill(`${d.measured.n} measured samples`, "ok") : pill("Cloud modeled", "tag");
    const blind = s.cloud_blind_frames;
    const cloudTime = s.link_down ? "No decision" : ms(c.decide_ms_p50);
    $("#kpis", el).innerHTML = [
      kpi(sv.bytes_x ? times(sv.bytes_x) : c.bytes_up ? bytes(sv.bytes) : "–", sv.bytes_x ? "less data sent upstream" : "not uploaded (edge sent nothing)",
        {cls: "edge", tip: m.cloud_bytes + " " + m.edge_bytes, sub: `Edge ${bytes(e.bytes_up)} · cloud-only ${bytes(c.bytes_up)}`}),
      s.prices_set ? kpi(usd(sv.usd), "API + data cost avoided", {cls: "good", tip: m.cloud_usd + " " + m.edge_usd, sub: `Cloud-only ${usd(c.usd)} for these frames`})
        : kpi("Set prices", "to see cost avoided", {tip: "Enter the provider's published token and data prices under Assumptions below.", sub: `Tokens avoided: ${pct(sv.tokens_pct)}`}),
      kpi(e.decide_ms_p50 != null ? ms(e.decide_ms_p50) : "–", "time to classify a plume", {tip: m.edge_time + " " + m.cloud_time, sub: `Cloud-only on this link: ${cloudTime}`}),
      kpi(num(s.edge_offline_decisions), "frames decided offline", {cls: s.edge_offline_decisions ? "hot" : "", tip: m.blind, sub: blind ? `Cloud-only was blind for ${num(blind)}` : "Use Test outage under System"}),
    ].join("");
    $("#vs", el).innerHTML = e.frames ? [
      row("Data uplinked", m.edge_bytes, e.bytes_up, c.bytes_up, bytes),
      row("VLM calls", m.edge_tokens, e.vlm_calls, c.vlm_calls, v => num(v)),
      row("Tokens", m.cloud_tokens, e.tokens, c.tokens, v => compact(v)),
      row("Time to decision", m.cloud_time, e.decide_ms_p50, s.link_down ? 0 : c.decide_ms_p50, ms, s.link_down ? "No decision" : undefined),
      s.prices_set ? row("Cost", m.cloud_usd, e.usd, c.usd, usd) : "",
    ].join("") : empty("No frames yet", "Run a drill or start the mobile camera: every frame the node analyses is compared here.",
      `<button class="btn" type="button" id="go-cam">${icon("camera")}Open cameras</button>`);
    const go = $("#go-cam", el); if (go) go.onclick = () => ctx.go("cameras");
    drawMore();
  }

  function outageTimeline() {
    const a = alerts.find(x => x.sent_at && x.detected_at && x.sent_at - Date.parse(x.detected_at) / 1000 > 3);
    if (!a) return `<p class="muted small" style="margin:0">Turn on <b>Test outage</b> (System), raise an ALERT, then restore the link: the timeline shows how long the alert waited while the edge had already decided.</p>`;
    const t0 = Date.parse(a.detected_at) / 1000, t1 = a.sent_at, span = t1 - t0;
    return `<div class="small muted" style="margin-bottom:6px">${esc(a.incident ? a.incident.name : a.tower_name)} · ALERT at ${isoTime(a.detected_at)}</div>
      <div class="tl"><div class="axis"></div><div class="band" style="left:4%;width:88%"></div>
        <div class="m" style="left:4%;background:var(--edge)"><span>Edge decided · 0 s</span></div>
        <div class="m" style="left:92%;background:var(--ok)"><span>Delivered · +${span < 120 ? span.toFixed(0) + " s" : (span / 60).toFixed(1) + " min"}</span></div></div>
      <div class="small muted" style="margin-top:22px">Cloud-only: nothing decided during the shaded outage; the camera frames from it would still have to be uploaded.</div>`;
  }

  function benchTable() {
    const b = d.benchmarks, edge = b.edge, cloud = b.cloud;
    const rows = [["Precision", "precision", pct100], ["Recall", "recall", pct100], ["False alarms", "false_alarms", v => num(v)],
      ["Frames per VLM call", "frames_per_vlm_call", v => num(v, 1)], ["Time to decision (p50)", "time_to_decision_s_p50", v => v == null ? "–" : `${num(v, 1)} s`]];
    return edge ? `<table class="t"><thead><tr><th></th><th class="r">Edge cascade</th><th class="r">Cloud-only</th></tr></thead><tbody>
      ${rows.map(([l, k, f]) => `<tr><td>${l}</td><td class="r num">${f(edge[k])}</td><td class="r num">${cloud ? f(cloud[k]) : "<span class='faint'>not measured</span>"}</td></tr>`).join("")}
      </tbody></table><div class="small faint" style="margin-top:8px">${edge.n_clips || "–"} labelled tower clips (results/bench_after.json).${cloud ? "" : " The cloud column needs a provider key."}</div>`
      : `<p class="muted small">No benchmark results on this node.</p>`;
  }
  function measured() {
    const m = d.measured;
    return `<div class="grid g3">${kpi(m.n, "frames sent to " + esc(m.host))}${kpi(pct(m.agreement_pct), "same source type as the edge")}${kpi(ms(m.cloud_ms_p50), "cloud latency (p50)")}</div>
      <div class="small faint" style="margin-top:8px">One frame per incident, at most ${m.cap_per_hour} an hour, cached so a frame is never billed twice.</div>`;
  }
  let built = false;
  function drawMore() {
    if (built) {                       // only the live parts change; inputs keep what the user typed
      $("#tl", el).innerHTML = outageTimeline();
      $("#bench", el).innerHTML = benchTable();
      const mm = $("#measured", el); if (mm) mm.innerHTML = measured();
      drawFleet();
      return;
    }
    built = true;
    const p = d.prices;
    $("#more", el).innerHTML = [
      more("Outage timeline", `<div id="tl">${outageTimeline()}</div>`),
      more("Benchmark", `<div id="bench">${benchTable()}</div>`),
      more("Fleet projection", `<div class="grid g3" style="margin-bottom:12px">
          ${slider("towers", "Towers", 1, 500, 1)}${slider("fps", "Frames per second", 0.2, 5, 0.2)}${slider("days", "Days", 1, 365, 1)}</div><div id="fleet"></div>`),
      d.measured.enabled ? more("Measured cloud samples", `<div id="measured">${measured()}</div>`) : "",
      more("Assumptions & prices", `<div class="grid g3">
          ${price("usd_per_mtok_in", "$ per 1M input tokens", p.usd_per_mtok_in)}${price("usd_per_mtok_out", "$ per 1M output tokens", p.usd_per_mtok_out)}${price("usd_per_gb", "$ per GB uplinked", p.usd_per_gb)}</div>
          <div class="row"><span class="grow small faint">Use the provider's current published rates. These are inputs, not measurements.</span><button class="btn sm" type="button" id="save">Apply</button></div>`),
      more("How this is computed", `<dl class="kv">${Object.entries(d.summary.method).map(([k, v]) => `<dt>${esc(k.replace(/_/g, " "))}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
          <button class="btn sm ghost" type="button" id="reset" style="margin-top:10px">Reset the comparison</button>`),
    ].filter(Boolean).join("");
    el.querySelectorAll("input[type=range]").forEach(r => r.oninput = () => {
      fleetArgs[r.dataset.k] = +r.value; r.previousElementSibling.lastElementChild.textContent = r.value; loadFleet();
    });
    $("#save", el).onclick = async () => {
      const body = {};
      el.querySelectorAll("input[data-price]").forEach(i => { if (i.value !== "") body[i.dataset.price] = +i.value; });
      try { await post("/api/edge-cloud/prices", body); toast("Prices applied."); refresh(); } catch (e) { toast(e.message, true); }
    };
    $("#reset", el).onclick = async () => { await post("/api/edge-cloud/reset"); refresh(); };
    drawFleet();
  }
  const pct100 = v => v == null ? "–" : `${(v * 100).toFixed(0)}%`;
  const slider = (k, label, min, max, step) => `<div class="field"><label class="row"><span class="grow">${label}</span><b class="num">${fleetArgs[k]}</b></label>
    <input class="range" type="range" min="${min}" max="${max}" step="${step}" value="${fleetArgs[k]}" data-k="${k}"></div>`;
  const price = (k, label, v) => `<div class="field"><label for="p-${k}">${label}</label><input class="input" id="p-${k}" data-price="${k}" type="number" min="0" step="any" value="${v || ""}" placeholder="0"></div>`;

  let fleetTimer = null;
  function loadFleet() {
    clearTimeout(fleetTimer);
    fleetTimer = setTimeout(async () => {
      try { fleet = await get(`/api/edge-cloud/fleet?towers=${fleetArgs.towers}&fps=${fleetArgs.fps}&days=${fleetArgs.days}`); drawFleet(); } catch (e) { /* keep last */ }
    }, 150);
  }
  function drawFleet() {
    const box = $("#fleet", el);
    if (!box || !fleet) return;
    const [cloud, local, casc] = fleet.rows;
    box.innerHTML = `<table class="t"><thead><tr><th></th><th class="r">VLM calls</th><th class="r">Data uplinked</th><th class="r">Cost</th><th class="r">Fits one GPU</th></tr></thead><tbody>
      ${[["Cloud VLM on every frame", cloud], ["Local VLM on every frame", local], ["Edge cascade (Sentinel)", casc]].map(([l, r]) => `<tr><td>${l}</td>
        <td class="r num">${compact(r.vlm_calls_total)}</td><td class="r num">${bytes(r.upstream_gb_total * 1e9)}</td>
        <td class="r num">${d.summary.prices_set ? usd(r.usd_per_day_total) : "–"}</td><td class="r">${r.feasible ? pill("Yes", "ok") : pill("No", "expired")}</td></tr>`).join("")}</tbody></table>
      <div class="small faint" style="margin-top:8px">Measured on this node: ${fleet.measured.length ? esc(fleet.measured.join(", ").replace(/_/g, " ")) : "nothing yet (defaults)"}; the rest are stated assumptions.</div>`;
  }

  async function refresh() {
    const [x, a] = await Promise.all([get(`/api/edge-cloud?profile=${profile}`), get("/api/alerts").catch(() => ({alerts: []}))]);
    d = x; alerts = a.alerts;
    draw();
  }
  drawProfiles();
  loadFleet();
  return poll(refresh, 4000);
}

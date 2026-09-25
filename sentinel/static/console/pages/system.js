// System: health of each part of the node, resources, and the uplink with a failover test.
import {get, poll, post} from "../api.js";
import {$, dot, esc, icon, info, openDrawer, pill, toast} from "../ui.js";

const uptime = s => s < 3600 ? `${Math.round(s / 60)} min` : s < 172800 ? `${(s / 3600).toFixed(1)} h` : `${Math.round(s / 86400)} d`;

export function mount(el, ctx) {
  el.innerHTML = `<div class="page">
    <div class="grid" style="grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);align-items:start" id="top">
      <div class="card" id="uplink"></div>
      <div class="card" id="res"></div>
    </div>
    <div class="section-title"><span class="grow">Health</span><button class="btn sm ghost" id="more" type="button">Details</button></div>
    <div class="grid g3" id="health"></div>
    <div class="section-title">Appearance</div>
    <div class="card row"><span class="grow">Theme</span><div class="seg" id="theme">
      <button type="button" data-v="dark">Dark</button><button type="button" data-v="light">Light</button></div></div>
  </div>`;
  if (window.matchMedia("(max-width: 900px)").matches) $("#top", el).style.gridTemplateColumns = "1fr";
  let sys = null;
  const gpu = [];
  function drawUplink(o) {
    const up = o.uplink.online, q = o.uplink.queued;
    $("#uplink", el).innerHTML = `<h3>${icon(up ? "wifi" : "wifiOff")}Uplink<span class="spacer"></span>${pill(up ? "Online" : "Offline", up ? "ok" : "warn")}</h3>
      <div class="row" style="gap:16px">
        <div class="grow"><div style="font-size:17px;font-weight:620">${up ? "Connected" : "No connection: deciding locally"}</div>
          <div class="muted small">${up ? "ALERTs go to dispatch and the team phone as they happen." : `${q} ALERT${q === 1 ? "" : "s"} held in the outbox, sent once the link returns.`}</div></div>
        <div class="row small muted">Test outage ${info("Cuts this node's uplink in software, as in a real outage. Detection, incidents and the local alarm keep working; ALERTs queue and are delivered when you switch it back.")}
          <button class="switch" id="outage" role="switch" aria-checked="${!up}" aria-label="Test outage"></button></div>
      </div>`;
    $("#outage", el).onclick = async () => {
      try {
        const r = await post("/api/uplink", {online: !up});
        toast(r.online ? (r.flushed ? `Link restored: ${r.flushed} queued alert${r.flushed === 1 ? "" : "s"} sent.` : "Link restored.") : "Outage started: the node keeps deciding on its own.");
        ctx.refresh();
      } catch (e) { toast(e.message, true); }
    };
  }
  function spark(xs) {
    if (xs.length < 2) return "";
    const w = 220, h = 36, pts = xs.map((v, i) => `${(i / (xs.length - 1)) * w},${h - (v / 100) * h}`).join(" ");
    return `<svg viewBox="0 0 ${w} ${h}" width="100%" height="${h}" preserveAspectRatio="none" aria-hidden="true">
      <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="1.6"/></svg>`;
  }
  async function refresh() {
    sys = await get("/api/system");
    const s = sys.stats || {};
    if (s.gpu_util_pct != null) { gpu.push(s.gpu_util_pct); if (gpu.length > 60) gpu.shift(); }
    const mem = s.mem_total_gb ? (100 * s.mem_used_gb / s.mem_total_gb) : null;
    $("#res", el).innerHTML = `<h3>${icon("cpu")}${esc(sys.node)}<span class="spacer"></span><span class="small faint">up ${uptime(sys.uptime_s)}</span></h3>
      <div class="grid g2">
        <div class="kpi"><div class="v">${s.gpu_util_pct != null ? Math.round(s.gpu_util_pct) : "–"}<small>%</small></div><div class="k">GPU</div></div>
        <div class="kpi"><div class="v">${s.mem_used_gb != null ? s.mem_used_gb.toFixed(0) : "–"}<small>/ ${s.mem_total_gb ? s.mem_total_gb.toFixed(0) : "–"} GB</small></div><div class="k">Unified memory</div></div>
      </div>${spark(gpu)}
      ${mem != null ? `<div style="height:6px;border-radius:3px;background:var(--panel-2);margin-top:8px;overflow:hidden"><div style="height:100%;width:${mem.toFixed(0)}%;background:var(--accent)"></div></div>` : ""}`;
    $("#health", el).innerHTML = sys.health.map(h => `<div class="card row">${dot(h.state === "off" ? "" : h.state)}
      <div class="grow"><div style="font-weight:580">${esc(h.label)}</div><div class="small muted ellipsis">${esc(h.note)}</div></div></div>`).join("");
  }
  $("#more", el).onclick = () => sys && openDrawer("Node details", `<dl class="kv">
    <dt>Node</dt><dd>${esc(sys.node)}</dd><dt>Uptime</dt><dd>${uptime(sys.uptime_s)}</dd>
    <dt>Served models</dt><dd>${sys.served_models.map(m => pill(m, "tag")).join(" ") || "none"}</dd>
    ${sys.models_error ? `<dt>Model server</dt><dd>${esc(sys.models_error)}</dd>` : ""}
    <dt>Internet needed</dt><dd>No. Models, map and console run on this node; only ALERT delivery uses the uplink.</dd></dl>`);
  const theme = () => document.documentElement.dataset.theme || "dark";
  const paintTheme = () => $("#theme", el).querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === theme()));
  $("#theme", el).onclick = e => {
    const b = e.target.closest("button"); if (!b) return;
    document.documentElement.dataset.theme = b.dataset.v;
    try { localStorage.setItem("sentinel-theme", b.dataset.v); } catch (err) { /* private mode */ }
    paintTheme();
  };
  paintTheme();
  const offO = ctx.onOverview(drawUplink);
  const stop = poll(refresh, 3000);
  return () => { stop(); offO(); };
}

// Alerts: every ALERT that left (or is waiting to leave) this device, one row each.
import {get, poll, post} from "../api.js";
import {$, empty, esc, icon, isoTime, kpi, openDrawer, pill, title, toast} from "../ui.js";

const SOURCE = {tower: ["tower", "Tower"], mobile: ["mobile", "Mobile unit"], field: ["upload", "Field report"]};

export function mount(el, ctx) {
  el.innerHTML = `<div class="page">
    <div class="grid g4" id="kpis"></div>
    <div class="section-title"><span class="grow">Deliveries</span>
      <button class="btn sm" id="test" type="button">${icon("send")}Send test alert</button></div>
    <div class="card" style="padding:0"><div class="tablewrap" id="rows"></div></div>
  </div>`;
  let data = null;
  async function refresh() {
    data = await get("/api/alerts");
    const ph = data.phone;
    $("#kpis", el).innerHTML = [
      kpi(data.delivered, "Delivered", {cls: data.delivered ? "good" : ""}),
      kpi(data.queued, "Waiting for the uplink", {cls: data.queued ? "hot" : "", tip: "ALERTs are held in a durable outbox on this device and sent, once each, as soon as the link returns."}),
      kpi(ph.configured ? "On" : "Off", "Phone push", {sub: ph.configured ? esc(ph.host) : "Set NTFY_TOPIC_URL on the node"}),
      kpi(data.dispatch_configured ? "On" : "Phone only", "Dispatch endpoint"),
    ].join("");
    $("#test", el).disabled = !ph.configured || !data.online;
    $("#test", el).title = !ph.configured ? "Phone alerts are not configured" : !data.online ? "The uplink is down" : "";
    const rows = data.alerts;
    $("#rows", el).innerHTML = rows.length ? `<table class="t"><thead><tr><th>Time</th><th>Incident</th><th>Source</th><th>Phone</th><th>Dispatch</th><th class="r">Status</th></tr></thead><tbody>
      ${rows.map((a, i) => {
        const [ic, label] = SOURCE[a.source] || SOURCE.tower;
        return `<tr class="click" data-i="${i}"><td class="num">${isoTime(a.detected_at)}</td>
          <td><div class="row">${pill(a.severity || "ALERT", a.severity || "ALERT")}<span class="ellipsis">${esc(a.incident ? a.incident.name : a.tower_name)}</span></div></td>
          <td><span class="row muted">${icon(ic)}${label}</span></td>
          <td>${channel(a.phone)}</td><td>${channel(a.dispatch)}</td>
          <td class="r">${pill(title(a.state), a.phone === "expired" ? "expired" : a.state)}</td></tr>`;
      }).join("")}</tbody></table>` : empty("No alerts yet", "Only ALERTs leave the device. Everything else stays in the local log.");
  }
  const channel = s => s === "sent" ? `<span class="row" style="color:var(--ok)">${icon("check")}Sent</span>` :
    s === "off" ? `<span class="faint">Off</span>` : s === "pending" ? `<span class="muted">Pending</span>` : `<span style="color:var(--alert)">${esc(title(s))}</span>`;
  $("#rows", el).addEventListener("click", e => {
    const r = e.target.closest("tr[data-i]"); if (!r) return;
    const a = data.alerts[+r.dataset.i];
    openDrawer(`Alert · ${esc(a.incident ? a.incident.name : a.tower_name)}`, `<dl class="kv">
      <dt>Detected</dt><dd>${esc(a.detected_at || "–")}</dd><dt>Camera</dt><dd>${esc(a.tower_name || "–")}</dd>
      <dt>Status</dt><dd>${pill(title(a.state), a.state)}</dd><dt>Phone</dt><dd>${esc(title(a.phone))}</dd>
      <dt>Dispatch</dt><dd>${esc(title(a.dispatch))}</dd><dt>Attempts</dt><dd>${a.attempts}</dd>
      <dt>Report size</dt><dd>${a.bytes ? (a.bytes / 1000).toFixed(1) + " KB" : "–"}</dd>
      ${a.error ? `<dt>Last error</dt><dd>${esc(a.error)}</dd>` : ""}
      <dt>Event id</dt><dd class="num">${esc(a.event_id)}</dd></dl>
      ${a.incident ? `<button class="btn sm" style="margin-top:16px" type="button" id="goinc">${icon("map")}Open incident</button>` : ""}`);
    const b = $("#goinc"); if (b) b.onclick = () => ctx.go("operations", a.incident.id);
  });
  $("#test", el).onclick = async () => {
    try { const r = await post("/api/alerts/test"); toast(r.status === "sent" ? "Test push sent to the team phone." : "Test suppressed: one per 30 s."); }
    catch (e) { toast(e.message, true); }
  };
  return poll(refresh, 2500);
}

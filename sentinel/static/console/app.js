// Sentinel Console shell: navigation, the always-on header, the local ALERT alarm and the Ask panel.
import {get, poll, post} from "./api.js";
import {$, $$, bytes, closeDrawer, esc, icon, isoTime, times, toast, title, usd} from "./ui.js";

try { const t = localStorage.getItem("sentinel-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* private mode */ }

const PAGES = [
  ["operations", "Operations", "map"],
  ["cameras", "Cameras", "camera"],
  ["alerts", "Alerts", "bell"],
  ["edge-cloud", "Edge vs Cloud", "scale"],
  ["models", "Models", "layers"],
  ["system", "System", "cpu"],
];
const listeners = new Set();
export const ctx = {
  overview: null,
  onOverview(cb) { listeners.add(cb); if (ctx.overview) cb(ctx.overview); return () => listeners.delete(cb); },
  refresh: () => refreshOverview(),
  go: (page, arg) => { location.hash = `#/${page}${arg ? "/" + encodeURIComponent(arg) : ""}`; },
};

// ---------------------------------------------------------------- navigation
$("#nav").innerHTML = PAGES.map(([id, label, ic]) =>
  `<a href="#/${id}" data-page="${id}">${icon(ic)}<span>${label}</span></a>`).join("");
let current = null, unmount = null;

async function route() {
  const [, page = "operations", arg] = location.hash.split("/");
  const known = PAGES.find(p => p[0] === page) ? page : "operations";
  closeDrawer();
  if (known === current && !arg) return;
  if (unmount) { try { unmount(); } catch (e) { /* page already gone */ } unmount = null; }
  current = known;
  $$("#nav a").forEach(a => a.classList.toggle("on", a.dataset.page === known));
  $("#page-title").textContent = PAGES.find(p => p[0] === known)[1];
  const outlet = $("#outlet");
  outlet.innerHTML = "";
  const mod = await import(`./pages/${known}.js`);
  if (current !== known) return;                 // navigated away while loading
  unmount = mod.mount(outlet, ctx, arg ? decodeURIComponent(arg) : null) || null;
}
window.addEventListener("hashchange", route);

// ---------------------------------------------------------------- header
function paintHeader(o) {
  $("#node-name").textContent = o.node;
  document.title = (o.incidents.alert ? `(${o.incidents.alert}) ` : "") + "Sentinel Console";
  const up = o.uplink.online, q = o.uplink.queued;
  const u = $("#chip-uplink");
  u.className = "chip" + (up ? "" : " off");
  u.innerHTML = `${icon(up ? "wifi" : "wifiOff")}<span>${up ? "Online" : "Offline"}${q ? ` · ${q} queued` : ""}</span>`;
  const n = o.incidents.active, a = o.incidents.alert;
  $("#chip-incidents").innerHTML = `<span class="dot ${a ? "ALERT live" : n ? "MONITOR" : "ok"}"></span><span>${n ? `${n} active` : "All clear"}</span>`;
  const s = o.savings;
  $("#chip-savings").innerHTML = s.frames && s.bytes > 0
    ? `<span>vs cloud-only: ${s.bytes_x ? times(s.bytes_x) + " less data" : bytes(s.bytes) + " kept local"}${s.prices_set && s.usd ? ` · ${usd(s.usd)} saved` : ""}</span>`
    : `<span>Edge vs cloud</span>`;
}
$("#chip-uplink").onclick = () => ctx.go("system");
$("#chip-savings").onclick = () => ctx.go("edge-cloud");

// ---------------------------------------------------------------- local alarm (works with no uplink)
let lastAlertId, audio;
function tone() {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const t = audio.currentTime;
    for (const [i, f] of [[0, 880], [0.22, 660], [0.44, 880]]) {
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = f; o.type = "sine";
      g.gain.setValueAtTime(0.0001, t + i); g.gain.exponentialRampToValueAtTime(0.25, t + i + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, t + i + 0.2);
      o.connect(g).connect(audio.destination); o.start(t + i); o.stop(t + i + 0.21);
    }
  } catch (e) { /* no audio: the banner still shows */ }
}
function alarm(a) {
  const inc = a.incident;
  const where = inc ? inc.name : a.tower_name;
  const what = `${where || "Camera"} · ${isoTime(a.detected_at)}`;
  $("#alarm").innerHTML = `<div class="alarm" role="alert">${icon("flame")}<strong>ALERT</strong><span class="grow ellipsis">${esc(what)}
    · ${a.state === "sent" ? "dispatch notified" : "queued for dispatch"}</span>
    ${inc ? `<button class="btn sm" type="button" data-open="${esc(inc.id)}">View</button>` : ""}
    <button class="btn sm" type="button" data-dismiss aria-label="Dismiss">${icon("x")}</button></div>`;
  tone();
  if ("Notification" in window && Notification.permission === "granted") {
    try { new Notification("Wildfire ALERT", {body: what, tag: a.event_id}); } catch (e) { /* not allowed here */ }
  }
}
$("#alarm").addEventListener("click", e => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.open) ctx.go("operations", b.dataset.open);
  $("#alarm").innerHTML = "";
});
// browsers only allow notifications and sound after a user gesture
document.addEventListener("click", () => {
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission().catch(() => {});
  if (audio && audio.state === "suspended") audio.resume();
}, {once: true});

async function refreshOverview() {
  const o = await get("/api/overview");
  ctx.overview = o;
  paintHeader(o);
  const a = o.last_alert;
  if (lastAlertId === undefined) lastAlertId = a ? a.event_id : null;    // no alarm for alerts from before this page
  else if (a && a.event_id !== lastAlertId) { lastAlertId = a.event_id; alarm(a); }
  listeners.forEach(cb => { try { cb(o); } catch (e) { console.error(e); } });
  return o;
}
let offlineToast = false;
poll(async () => {
  try { await refreshOverview(); offlineToast = false; }
  catch (e) { if (!offlineToast) { offlineToast = true; toast("Cannot reach the Sentinel service on this device. Retrying…", true); } throw e; }
}, 2000);

// ---------------------------------------------------------------- drawer + Ask
$("#drawer-close").onclick = closeDrawer;
$("#scrim").onclick = () => { closeDrawer(); $("#ask").classList.remove("on"); $$(".modal").forEach(m => m.remove()); $("#scrim").classList.remove("on"); };
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  closeDrawer(); $("#ask").classList.remove("on"); $$(".modal").forEach(m => m.remove()); $("#scrim").classList.remove("on");
});
$("#ask-open").onclick = () => { $("#ask").classList.add("on"); $("#ask-q").focus(); };
$("#ask-close").onclick = () => $("#ask").classList.remove("on");
$("#ask-suggest").addEventListener("click", e => { const b = e.target.closest("button"); if (b) ask(b.textContent); });
$("#ask-form").addEventListener("submit", e => { e.preventDefault(); const q = $("#ask-q").value.trim(); if (q) ask(q); });

async function ask(q) {
  const box = $("#ask-msgs");
  $("#ask-q").value = "";
  box.insertAdjacentHTML("beforeend", `<div class="msg q">${esc(q)}</div>`);
  const pending = document.createElement("div");
  pending.className = "msg a muted";
  pending.textContent = "Thinking on the device…";
  box.appendChild(pending);
  box.scrollTop = box.scrollHeight;
  try {
    const r = await post("/ops/api/ask", {question: q, selected: ctx.selectedIncident || null});
    pending.className = "msg a";
    pending.innerHTML = `${esc(r.answer)}<div class="meta">${esc(title(r.source || "assistant"))} · ${(r.latency_ms / 1000).toFixed(1)} s · 0 bytes sent off the device</div>`;
  } catch (e) {
    pending.className = "msg a";
    pending.textContent = "The assistant is not available right now: " + e.message;
  }
  box.scrollTop = box.scrollHeight;
}

route();

// Small UI kit for the console: formatting, icons, drawer, modal, tooltips, toasts. No dependencies.
export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => [...el.querySelectorAll(s)];
export const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

// ---------------------------------------------------------------- formatting
export const num = (n, d = 0) => n == null || !isFinite(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
export const compact = n => n == null || !isFinite(n) ? "–" : Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(n);
export function bytes(b) {
  if (b == null || !isFinite(b)) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (Math.abs(b) >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b.toFixed(b < 10 && i ? 1 : 0)} ${u[i]}`;
}
export function ms(v) {
  if (v == null || !isFinite(v)) return "–";
  return v < 1000 ? `${Math.round(v)} ms` : v < 60000 ? `${(v / 1000).toFixed(1)} s` : `${(v / 60000).toFixed(1)} min`;
}
export const usd = v => v == null || !isFinite(v) ? "–" : v >= 100 ? `$${num(v)}` : v >= 1 ? `$${v.toFixed(2)}` : v >= 0.0001 ? `$${v.toFixed(4)}` : v > 0 ? "< $0.0001" : "$0";
export const pct = (v, d = 0) => v == null || !isFinite(v) ? "–" : `${v.toFixed(d)}%`;
export const times = v => v == null || !isFinite(v) ? "–" : v >= 100 ? `${compact(v)}×` : `${v.toFixed(v < 10 ? 1 : 0)}×`;
export function ago(t) {
  if (!t) return "–";
  const s = Date.now() / 1000 - t;
  return s < 45 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 172800 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} d ago`;
}
export const clock = t => t ? new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}) : "–";
export const isoTime = iso => iso ? clock(Date.parse(iso) / 1000) : "–";
export const words = s => String(s || "unknown").replace(/_/g, " ");
export const title = s => words(s).replace(/\b\w/g, c => c.toUpperCase());

// ---------------------------------------------------------------- icons (Lucide-style paths, inline)
const P = {
  map: '<path d="M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2z"/><path d="M9 4v14M15 6v14"/>',
  camera: '<path d="M4 7h3l2-3h6l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1z"/><circle cx="12" cy="13" r="4"/>',
  bell: '<path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.9 1.9 0 0 0 3.4 0"/>',
  scale: '<path d="M12 3v18M5 21h14M3 7h18"/><path d="m6 7-3 7a3 3 0 0 0 6 0L6 7zM18 7l-3 7a3 3 0 0 0 6 0l-3-7z"/>',
  cpu: '<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5z"/><path d="m3 13 9 5 9-5"/>',
  chat: '<path d="M21 15a2 2 0 0 1-2 2H8l-5 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
  flame: '<path d="M12 2c1 3 5 5 5 10a5 5 0 0 1-10 0c0-2 1-3.5 2-4.5 0 2 1 3 2 3-1-3 0-6 1-8.5z"/>',
  tower: '<path d="M8 21 12 9l4 12M9.5 17h5M12 9V5"/><path d="M8.5 4.5a5 5 0 0 1 7 0M6 2a8.5 8.5 0 0 1 12 0"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
  chev: '<path d="m9 6 6 6-6 6"/>',
  wifi: '<path d="M5 12.5a10 10 0 0 1 14 0M8.5 16a5 5 0 0 1 7 0M2 9a15 15 0 0 1 20 0"/><circle cx="12" cy="19.5" r=".6"/>',
  wifiOff: '<path d="m2 2 20 20M8.5 16a5 5 0 0 1 7 0M5 12.5a10 10 0 0 1 5-2.7M2 9a15 15 0 0 1 4.2-2.7M16.7 10.2A10 10 0 0 1 19 12.5M22 9a15 15 0 0 0-10-4"/><circle cx="12" cy="19.5" r=".6"/>',
  upload: '<path d="M12 15V3M7 8l5-5 5 5"/><path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4"/>',
  play: '<path d="m7 4 13 8-13 8z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
  phone: '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/>',
  send: '<path d="m22 2-7 20-4-9-9-4z"/><path d="M22 2 11 13"/>',
  target: '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>',
  mobile: '<path d="M3 17h2l2-5h8l2 5h4"/><circle cx="7" cy="17.5" r="2"/><circle cx="17" cy="17.5" r="2"/><path d="M9 12V8h5l2 4"/>',
  check: '<path d="m5 12 5 5 9-10"/>',
  dash: '<path d="M4 20h16M7 16V9M12 16V5M17 16v-4"/>',
};
export const icon = (name, cls = "") => `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true">${P[name] || ""}</svg>`;

// ---------------------------------------------------------------- components
export const info = text => `<button type="button" class="info" data-tip="${esc(text)}" aria-label="${esc(text)}">i</button>`;
export function kpi(value, label, {cls = "", sub = "", tip = "", unit = ""} = {}) {
  return `<div class="card kpi ${cls}"><div class="v">${value}${unit ? `<small>${esc(unit)}</small>` : ""}</div>
    <div class="k">${esc(label)}${tip ? info(tip) : ""}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}
export const pill = (text, cls = "") => `<span class="pill ${esc(cls)}">${esc(text)}</span>`;
export const dot = (cls = "") => `<span class="dot ${esc(cls)}"></span>`;
export function more(summary, body, open = false) {
  return `<details class="more"${open ? " open" : ""}><summary>${summary}${icon("chev", "chev")}</summary><div class="body">${body}</div></details>`;
}
export function seg(options, value, attr = "data-v") {
  return `<div class="seg" role="radiogroup">${options.map(([v, label]) =>
    `<button type="button" role="radio" aria-checked="${v === value}" class="${v === value ? "on" : ""}" ${attr}="${esc(v)}">${esc(label)}</button>`).join("")}</div>`;
}
export function empty(big, text = "", action = "") {
  return `<div class="empty"><div class="big">${esc(big)}</div>${text ? `<div>${esc(text)}</div>` : ""}${action}</div>`;
}

// tooltips: one floating element, for every [data-tip]
let tipEl = null;
function showTip(e) {
  const t = e.target.closest("[data-tip]");
  if (!t) return;
  tipEl = tipEl || document.body.appendChild(Object.assign(document.createElement("div"), {className: "tip"}));
  tipEl.textContent = t.dataset.tip;
  tipEl.hidden = false;
  const r = t.getBoundingClientRect(), w = Math.min(300, window.innerWidth - 24);
  tipEl.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - w - 12))}px`;
  tipEl.style.top = `${r.bottom + 8}px`;
}
const hideTip = e => { if (tipEl && e.target.closest("[data-tip]")) tipEl.hidden = true; };
document.addEventListener("mouseover", showTip);
document.addEventListener("focusin", showTip);
document.addEventListener("mouseout", hideTip);
document.addEventListener("focusout", hideTip);

// drawer: one at a time, for details on demand
const scrim = () => $("#scrim");
export function openDrawer(heading, html, {onClose} = {}) {
  const d = $("#drawer");
  $("#drawer-title").innerHTML = heading;
  $("#drawer-body").innerHTML = html;
  d.classList.add("on");
  d.setAttribute("aria-hidden", "false");
  scrim().classList.add("on");
  d._onClose = onClose;
  $("#drawer-close").focus({preventScroll: true});
  return $("#drawer-body");
}
export function closeDrawer() {
  const d = $("#drawer");
  if (!d.classList.contains("on")) return;
  d.classList.remove("on");
  d.setAttribute("aria-hidden", "true");
  scrim().classList.remove("on");
  const cb = d._onClose; d._onClose = null;
  if (cb) cb();
}
export const drawerOpen = () => $("#drawer").classList.contains("on");

export function modal(html) {
  const m = document.createElement("div");
  m.className = "modal";
  m.innerHTML = `<div class="box" role="dialog" aria-modal="true">${html}</div>`;
  scrim().classList.add("on");
  document.body.appendChild(m);
  const close = () => { m.remove(); if (!drawerOpen()) scrim().classList.remove("on"); };
  m.addEventListener("click", e => { if (e.target === m || e.target.closest("[data-close]")) close(); });
  return {el: m, close};
}

export function toast(text, err = false) {
  const t = document.createElement("div");
  t.className = "toast" + (err ? " err" : "");
  t.textContent = text;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), err ? 6000 : 3500);
}

// numbers that count up when they change (only when the tab can see it)
export function countUp(el, to, fmt) {
  const from = +(el.dataset.v || 0);
  el.dataset.v = to;
  if (!isFinite(to) || !isFinite(from) || from === to || document.hidden || matchMedia("(prefers-reduced-motion: reduce)").matches) {
    el.textContent = fmt(to);
    return;
  }
  const t0 = performance.now(), dur = 600;
  const step = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

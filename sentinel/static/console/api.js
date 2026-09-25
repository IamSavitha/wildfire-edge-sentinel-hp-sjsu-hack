// Fetch helpers for the console. Every path is on this device: /api (console), /ops (incident map), /lab (image
// and mobile camera). Nothing here needs the internet.
export async function get(path, opts = {}) {
  const r = await fetch(path, {cache: "no-store", ...opts});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = j.detail;
    throw new Error(typeof d === "string" ? d : d ? JSON.stringify(d) : `${r.status} ${r.statusText}`);
  }
  return j;
}
export const post = (path, body, method = "POST") =>
  get(path, {method, headers: {"Content-Type": "application/json"}, body: JSON.stringify(body ?? {})});
export const patch = (path, body) => post(path, body, "PATCH");
export const del = path => get(path, {method: "DELETE"});
export const form = (path, fd) => get(path, {method: "POST", body: fd});

// poll(fn, ms): runs fn now and every `ms` while the tab is visible; returns a stop function
export function poll(fn, every) {
  let timer = null, stopped = false, busy = false;
  const tick = async () => {
    if (stopped) return;
    if (!document.hidden && !busy) {
      busy = true;
      try { await fn(); } catch (e) { /* the next tick retries */ } finally { busy = false; }
    }
    if (!stopped) timer = setTimeout(tick, every);
  };
  const wake = () => { if (!document.hidden && !stopped) { clearTimeout(timer); tick(); } };
  document.addEventListener("visibilitychange", wake);
  tick();
  return () => { stopped = true; clearTimeout(timer); document.removeEventListener("visibilitychange", wake); };
}

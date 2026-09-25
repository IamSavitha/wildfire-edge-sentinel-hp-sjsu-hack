// Operations: the map, the incident list and the incident drawer. Map and tiles are served by this device.
import {form, get, patch, poll} from "../api.js";
import {$, $$, ago, clock, closeDrawer, dot, empty, esc, icon, modal, more, openDrawer, pill, seg, title, toast, words} from "../ui.js";

const RANK = {ALERT: 3, MONITOR: 2, LOG: 1};
const LOC = {exif_gps: "GPS in the photo", reported_gps: "Reported with the photo", camera_bearing: "Camera bearing + assumed range",
             camera_site: "At the camera", map_pin: "Placed by an operator", triangulated: "Triangulated from several towers"};
const fc = f => ({type: "FeatureCollection", features: f});
const poly = ring => ({type: "Feature", properties: {}, geometry: {type: "Polygon", coordinates: [ring]}});

export function mount(el, ctx, openId) {
  el.innerHTML = `<div class="page full"><div class="ops">
    <div id="map"></div>
    <section class="overlay" aria-label="Incidents">
      <div class="oh">
        <div class="row"><h2>Incidents</h2><button class="btn sm" id="report" type="button">${icon("plus")}Field report</button></div>
        <div class="row">${seg([["active", "Active"], ["all", "All"]], "active")}<span class="grow"></span><span class="small faint" id="count"></span></div>
      </div>
      <div class="inc-list" id="list"></div>
    </section></div></div>`;
  let state = null, filter = "active", selected = openId || null, map = null, fitted = false;
  const markers = new Map();

  // ---------------------------------------------------------------- map
  function initMap() {
    const gl = window.maplibregl;
    if (!gl || !window.pmtiles || !window.basemaps) {
      $("#map", el).innerHTML = `<div class="map-fallback"><div>${icon("map")}<div style="margin-top:8px">Map layer not installed on this node.</div>
        <div class="small faint">Incidents are listed on the left; every other part of the console works.</div></div></div>`;
      return;
    }
    try {
      if (!initMap.protocol) { initMap.protocol = new pmtiles.Protocol(); gl.addProtocol("pmtiles", initMap.protocol.tile); }
      const dark = document.documentElement.dataset.theme !== "light";
      map = new gl.Map({
        container: $("#map", el), center: [-117.0, 33.1], zoom: 8.3, attributionControl: {compact: true},
        style: {version: 8, glyphs: location.origin + "/assets/fonts/{fontstack}/{range}.pbf",
          sprite: location.origin + "/assets/sprites/v4/" + (dark ? "dark" : "light"),
          sources: {protomaps: {type: "vector", url: "pmtiles://" + location.origin + "/tiles/sierra-pacific.pmtiles",
                                attribution: "© OpenStreetMap · Protomaps"}},
          layers: basemaps.layers("protomaps", basemaps.namedFlavor(dark ? "dark" : "light"), {lang: "en"})},
      });
      map.addControl(new gl.NavigationControl({showCompass: false}), "bottom-right");
      map.on("load", () => {
        for (const id of ["fov", "wedges", "cone"]) map.addSource(id, {type: "geojson", data: fc([])});
        map.addLayer({id: "fov", type: "fill", source: "fov", paint: {"fill-color": "#3b82f6", "fill-opacity": .05}});
        map.addLayer({id: "fov-l", type: "line", source: "fov", paint: {"line-color": "#3b82f6", "line-opacity": .25, "line-width": 1}});
        map.addLayer({id: "cone", type: "fill", source: "cone", paint: {"fill-color": "#ff5a3c", "fill-opacity": .18}});
        map.addLayer({id: "cone-l", type: "line", source: "cone", paint: {"line-color": "#ff5a3c", "line-width": 1.2, "line-dasharray": [2, 2]}});
        map.addLayer({id: "wedges", type: "fill", source: "wedges", paint: {"fill-color": "#f5a524", "fill-opacity": .16}});
        map.loaded_ = true;
        draw();
      });
    } catch (e) {
      map = null;
      $("#map", el).innerHTML = `<div class="map-fallback">The map needs WebGL, which is off in this browser.</div>`;
    }
  }

  function marker(key, lngLat, cls, html, onClick) {
    let m = markers.get(key);
    if (!m) {
      const node = document.createElement("div");
      node.addEventListener("click", e => { e.stopPropagation(); onClick(); });
      m = new maplibregl.Marker({element: node}).setLngLat(lngLat).addTo(map);
      markers.set(key, m);
    }
    m.setLngLat(lngLat);
    const node = m.getElement();
    node.className = cls;
    node.innerHTML = html;
    node._seen = true;
    return m;
  }

  function drawMap() {
    if (!map || !map.loaded_ || !state) return;
    for (const m of markers.values()) m.getElement()._seen = false;
    for (const c of state.cameras) {
      marker("cam:" + c.id, [c.lon, c.lat], "cam-marker" + (c.mobile ? " mobile" : ""), icon(c.mobile ? "mobile" : "tower"),
        () => toast(`${c.name} · ${c.site || ""}`)).getElement().title = c.name;
    }
    for (const i of visible()) {
      const cls = `marker ${i.status === "resolved" ? "resolved" : i.severity}${i.id === selected ? " sel" : ""}`;
      marker("inc:" + i.id, [i.lon, i.lat], cls, "", () => open(i.id)).getElement().title = i.name;
    }
    for (const [k, m] of markers) if (!m.getElement()._seen) { m.remove(); markers.delete(k); }
    map.getSource("fov").setData(fc(state.cameras.filter(c => c.fov && !c.mobile).map(c => poly(c.fov))));
    const sel = state.incidents.find(i => i.id === selected);
    map.getSource("wedges").setData(fc(sel ? (sel.wedges || []).map(poly) : []));
    map.getSource("cone").setData(fc(sel && sel.cone ? [poly(sel.cone.ring)] : []));
    if (!fitted) {
      fitted = true;
      const b = new maplibregl.LngLatBounds();
      const pts = visible().length ? visible() : state.cameras;
      pts.forEach(p => b.extend([p.lon, p.lat]));
      if (!b.isEmpty()) map.fitBounds(b, {padding: {top: 60, bottom: 60, left: 380, right: 60}, maxZoom: 11, duration: 0});
    }
  }

  // ---------------------------------------------------------------- list
  const visible = () => (state ? state.incidents : []).filter(i => filter === "all" || i.status === "active")
    .sort((a, b) => (a.status === "active") - (b.status === "active") || (RANK[a.severity] || 0) - (RANK[b.severity] || 0) || a.updated_at - b.updated_at)
    .reverse();

  function drawList() {
    const list = $("#list", el), incs = visible();
    $("#count", el).textContent = state ? `${incs.length} shown` : "";
    if (!incs.length) {
      const cams = state ? state.cameras.length : 0;
      list.innerHTML = empty("All clear", `${cams} camera${cams === 1 ? "" : "s"} watching. New smoke opens an incident here.`);
      return;
    }
    list.innerHTML = incs.map(i => {
      const esc_ = i.escalation || {};
      const flag = esc_.decision === "queued" ? pill("Queued", "queued") : esc_.decision === "sent" ? pill("Sent", "sent") : "";
      return `<div class="inc${i.id === selected ? " sel" : ""}" data-id="${esc(i.id)}" role="button" tabindex="0">
        ${dot((i.status === "resolved" ? "resolved" : i.severity) + (i.severity === "ALERT" && i.status === "active" ? " live" : ""))}
        <div class="grow"><div class="row"><span class="name grow ellipsis">${esc(i.name)}</span>${flag}</div>
        <div class="meta ellipsis">${esc(title(i.source_type))} · ${esc(i.camera_name || "field report")} · ${ago(i.updated_at)}</div></div></div>`;
    }).join("");
  }
  $("#list", el).addEventListener("click", e => { const r = e.target.closest(".inc"); if (r) open(r.dataset.id); });
  $("#list", el).addEventListener("keydown", e => { const r = e.target.closest(".inc"); if (r && e.key === "Enter") open(r.dataset.id); });
  $(".seg", el).addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    filter = b.dataset.v;
    $$(".seg button", el).forEach(x => { x.classList.toggle("on", x === b); x.setAttribute("aria-checked", x === b); });
    draw();
  });

  function draw() { drawList(); drawMap(); }

  // ---------------------------------------------------------------- incident drawer
  async function open(id) {
    selected = id;
    ctx.selectedIncident = id;
    draw();
    let inc;
    try { inc = await get(`/ops/api/incidents/${encodeURIComponent(id)}?hours=2`); }
    catch (e) { toast("Incident not found: " + e.message, true); return; }
    if (map && map.loaded_) map.easeTo({center: [inc.lon, inc.lat], zoom: Math.max(map.getZoom(), 10), duration: 500});
    const frames = inc.frames || [], key = inc.key_frame ?? (frames.length ? frames[frames.length - 1].n : null);
    const src = n => `/ops/api/incidents/${encodeURIComponent(inc.id)}/frames/${n}`;
    const escn = inc.escalation || {};
    const w = inc.wind;
    const updates = (inc.updates || []).slice().reverse();
    const body = openDrawer(`<span class="row">${dot(inc.severity)}<span class="ellipsis">${esc(inc.name)}</span></span>`, `
      <div class="frame-hero"><img id="hero" alt="Key frame" src="${key != null ? src(key) : `/ops/api/incidents/${encodeURIComponent(inc.id)}/thumb`}"></div>
      ${frames.length > 1 ? `<div class="strip" id="strip">${frames.map(f => `<img loading="lazy" alt="Frame ${f.n}" data-n="${f.n}" class="${f.n === key ? "on" : ""}" src="${src(f.n)}">`).join("")}</div>` : ""}
      <div class="row wrap" style="margin-bottom:14px">${pill(inc.severity, inc.severity)}${pill(title(inc.source_type), "tag")}
        ${inc.trend ? pill(title(inc.trend), "tag") : ""}${pill(inc.status === "active" ? "Active" : "Resolved", "tag")}</div>
      <div class="card" style="margin-bottom:12px"><div class="small muted">Latest</div><div>${esc(inc.last_update || inc.description || "–")}</div></div>
      <div class="grid g2" style="margin-bottom:12px">
        <div class="card kpi"><div class="v" style="font-size:20px">${escn.decision ? title(escn.decision) : "Logged"}</div><div class="k">Dispatch</div></div>
        <div class="card kpi"><div class="v" style="font-size:20px">${inc.cameras_seeing && inc.cameras_seeing.length ? inc.cameras_seeing.length : 1}</div><div class="k">Camera${(inc.cameras_seeing || []).length > 1 ? "s" : ""} seeing it</div></div>
      </div>
      ${more("Timeline", `<div class="timeline">${updates.map(u => `<div class="ev"><span class="t">${clock(u.t)}</span>${dot(u.severity)}<span>${esc(u.text)}</span></div>`).join("") || "<span class='muted'>No updates yet.</span>"}</div>`)}
      ${more("Location & wind", `<dl class="kv">
        <dt>Position</dt><dd class="num">${inc.lat.toFixed(4)}, ${inc.lon.toFixed(4)}</dd>
        <dt>Method</dt><dd>${esc(LOC[inc.loc_source] || words(inc.loc_source))}</dd>
        ${inc.county ? `<dt>County</dt><dd>${esc(inc.county)}</dd>` : ""}
        ${w && w.speed_mps != null ? `<dt>Wind</dt><dd>from ${Math.round(w.dir_from_deg)}° at ${(w.speed_mps * 2.237).toFixed(0)} mph <span class="faint">(${esc(w.label || w.source)})</span></dd>` : "<dt>Wind</dt><dd class='muted'>No reading</dd>"}
        ${inc.cone ? `<dt>Spread</dt><dd>toward ${Math.round(inc.cone.toward_deg)}° (cone on the map)</dd>` : ""}</dl>`)}
      ${inc.report_text ? more("Dispatch report", `<pre class="report">${esc(inc.report_text)}</pre>
        <div class="small faint" style="margin-top:8px">${esc(escn.note || "")}</div>`) : ""}
      <div class="row" style="margin-top:16px">
        <button class="btn sm" type="button" id="status">${inc.status === "active" ? `${icon("check")}Mark resolved` : "Reopen"}</button>
        <span class="grow"></span><span class="small faint">Opened ${ago(inc.created_at)}</span></div>`,
      {onClose: () => { selected = null; ctx.selectedIncident = null; draw(); }});
    const strip = $("#strip", body);
    if (strip) strip.addEventListener("click", e => {
      const im = e.target.closest("img"); if (!im) return;
      $("#hero", body).src = src(im.dataset.n);
      $$("img", strip).forEach(x => x.classList.toggle("on", x === im));
    });
    $("#status", body).onclick = async () => {
      try { await patch(`/ops/api/incidents/${encodeURIComponent(inc.id)}`, {status: inc.status === "active" ? "resolved" : "active"}); closeDrawer(); refresh(); }
      catch (e) { toast(e.message, true); }
    };
  }

  // ---------------------------------------------------------------- field report
  $("#report", el).onclick = () => {
    const cams = (state ? state.cameras : []).filter(c => !c.mobile);
    const m = modal(`<h2>Field report</h2>
      <p class="muted small" style="margin-top:-6px">A photo from a ranger or a member of the public. It runs through the same models on this device.</p>
      <label class="drop" id="drop">${icon("upload")}<div>Choose a photo or drop it here</div><input type="file" accept="image/*" id="file" hidden></label>
      <div class="field" style="margin-top:12px"><label for="cam">Seen from</label><select class="input" id="cam">
        <option value="">Its own location (GPS or coordinates)</option>${cams.map(c => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("")}</select></div>
      <div class="grid g2"><div class="field"><label for="lat">Latitude</label><input class="input" id="lat" inputmode="decimal" placeholder="optional"></div>
        <div class="field"><label for="lon">Longitude</label><input class="input" id="lon" inputmode="decimal" placeholder="optional"></div></div>
      <div class="row"><span class="grow small faint" id="fname"></span><button class="btn" type="button" data-close>Cancel</button>
        <button class="btn primary" type="button" id="send" disabled>Analyse</button></div>`);
    const box = m.el;
    let file = null;
    const pick = f => { file = f; $("#fname", box).textContent = f ? f.name : ""; $("#send", box).disabled = !f; };
    $("#file", box).onchange = e => pick(e.target.files[0]);
    const drop = $("#drop", box);
    drop.addEventListener("dragover", e => { e.preventDefault(); drop.classList.add("over"); });
    drop.addEventListener("dragleave", () => drop.classList.remove("over"));
    drop.addEventListener("drop", e => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); });
    $("#send", box).onclick = async () => {
      const fd = new FormData();
      fd.append("image", file);
      if ($("#cam", box).value) fd.append("camera_id", $("#cam", box).value);
      if ($("#lat", box).value && $("#lon", box).value) { fd.append("lat", $("#lat", box).value); fd.append("lon", $("#lon", box).value); }
      $("#send", box).disabled = true; $("#send", box).textContent = "Analysing…";
      try {
        const r = await form("/ops/api/report", fd);
        m.close();
        if (r.incident) { toast(`${r.result.severity}: ${words(r.incident.source_type)}`); await refresh(); open(r.incident.id); }
        else toast(r.note || "No smoke found: nothing stored.");
      } catch (e) { toast(e.message, true); $("#send", box).disabled = false; $("#send", box).textContent = "Analyse"; }
    };
  };

  async function refresh() {
    state = await get("/ops/api/state?hours=2");
    draw();
  }
  initMap();
  const stop = poll(refresh, 3000);
  if (openId) setTimeout(() => open(openId), 0);
  return () => { stop(); for (const m of markers.values()) m.remove(); if (map) map.remove(); };
}

// Models: what is deployed on this node, how well it scores, which VLM adapter is live, and a side-by-side
// comparison with the base models on any image.
import {form, get, post} from "../api.js";
import {$, esc, icon, kpi, more, ms, num, openDrawer, pill, title, toast} from "../ui.js";

const p100 = v => v == null ? "–" : `${(v * 100).toFixed(1)}`;

export function mount(el) {
  el.innerHTML = `<div class="page">
    <div class="grid g2" id="cards"></div>
    <div class="section-title">Compare on an image</div>
    <div class="card">
      <div class="row wrap" style="margin-bottom:14px">
        <span class="grow muted">Base models vs the fine-tuned ones, on the same picture.</span>
        <label class="btn" for="cmp-file">${icon("upload")}Choose image</label><input type="file" id="cmp-file" accept="image/*" hidden>
      </div>
      <div class="grid g2" id="cmp"></div>
    </div>
  </div>`;
  let m = null;

  function draw() {
    const d = m.detector, v = m.vlm, dr = d.results || {}, vr = v.results || {};
    const cur = vr[v.active] || vr.context || {};
    $("#cards", el).innerHTML = `
      <div class="card">
        <h3>${icon("target")}Detector<span class="spacer"></span>${pill(d.version, "tag")}</h3>
        <div class="row" style="align-items:flex-end;gap:18px">
          <div class="kpi"><div class="v">${d.headline.value != null ? d.headline.value.toFixed(3) : "–"}</div><div class="k">${esc(d.headline.label)}</div></div>
          <div class="kpi"><div class="v" style="font-size:20px">${dr.tower && dr.tower.ms_per_image != null ? ms(dr.tower.ms_per_image) : "–"}</div><div class="k">per frame</div></div>
        </div>
        <div class="muted small" style="margin-top:12px">${esc(d.name)} · trained on ${esc(d.trained_on)}</div>
        <button class="btn sm ghost" type="button" id="det-more" style="margin-top:8px">Details</button>
      </div>
      <div class="card">
        <h3>${icon("layers")}Context VLM<span class="spacer"></span>${pill(v.active, "ok")}</h3>
        <div class="row" style="align-items:flex-end;gap:18px">
          <div class="kpi"><div class="v">${p100(cur.source_type_acc)}<small>%</small></div><div class="k">${esc(v.headline.label)}</div></div>
          <div class="kpi"><div class="v" style="font-size:20px">${cur.tokens_per_call != null ? num(cur.tokens_per_call) : "–"}</div><div class="k">tokens per call</div></div>
        </div>
        <div class="row wrap" style="margin-top:12px"><span class="muted small grow">${esc(v.name)} · distilled from ${esc(v.distilled_from)}</span>
          ${v.options.length > 1 ? `<div class="seg" id="adapter">${v.options.map(o => `<button type="button" data-v="${esc(o)}" class="${o === v.active ? "on" : ""}">${esc(o)}</button>`).join("")}</div>` : ""}</div>
        <button class="btn sm ghost" type="button" id="vlm-more" style="margin-top:8px">Details</button>
      </div>`;
    $("#det-more", el).onclick = () => openDrawer("Detector · evaluation", `
      <table class="t"><thead><tr><th>Validation set</th><th class="r">mAP50</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">Images</th></tr></thead><tbody>
      ${[["Tower (HPWREN)", dr.tower], ["D-Fire", dr.dfire], ["YOLO-World zero-shot (before)", dr.before]].map(([l, r]) => r ? `<tr><td>${l}</td>
        <td class="r num">${r.map50 != null ? r.map50.toFixed(3) : "–"}</td><td class="r num">${r.precision != null ? r.precision.toFixed(2) : "–"}</td>
        <td class="r num">${r.recall != null ? r.recall.toFixed(2) : "–"}</td><td class="r num">${num(r.n_images)}</td></tr>` : "").join("")}</tbody></table>
      <dl class="kv" style="margin-top:16px"><dt>Weights</dt><dd>${esc(d.weights || "–")}</dd><dt>Release</dt><dd>${esc(d.version)}</dd></dl>`);
    $("#vlm-more", el).onclick = () => {
      const cols = [["before", "Base 7B"], ["context", "LoRA v1"], ["context_v2", "LoRA v2"]].filter(([k]) => vr[k]);
      const ps = cols.map(([k]) => vr[k].per_source || {});
      const classes = [...new Set(ps.flatMap(o => Object.keys(o)))];
      const cell = (o, c) => { const x = o[c]; if (!x) return "–"; return typeof x === "object" ? `${x.correct ?? x.hit ?? "–"}/${x.n ?? x.total ?? "–"}` : esc(x); };
      openDrawer("Context VLM · evaluation", `
        <table class="t"><thead><tr><th></th>${cols.map(([, l]) => `<th class="r">${l}</th>`).join("")}</tr></thead><tbody>
        <tr><td>Source-type agreement</td>${cols.map(([k]) => `<td class="r num">${p100(vr[k].source_type_acc)}%</td>`).join("")}</tr>
        <tr><td>Danger / benign / look-alike</td>${cols.map(([k]) => `<td class="r num">${p100(vr[k].group_acc)}%</td>`).join("")}</tr>
        <tr><td>Tokens per call</td>${cols.map(([k]) => `<td class="r num">${num(vr[k].tokens_per_call)}</td>`).join("")}</tr>
        <tr><td>Parse failures</td>${cols.map(([k]) => `<td class="r num">${p100(vr[k].parse_fail_rate)}%</td>`).join("")}</tr></tbody></table>
        ${classes.length ? more("Per source type", `<table class="t"><tbody>${classes.map(c => `<tr><td>${esc(title(c))}</td>${ps.map(o => `<td class="r num">${cell(o, c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`) : ""}
        <p class="small faint">Agreement with the Qwen2.5-VL-32B teacher on ${num((vr.context || {}).n)} held-out crops.</p>`);
    };
    const seg = $("#adapter", el);
    if (seg) seg.onclick = async e => {
      const b = e.target.closest("button"); if (!b || b.dataset.v === v.active) return;
      try { await post("/api/models/vlm", {model: b.dataset.v}); toast(`Now using ${b.dataset.v} for every camera.`); load(); }
      catch (err) { toast(err.message, true); }
    };
  }
  async function load() { m = await get("/api/models"); draw(); }

  // ---------------------------------------------------------------- compare on an image
  let url = null;
  $("#cmp-file", el).onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    if (url) URL.revokeObjectURL(url);
    url = URL.createObjectURL(f);
    $("#cmp", el).innerHTML = ["Before", "After"].map(l => `<div><div class="frame-hero" style="margin:0 0 10px"><img alt="" src="${url}"></div><div class="muted small">${l}: running…</div></div>`).join("");
    const fd = new FormData();
    fd.append("image", f);
    fd.append("pipelines", "before,after");
    fd.append("force_vlm", "1");
    try { const r = await form("/lab/api/analyze", fd); paint(r); }
    catch (err) { toast(err.message, true); $("#cmp", el).innerHTML = ""; }
    e.target.value = "";
  };
  function paint(r) {
    const W = r.image.width, H = r.image.height;
    $("#cmp", el).innerHTML = r.results.map(p => {
      if (p.error) return `<div class="card">${esc(p.label)}<div class="muted small">${esc(p.error)}</div></div>`;
      const boxes = (p.detections || []).slice(0, 20).map(d => {
        const [x1, y1, x2, y2] = d.box;
        return `<rect x="${x1}" y="${y1}" width="${Math.max(1, x2 - x1)}" height="${Math.max(1, y2 - y1)}" fill="none"
          stroke="${d.above_gate ? "#ff5a3c" : "rgba(255,255,255,.5)"}" stroke-width="${Math.max(2, W / 300)}" ${d.above_gate ? "" : `stroke-dasharray="${W / 100}"`}/>`;
      }).join("");
      const ctx = p.context || {};
      return `<div><div class="frame-hero" style="margin:0 0 10px"><img alt="" src="${url}">
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" style="position:absolute;inset:0;width:100%;height:100%">${boxes}</svg></div>
        <div class="row wrap">${pill(p.pipeline === "after" ? "After" : "Before", p.pipeline === "after" ? "ok" : "tag")}
          ${p.severity ? pill(p.severity, p.severity) : ""}<strong>${esc(title(ctx.source_type || (p.severity === "IGNORE" ? "no smoke" : "–")))}</strong></div>
        <div class="small muted" style="margin-top:6px">${esc(p.detector_label)} + ${esc(p.vlm_label)} · ${ms((p.detect_ms || 0) + (p.vlm_ms || 0))}${p.tokens ? ` · ${num(p.tokens)} tokens` : ""}</div>
        ${ctx.description ? `<div class="small" style="margin-top:6px">${esc(ctx.description)}</div>` : ""}</div>`;
    }).join("");
  }
  load().catch(e => toast(e.message, true));
  return () => { if (url) URL.revokeObjectURL(url); };
}

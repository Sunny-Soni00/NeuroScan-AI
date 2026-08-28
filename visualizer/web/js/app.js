/* app.js — data loading, theme, transport, per-stage viewer, live threshold. */
"use strict";

const DATA_ROOT = "../data";
const $ = (s) => document.querySelector(s);

let MANIFEST = null, modelKey = null, sampleName = null;
let stageIdx = 0, playTimer = null, ARCH = null, booting = true;

const asset = (f) => `${DATA_ROOT}/${modelKey}/${sampleName}/${f}`;
const model = () => MANIFEST.models[modelKey];
const sample = () => model().samples.find((s) => s.name === sampleName);
const stages = () => model().stages;
const stAsset = (id) => (sample().stages[id] || {});
const fmt = (n) => n.toLocaleString("en-US");

/* ---- theme ---- */
function initTheme() {
  let t = null;
  const urlT = new URLSearchParams(location.hash.slice(1)).get("theme");
  try { t = localStorage.getItem("seg-theme"); } catch (e) {}
  if (urlT === "light" || urlT === "dark") t = urlT;
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  $("#themeBtn").onclick = () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("seg-theme", next); } catch (e) {}
  };
}

/* ---- shareable URL state ---- */
function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { model: h.get("model"), sample: h.get("sample"), stage: +h.get("stage") };
}
function writeHash() {
  if (booting) return;
  location.replace(`#model=${modelKey}&sample=${sampleName}&stage=${stageIdx}`);
}
function instructiveness(s) {
  const t = s.tumor_pct_gt, d = s["dice_at_0.5"];
  return (t >= 1 && t <= 9 ? 10 : t > 0.3 ? 3.5 : 0) + d;
}

/* ---- boot ---- */
async function boot() {
  initTheme();
  MANIFEST = await fetch(`${DATA_ROOT}/manifest.json`).then((r) => r.json());
  const want = readHash();
  const mSel = $("#modelSel");
  Object.entries(MANIFEST.models).forEach(([k, m]) => mSel.add(new Option(m.label, k)));
  modelKey = want.model && MANIFEST.models[want.model] ? want.model : mSel.value;
  mSel.value = modelKey;
  mSel.onchange = () => { modelKey = mSel.value; fillSamples(); };
  if (Number.isFinite(want.stage)) stageIdx = want.stage;
  fillSamples(want.sample);
  booting = false;
  writeHash();

  $("#prevBtn").onclick = () => step(-1);
  $("#nextBtn").onclick = () => step(1);
  $("#playBtn").onclick = togglePlay;
  $("#stageRange").oninput = (e) => selectStage(+e.target.value);
  $("#lightbox").onclick = () => $("#lightbox").classList.remove("on");
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.key === "ArrowRight") step(1);
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "Escape") $("#lightbox").classList.remove("on");
  });
}

function fillSamples(preferred) {
  const sSel = $("#sampleSel");
  sSel.innerHTML = "";
  const ordered = [...model().samples].sort((a, b) => instructiveness(b) - instructiveness(a));
  ordered.forEach((s) => sSel.add(new Option(`${s.name}  —  dice ${s["dice_at_0.5"]}`, s.name)));
  const hit = preferred && ordered.find((s) => s.name === preferred);
  sampleName = hit ? preferred : ordered[0].name;
  sSel.value = sampleName;
  sSel.onchange = () => { sampleName = sSel.value; onSampleChange(); };
  onSampleChange();
}

function onSampleChange() {
  stopPlay();
  const s = sample();
  $("#refInput").src = asset("input_mid.png");
  $("#refGT").src = asset(stAsset("head").gt || "gt_mask.png");
  $("#refPred").src = asset(stAsset("head").prob_heat || "prob_heat.png");
  $("#railStats").innerHTML = `
    <div><span>Model</span><b>${modelKey}</b></div>
    <div><span>Dice @ 0.50</span><b>${s["dice_at_0.5"]}</b></div>
    <div><span>Tumour in GT</span><b>${s.tumor_pct_gt}%</b></div>
    <div><span>Tumour predicted</span><b>${s.tumor_pct_pred}%</b></div>`;

  ARCH = Arch.render($("#arch"), stages(), s.stages, asset, (id) => {
    const i = stages().findIndex((x) => x.id === id);
    if (i >= 0) selectStage(i);
  });
  $("#stageRange").max = stages().length - 1;
  selectStage(Math.min(stageIdx, stages().length - 1));
}

/* ---- transport ---- */
function step(d) { selectStage((stageIdx + d + stages().length) % stages().length); }
function togglePlay() { playTimer ? stopPlay() : startPlay(); }
/* slider reads as SPEED: move right -> shorter delay -> faster */
function stepDelay() {
  const s = $("#speed");
  return (+s.min + +s.max) - +s.value;
}
function startPlay() {
  const b = $("#playBtn"); b.classList.add("on"); b.textContent = "⏸";
  if (ARCH) ARCH.setPlaying(true);
  const tick = () => {
    if (stageIdx >= stages().length - 1) { stopPlay(); return; }
    selectStage(stageIdx + 1);
    playTimer = setTimeout(tick, stepDelay());
  };
  playTimer = setTimeout(tick, stepDelay());
}
function stopPlay() {
  clearTimeout(playTimer); playTimer = null;
  const b = $("#playBtn"); b.classList.remove("on"); b.textContent = "▶";
  if (ARCH) ARCH.setPlaying(false);
}

/* ---- stage selection ---- */
function selectStage(i) {
  stageIdx = i;
  const st = stages()[i];
  $("#stageRange").value = i;
  $("#stageCount").textContent = `stage ${i + 1} / ${stages().length}`;
  $("#stageNow").textContent = st.title;
  $("#stageTitle").textContent = st.title;
  $("#stageKind").textContent = st.kind;
  $("#stageCaption").textContent = st.caption || "";

  const p = ARCH && ARCH.setActive(st.id);
  if (p) {
    const box = $("#arch");
    box.scrollTo({ left: p.x - box.clientWidth / 2, behavior: "smooth" });
  }
  renderIO(st);
  renderOps(st);
  renderViewer(st);
  writeHash();
}

function renderIO(st) {
  const parts = [];
  if (st.io) parts.push(st.io.replace(/->/g, "<b>→</b>"));
  if (st.params) parts.push(`${fmt(st.params)} params`);
  $("#ioLine").innerHTML = parts.join("  ·  ");
  $("#ioLine").style.display = parts.length ? "" : "none";
}

function renderOps(st) {
  const box = $("#opsBox");
  if (!st.ops || !st.ops.length) { box.style.display = "none"; return; }
  box.style.display = "";
  box.innerHTML = `<div class="ops-h">What this step computes</div>
    <ol>${st.ops.map((o) => `<li>${o}</li>`).join("")}</ol>`;
}

/* ---- lightbox ---- */
function zoomable(imgEl, caption) {
  imgEl.classList.add("zoomable");
  imgEl.onclick = () => {
    $("#lightboxImg").src = imgEl.src;
    $("#lightboxCap").textContent = caption || "";
    $("#lightbox").classList.add("on");
  };
}

/* ---- viewer ---- */
function cell(k, v, cls = "") {
  return `<div class="cell"><div class="k">${k}</div><div class="v ${cls}">${v}</div></div>`;
}

function renderViewer(st) {
  const v = $("#viewer"), stats = $("#stageStats");
  v.innerHTML = ""; stats.innerHTML = "";
  const a = stAsset(st.id);

  if (st.kind === "input") {
    v.innerHTML = `<div class="viewrow">
      <div class="viewcol frame-lg smooth"><h3>2.5D stack · Z-1 / Z / Z+1 → R / G / B</h3>
        <img src="${asset(a.stack || "input_stack.png")}"></div>
      <div class="viewcol frame-md"><h3>middle slice — the target</h3>
        <img src="${asset(a.mid || "input_mid.png")}"></div></div>`;
    return;
  }

  if (st.kind === "encoder" || st.kind === "bottleneck" || st.kind === "decoder") {
    v.innerHTML = `<div class="viewrow">
      <div class="viewcol frame-lg"><h3>feature-map channels
        (${(a.channels_shown || []).length} of ${a.shape ? a.shape[0] : "?"})
        <span class="zoomhint">· click to enlarge</span></h3>
        <img id="montage" src="${asset(a.montage)}"></div>
      <div class="viewcol frame-md"><h3>peak activation (strongest channel / pixel)</h3>
        <img id="peak" src="${asset(a.thumb)}">
        ${a.se_gains ? `<h3 style="margin-top:12px">Squeeze-Excitation channel gains</h3>
        <canvas id="seChart" width="250" height="92"
          style="width:250px;height:92px;image-rendering:auto"></canvas>
        <div class="note">One bar per channel (0–1). Tall = the network amplified
        that channel for this slice; short = suppressed it.</div>` : ""}
      </div></div>`;
    zoomable($("#montage"), `${st.title} — feature-map channels`);
    zoomable($("#peak"), `${st.title} — peak activation`);
    stats.innerHTML =
      cell("Tensor shape", a.shape ? `${a.shape[0]} ch · ${a.shape[1]}×${a.shape[2]}` : "—")
      + (a.se_gains ? cell("SE gain range",
        `${Math.min(...a.se_gains).toFixed(2)} – ${Math.max(...a.se_gains).toFixed(2)}`) : "");
    if (a.se_gains) drawSEChart($("#seChart"), a.se_gains);
    return;
  }

  if (st.kind === "attention") {
    v.innerHTML = `<div class="viewrow">
      <div class="viewcol frame-md"><h3>attention ψ over the input slice</h3>
        <canvas id="attCanvas" width="256" height="256"></canvas>
        <div class="slider"><label>overlay opacity <b id="attVal">0.60</b></label>
          <input type="range" id="attOpacity" min="0" max="1" step="0.01" value="0.6"></div></div>
      <div class="viewcol frame-md"><h3>raw ψ map · 0 = ignore, 1 = keep</h3>
        <img id="psiImg" src="${asset(a.heat)}"></div></div>`;
    setupAttention(a);
    zoomable($("#psiImg"), `${st.title} — raw ψ map`);
    stats.innerHTML = cell("ψ > 0.5 coverage", `${(a.coverage * 100).toFixed(1)}%`)
      + cell("Peak ψ", a.peak);
    return;
  }

  if (st.kind === "head") {
    v.innerHTML = `<div class="viewrow">
      <div class="viewcol frame-md smooth"><h3>tumour probability (sigmoid)</h3>
        <img id="probImg" src="${asset(a.prob_heat || "prob_heat.png")}"></div>
      <div class="viewcol frame-md"><h3>prediction vs ground truth</h3>
        <canvas id="segCanvas" width="256" height="256"></canvas>
        <div class="legend-inline"><span class="tp">hit</span><span class="fp">false +</span>
          <span class="fn">missed</span><span class="flip">flips near threshold</span></div></div>
      <div class="viewcol"><h3>probability histogram</h3>
        <canvas id="histCanvas" width="300" height="150"></canvas>
        <div class="note" id="histNote"></div></div></div>
    <div class="slider"><label>mask threshold <b id="thrVal">0.50</b></label>
      <input type="range" id="thr" min="0.02" max="0.98" step="0.01" value="0.5"></div>`;
    zoomable($("#probImg"), "tumour probability");
    stats.innerHTML =
      `<div class="cell"><div class="k">Live Dice</div>
        <div class="v dicebig" id="diceLive">–</div><div class="k" id="diceDelta"></div></div>
       <div class="cell"><div class="k">Mask pixels</div><div class="v" id="cMask">–</div></div>
       <div class="cell"><div class="k">Hits / False+ / Missed</div><div class="v" id="cTFN">–</div></div>`;
    setupHead(a);
    return;
  }
}

/* ---- SE bar chart ---- */
function drawSEChart(cv, gains) {
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const n = gains.length, bw = W / n;
  gains.forEach((g, i) => {
    ctx.fillStyle = `hsl(${200 - g * 170} 75% ${42 + g * 18}%)`;
    ctx.fillRect(i * bw, H - g * (H - 3), Math.max(1, bw - 0.5), g * (H - 3));
  });
}

/* ---- attention blend ---- */
function setupAttention(a) {
  const cv = $("#attCanvas"), ctx = cv.getContext("2d");
  const base = new Image(), heat = new Image();
  let ready = 0;
  const draw = () => {
    const op = +$("#attOpacity").value;
    $("#attVal").textContent = op.toFixed(2);
    ctx.globalAlpha = 1; ctx.drawImage(base, 0, 0, 256, 256);
    ctx.globalAlpha = op; ctx.drawImage(heat, 0, 0, 256, 256);
    ctx.globalAlpha = 1;
  };
  base.onload = heat.onload = () => { if (++ready === 2) draw(); };
  base.src = asset("input_mid.png");
  heat.src = asset(a.heat);
  $("#attOpacity").oninput = draw;
}

/* ---- head: threshold + histogram ---- */
function setupHead(a) {
  const seg = $("#segCanvas"), ctx = seg.getContext("2d");
  const hist = $("#histCanvas"), hctx = hist.getContext("2d");
  const off = document.createElement("canvas"); off.width = off.height = 256;
  const octx = off.getContext("2d");
  const baseImg = new Image(), probImg = new Image(), gtImg = new Image();
  let ready = 0, baseData, probData, gtData, bins, binMax, band = 0;
  const baseDice = sample()["dice_at_0.5"];
  const grab = (img) => {
    octx.clearRect(0, 0, 256, 256); octx.drawImage(img, 0, 0, 256, 256);
    return octx.getImageData(0, 0, 256, 256).data;
  };
  const buildHist = () => {
    const NB = 60; bins = new Array(NB).fill(0); band = 0;
    for (let i = 0; i < 65536; i++) {
      const p = probData[i * 4] / 255;
      bins[Math.min(NB - 1, Math.floor(p * NB))]++;
      if (p > 0.05 && p < 0.95) band++;
    }
    binMax = Math.max(...bins);
  };
  const drawHist = (thr) => {
    const W = hist.width, H = hist.height, NB = bins.length, bw = W / NB;
    hctx.clearRect(0, 0, W, H);
    const lmax = Math.log(binMax + 1);
    for (let i = 0; i < NB; i++) {
      const h = (Math.log(bins[i] + 1) / lmax) * (H - 14);
      const mid = (i + 0.5) / NB;
      hctx.fillStyle = mid >= thr ? "rgba(45,150,140,.9)" : "rgba(140,150,165,.5)";
      hctx.fillRect(i * bw, H - 12 - h, Math.max(1, bw - 0.6), h);
    }
    const x = thr * W;
    hctx.strokeStyle = "#ef4444"; hctx.lineWidth = 1.5;
    hctx.beginPath(); hctx.moveTo(x, 0); hctx.lineTo(x, H - 12); hctx.stroke();
    hctx.fillStyle = "#8b95a6"; hctx.font = "10px sans-serif";
    hctx.fillText("0", 1, H - 2); hctx.fillText("probability", W / 2 - 24, H - 2);
    hctx.fillText("1", W - 7, H - 2);
  };
  const render = () => {
    const t = +$("#thr").value, thr = t * 255;
    $("#thrVal").textContent = t.toFixed(2);
    const out = ctx.createImageData(256, 256);
    let inter = 0, pSum = 0, gSum = 0, tp = 0, fp = 0, fn = 0, flip = 0;
    for (let i = 0; i < 65536; i++) {
      const pv = probData[i * 4];
      const p = pv >= thr ? 1 : 0, g = gtData[i * 4] > 127 ? 1 : 0;
      inter += p & g; pSum += p; gSum += g;
      const near = Math.abs(pv - thr) < 26;
      const b = baseData[i * 4];
      let r = b, gg = b, bl = b;
      if (near && (p || g)) { r = 250; gg = 204; bl = 21; flip++; }
      else if (p && g) { r = 34; gg = 197; bl = 94; tp++; }
      else if (p && !g) { r = 239; gg = 68; bl = 68; fp++; }
      else if (!p && g) { r = 59; gg = 130; bl = 246; fn++; }
      out.data[i * 4] = r; out.data[i * 4 + 1] = gg; out.data[i * 4 + 2] = bl; out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    const dice = (2 * inter + 1e-6) / (pSum + gSum + 1e-6);
    $("#diceLive").textContent = dice.toFixed(3);
    const d = dice - baseDice, dd = $("#diceDelta");
    dd.textContent = `${d >= 0 ? "+" : ""}${d.toFixed(3)} vs threshold 0.50`;
    dd.className = "k " + (d >= 0 ? "delta-pos" : "delta-neg");
    $("#cMask").textContent = fmt(pSum);
    $("#cTFN").textContent = `${fmt(tp)} / ${fmt(fp)} / ${fmt(fn)}`;
    drawHist(t);
    $("#histNote").textContent =
      `This network is very confident: only ${fmt(band)} of 65,536 pixels sit `
      + `between 0.05 and 0.95, so the threshold mostly nudges the tumour `
      + `boundary. The yellow pixels are the ones flipping right now (${flip}).`;
  };
  baseImg.onload = probImg.onload = gtImg.onload = () => {
    if (++ready < 3) return;
    baseData = grab(baseImg); probData = grab(probImg); gtData = grab(gtImg);
    buildHist(); render();
  };
  baseImg.src = asset("input_mid.png");
  probImg.src = asset(a.prob || "prob.png");
  gtImg.src = asset(a.gt || "gt_mask.png");
  $("#thr").oninput = render;
}

boot();

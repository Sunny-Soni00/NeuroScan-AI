/* arch.js — horizontal animated pipeline of the whole network.
   Non-attention stages are laid left→right as image tiles; attention gates
   are diamonds on the skip arcs above. A pulse + playhead travel along the
   flow line so playing it feels like watching the signal move. */
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const KIND_COLOR = {
  input: "var(--faint)", encoder: "var(--enc)", bottleneck: "var(--bott)",
  attention: "var(--att)", decoder: "var(--dec)", head: "var(--head)",
};

function el(name, attrs = {}, parent = null) {
  const n = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function shortLabel(st) {
  if (st.kind === "input") return "Input";
  if (st.kind === "head") return "Head";
  if (st.kind === "bottleneck") return "Bottleneck";
  const m = String(st.id).match(/(\d+)$/);
  return (st.kind === "encoder" ? "Enc " : "Dec ") + (m ? m[1] : "");
}

const Arch = {
  /* returns { setActive(id), setPlaying(bool), pos: {id -> {x,y}} } */
  render(container, stages, sampleStages, assetFn, onSelect) {
    container.innerHTML = "";
    const chain = stages.filter((s) => s.kind !== "attention");
    const atts = stages.filter((s) => s.kind === "attention");

    const NW = 92, NH = 66, GAP = 30, PAD = 26;
    const idxOf = {}; chain.forEach((s, i) => (idxOf[s.id] = i));
    const cx = (i) => PAD + i * (NW + GAP) + NW / 2;

    // skip arcs: encoder <-> decoder chain node with equal spatial
    const arcs = [];
    chain.filter((s) => s.kind === "encoder").forEach((e) => {
      const d = chain.find((x) => x.kind === "decoder" && x.spatial === e.spatial);
      if (!d) return;
      const ie = idxOf[e.id], id = idxOf[d.id];
      const span = Math.abs(id - ie);
      const gate = atts.find((a) => a.spatial === e.spatial);
      arcs.push({ ie, id, span, gate });
    });
    const maxSpan = arcs.reduce((m, a) => Math.max(m, a.span), 1);
    const arcSpace = 34 + maxSpan * 13;
    const rowTop = arcSpace + 6;
    const cyRow = rowTop + NH / 2;
    const W = PAD * 2 + chain.length * NW + (chain.length - 1) * GAP;
    const H = rowTop + NH + 40;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, role: "img" }, container);
    const defs = el("defs", {}, svg);
    const arrow = el("marker", { id: "ar", viewBox: "0 0 10 10", refX: 8, refY: 5,
      markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--line2)" }, arrow);

    // ---- flow line (behind everything) ----
    const x0 = cx(0), x1 = cx(chain.length - 1);
    el("line", { class: "flow-base", x1: x0, y1: cyRow, x2: x1, y2: cyRow,
      "marker-end": "url(#ar)" }, svg);
    const dash = el("line", { class: "flow-dash", x1: x0, y1: cyRow, x2: x1, y2: cyRow }, svg);

    // ---- skip arcs ----  (encoder -> matching decoder, up and over)
    const gArc = el("g", {}, svg);
    arcs.forEach((a) => {
      const xa = cx(a.ie), xb = cx(a.id);
      const apexY = rowTop - (10 + a.span * 12);
      const midX = (xa + xb) / 2;
      a.path = el("path", {
        class: "skip",
        d: `M ${xa} ${rowTop} Q ${midX} ${apexY} ${xb} ${rowTop}`,
      }, gArc);
      // the gate marks where the (filtered) skip enters its decoder node
      const gx = xb, gy = rowTop - 13;
      a.gatePt = { x: gx, y: gy };
      if (a.gate) {
        const dm = el("path", { class: "att-diamond", "data-id": a.gate.id,
          d: diamond(gx, gy, 7) }, gArc);
        el("title", {}, dm).textContent = a.gate.title;
        dm.addEventListener("click", () => onSelect(a.gate.id));
        el("text", { class: "att-label", x: gx + 11, y: gy + 3 }, gArc).textContent = "ψ";
      }
    });

    // ---- nodes ----
    const pos = {};
    const gNodes = el("g", {}, svg);
    chain.forEach((st, i) => {
      const x = PAD + i * (NW + GAP), y = rowTop;
      pos[st.id] = { x: x + NW / 2, y: cyRow };
      const g = el("g", { class: "node", "data-id": st.id, color: undefined }, gNodes);
      g.style.color = KIND_COLOR[st.kind];

      const tile = el("g", { class: "tile" }, g);
      const clip = "cl-" + st.id;
      el("rect", { x, y, width: NW, height: NH, rx: 9 }, el("clipPath", { id: clip }, defs));
      const a = sampleStages[st.id] || {};
      const img = a.thumb || (st.kind === "input" ? a.mid : null)
        || (st.kind === "head" ? a.prob_heat || "prob_heat.png" : null);
      if (img) {
        el("image", { class: "node-img", href: assetFn(img), x, y, width: NW, height: NH,
          preserveAspectRatio: "xMidYMid slice", "clip-path": `url(#${clip})` }, tile);
      }
      el("rect", { class: "node-border", x, y, width: NW, height: NH, rx: 9,
        stroke: "currentColor" }, tile);
      // ping ring (one-shot on activate)
      el("rect", { class: "ping-ring", x, y, width: NW, height: NH, rx: 9,
        fill: "none", stroke: "currentColor", "stroke-opacity": 0 }, tile);

      el("text", { class: "node-label", x: x + NW / 2, y: y + NH + 15,
        "text-anchor": "middle" }, g).textContent = shortLabel(st);
      if (st.kind !== "input" && st.kind !== "head" && st.channels) {
        el("text", { class: "node-sub", x: x + NW / 2, y: y + NH + 27,
          "text-anchor": "middle" }, g).textContent = `${st.spatial}²·${st.channels}c`;
      }
      g.addEventListener("click", () => onSelect(st.id));
    });

    // attention playhead positions = its gate diamond
    atts.forEach((at) => {
      const arc = arcs.find((a) => a.gate && a.gate.id === at.id);
      if (arc) pos[at.id] = { ...arc.gatePt };
    });

    // ---- playhead ----
    const play = el("g", { class: "playhead" }, svg);
    el("circle", { class: "playhead-glow", r: 13, cx: 0, cy: 0 }, play);
    el("circle", { class: "playhead-core", r: 6, cx: 0, cy: 0 }, play);

    // ---- legend ----
    const legend = document.getElementById("archLegend");
    if (legend) {
      const items = [["encoder", "Encoder"], ["bottleneck", "Bottleneck"]];
      if (atts.length) items.push(["attention", "Attention gate"]);
      items.push(["decoder", "Decoder"], ["head", "Head"]);
      legend.innerHTML = items.map(([k, l]) =>
        `<span><i style="background:${KIND_COLOR[k]}"></i>${l}</span>`).join("")
        + `<span><i style="background:var(--faint)"></i>skip copy</span>`;
    }

    let lastPing = null;
    const setActive = (id) => {
      svg.querySelectorAll(".node,.att-diamond").forEach((n) =>
        n.classList.toggle("active", n.dataset.id === id));
      // light the skip arc whose gate is active, else the arc feeding an active decoder
      arcs.forEach((a) => {
        const lit = a.gate && a.gate.id === id
          || chain[a.id] && chain[a.id].id === id;
        a.path.classList.toggle("lit", !!lit);
      });
      const p = pos[id];
      if (p) play.setAttribute("transform", `translate(${p.x} ${p.y})`);
      // one-shot ping on the active node tile
      if (lastPing) lastPing.classList.remove("ping");
      const node = svg.querySelector(`.node[data-id="${id}"] .ping-ring`);
      if (node) { node.classList.add("ping"); lastPing = node; }
      return p;
    };
    const setPlaying = (on) => svg.classList.toggle("playing", on);

    return { setActive, setPlaying, pos };
  },
};

function diamond(cx, cy, r) {
  return `M${cx},${cy - r} L${cx + r},${cy} L${cx},${cy + r} L${cx - r},${cy} Z`;
}
window.Arch = Arch;

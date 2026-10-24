/**
 * graphview.js — "Graph view for two users" on the PPR tab.
 *
 * Display conventions copied from the author's week6/graph.js: d3 v7.9.0 (vendored), a two-column bipartite
 * layout (users left, programmes right), Okabe–Ito colours (users #0072B2, programmes #E69F00, edges #b9c0ca),
 * fill = node kind or score on a single-hue ramp, radius = degree or score, hover tooltip, legend, and a
 * node cap (week 6 used 120; default here 60 because two neighbourhoods are drawn). Additions for this exam:
 *   - two seed users A and B instead of one, so their neighbourhoods and shared programmes are visible;
 *   - edge width = edge weight (the new PPR input: binary / log(1+n) / n full views), A's and B's own edges
 *     drawn in their own colours;
 *   - the score seed's PPR Top-5 programmes get a dark ring and a "#rank title" label.
 * Node selection (deterministic): A and B; each one's top-H training programmes by weight; each one's PPR
 * Top-5; then the other users ranked by how many of the selected programmes they are linked to (ties: total
 * weight, then index) until the node cap is reached. Edges: every training edge among the selected nodes.
 * The view is read-only; it never changes the context or the model.
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;
  const THEME = {
    width: 1000, height: 520, margin: { top: 52, right: 250, bottom: 26, left: 170 },
    userColor: "#0072B2", movieColor: "#E69F00", edgeColor: "#b9c0ca", edgeA: "#0072B2", edgeB: "#CC79A7",
    nodeStroke: "#ffffff", recStroke: "#1c2128", labelColor: "#1c2128", noteColor: "#5b6472",
    minRadius: 5, maxRadius: 20, rampLight: [232, 243, 252], rampDark: [8, 48, 96], tooltipId: "graph-tooltip",
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const ramp = (t) => {
    const x = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
    const m = (a, b) => Math.round(a + (b - a) * x);
    return `rgb(${m(THEME.rampLight[0], THEME.rampDark[0])}, ${m(THEME.rampLight[1], THEME.rampDark[1])}, ${m(THEME.rampLight[2], THEME.rampDark[2])})`;
  };
  const short = (s, n = 30) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  /** Training adjacency of a context (full views only), cached per context. */
  const ADJ = new WeakMap();
  function adjacency(ctx) {
    if (ADJ.has(ctx)) return ADJ.get(ctx);
    const byItem = new Map(), degU = new Int32Array(ctx.nUsers), degI = new Int32Array(ctx.nItems);
    for (let u = 0; u < ctx.nUsers; u++) for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) {
      if (ctx.aN[k] === 0) continue;
      const i = ctx.aItem[k];
      if (!byItem.has(i)) byItem.set(i, []);
      byItem.get(i).push([u, ctx.aN[k]]);
      degU[u]++; degI[i]++;
    }
    const a = { byItem, degU, degI };
    ADJ.set(ctx, a);
    return a;
  }
  const weightFn = (w) => (w === "count" ? (n) => n : w === "log" ? (n) => Math.log1p(n) : () => 1);

  /** Personalized PageRank scores for the two seeds (TF.js when ready, else the float64 reference). */
  async function seedScores(ctx, opts, seeds) {
    const out = new Map();
    if (root.tf && root.ExamPPRTF) {
      const m = await root.ExamPPRTF.makePPRTF(ctx, opts);
      await m.prepare(seeds, 1200, true);
      for (const u of seeds) {
        const itemRow = m.itemRow(u), userRow = m.userScores.get(u);
        const item = new Float64Array(ctx.nItems), user = new Float64Array(ctx.nUsers);
        if (itemRow) for (let i = 0; i < ctx.nItems; i++) { const j = m.blocks.iMap[i]; if (j >= 0) item[i] = itemRow[j]; }
        if (userRow) for (let v = 0; v < ctx.nUsers; v++) { const j = m.blocks.uMap[v]; if (j >= 0) user[v] = userRow[j]; }
        out.set(u, { item, user, scores: m.scoreUser(u), source: `TF.js (${root.tf.getBackend()})` });
      }
      const conv = { ...m.conv };
      m.dispose();
      out.conv = conv;
    } else {
      const m = root.ExamPPR.makePPR(ctx, opts);
      for (const u of seeds) {
        const r = m.run(u);
        out.set(u, { item: r.p.slice(ctx.nUsers), user: r.p.slice(0, ctx.nUsers), scores: m.scoreUser(u), source: "float64 reference implementation" });
      }
      out.conv = { ...m.conv };
    }
    return out;
  }

  function buildView(ctx, opts, st, sc) {
    const E = ctx.E, adj = adjacency(ctx), wOf = weightFn(opts.weight);
    const seeds = [st.a, st.b];
    const hist = (u) => C.history(ctx, u).filter((h) => h.n > 0);
    const progs = new Map();                                    // item -> {inA, inB, recA, recB}
    const mark = (i, key, val) => { if (!progs.has(i)) progs.set(i, {}); progs.get(i)[key] = val; };
    seeds.forEach((u, s) => {
      hist(u).slice(0, st.perUser).forEach((h) => mark(h.item, s ? "inB" : "inA", h.n));
      const top = C.topK(ctx, sc.get(u).scores);
      top.items.forEach((i, r) => { if (r < top.fromMethod && adj.degI[i] > 0) mark(i, s ? "recB" : "recA", r + 1); });
    });
    const progList = [...progs.keys()];
    const others = new Map();                                   // user -> [links, weight]
    for (const i of progList) for (const [u, n] of adj.byItem.get(i) || []) {
      if (u === st.a || u === st.b) continue;
      const o = others.get(u) || [0, 0];
      o[0]++; o[1] += wOf(n);
      others.set(u, o);
    }
    const room = Math.max(0, st.maxNodes - 2 - progList.length);
    const otherList = [...others.entries()].sort((x, y) => y[1][0] - x[1][0] || y[1][1] - x[1][1] || x[0] - y[0]);
    const pickedOthers = otherList.slice(0, room).map((x) => x[0]);
    const users = [st.a, ...pickedOthers, st.b];             // A at the top, B at the bottom of the user column
    const userSet = new Set(users), progSet = new Set(progList);
    const scoreSeed = st.scoreSeed === "B" ? st.b : st.a;
    const S = sc.get(scoreSeed);
    const nodes = [];
    users.forEach((u) => nodes.push({ key: "u" + u, kind: "user", idx: u, label: u === st.a ? "A" : u === st.b ? "B" : "", id: E.users[u], degree: adj.degU[u], score: S.user[u] }));
    // programme order: shared by A and B, A only, B only, recommended only
    const group = (p) => (p.inA && p.inB ? 0 : p.inA ? 1 : p.inB ? 2 : 3);
    progList.sort((x, y) => group(progs.get(x)) - group(progs.get(y)) || S.item[y] - S.item[x] || x - y)
      .forEach((i) => { const p = progs.get(i); nodes.push({ key: "i" + i, kind: "movie", idx: i, id: E.items.id[i], title: E.items.title[i] || "", category: E.items.category[i] || "",
        degree: adj.degI[i], score: S.item[i], inA: p.inA || 0, inB: p.inB || 0, rec: st.scoreSeed === "B" ? p.recB : p.recA, group: group(p) }); });
    const edges = [];
    let maxW = 0;
    for (const u of users) for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) {
      const i = ctx.aItem[k];
      if (ctx.aN[k] === 0 || !progSet.has(i)) continue;
      const w = wOf(ctx.aN[k]);
      maxW = Math.max(maxW, w);
      edges.push({ source: "u" + u, target: "i" + i, w, n: ctx.aN[k], seed: u === st.a ? "A" : u === st.b ? "B" : "" });
    }
    const reachable = others.size + 2 + progList.length;
    return { nodes, edges, maxW, omitted: reachable - nodes.length, userSet, scoreSeed, conv: sc.conv, source: S.source,
      shared: progList.filter((i) => progs.get(i).inA && progs.get(i).inB).length };
  }

  function draw(svgEl, view, st, E) {
    const d3 = root.d3;
    const { width, height, margin } = THEME;
    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();
    svg.attr("viewBox", `0 0 ${width} ${height}`).attr("preserveAspectRatio", "xMidYMid meet");
    const uw = width - margin.left - margin.right, uh = height - margin.top - margin.bottom;
    const leftX = margin.left + uw * 0.2, rightX = margin.left + uw * 0.8;
    const pos = new Map();
    const place = (list, x) => list.forEach((n, i) => pos.set(n.key, { x, y: list.length <= 1 ? margin.top + uh / 2 : margin.top + (uh * i) / (list.length - 1) }));
    place(view.nodes.filter((n) => n.kind === "user"), leftX);
    place(view.nodes.filter((n) => n.kind === "movie"), rightX);
    svg.append("text").attr("x", leftX).attr("y", 18).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", THEME.noteColor).text("users");
    svg.append("text").attr("x", rightX).attr("y", 18).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", THEME.noteColor).text("programmes");
    const byScore = st.colorBy === "score";
    const maxScore = Math.max(...view.nodes.map((n) => (Number.isFinite(n.score) ? n.score : 0)), 0);
    const maxDeg = Math.max(1, ...view.nodes.map((n) => n.degree || 0));
    const norm = (n) => (byScore ? (maxScore > 0 ? Math.min(1, (n.score || 0) / maxScore) : 0) : Math.min(1, (n.degree || 0) / maxDeg));
    const radius = (n) => THEME.minRadius + (THEME.maxRadius - THEME.minRadius) * Math.sqrt(norm(n));
    const ew = (e) => (st.edgeWidth === "weight" && view.maxW > 0 ? 0.6 + 4.4 * Math.sqrt(e.w / view.maxW) : 1);
    const edges = view.edges.slice().sort((a, b) => (a.seed ? 1 : 0) - (b.seed ? 1 : 0));   // seed edges on top
    svg.append("g").selectAll("line").data(edges).join("line")
      .attr("x1", (d) => pos.get(d.source).x).attr("y1", (d) => pos.get(d.source).y)
      .attr("x2", (d) => pos.get(d.target).x).attr("y2", (d) => pos.get(d.target).y)
      .attr("stroke", (d) => (d.seed === "A" ? THEME.edgeA : d.seed === "B" ? THEME.edgeB : THEME.edgeColor))
      .attr("stroke-opacity", (d) => (d.seed ? 0.85 : 0.3)).attr("stroke-width", ew);
    const g = svg.append("g").selectAll("g").data(view.nodes, (d) => d.key).join("g")
      .attr("transform", (d) => `translate(${pos.get(d.key).x},${pos.get(d.key).y})`).style("cursor", "default");
    g.each(function (d) {
      const node = d3.select(this);
      const r = radius(d);
      const fill = byScore ? ramp(norm(d)) : d.kind === "user" ? THEME.userColor : THEME.movieColor;
      node.append("circle").attr("r", r).attr("fill", fill)
        .attr("stroke", d.rec ? THEME.recStroke : d.label ? (d.label === "A" ? THEME.edgeA : THEME.edgeB) : THEME.nodeStroke)
        .attr("stroke-width", d.rec || d.label ? 2.5 : 1.5);
      if (d.label) node.append("text").attr("x", -(r + 6)).attr("y", 4).attr("text-anchor", "end").attr("font-size", 11).attr("fill", THEME.labelColor).text(`${d.label} · ${d.id}`);
      if (d.rec) node.append("text").attr("x", r + 6).attr("y", 4).attr("font-size", 11).attr("fill", THEME.labelColor).text(`#${d.rec} ${short(d.title)}`);
      const tip = d.kind === "user"
        ? `user ${d.id}${d.label ? " (" + d.label + ")" : ""}  |  degree ${d.degree}  |  PPR score ${Number(d.score || 0).toExponential(3)}`
        : `${d.title} #${d.id} · ${d.category}  |  degree ${d.degree}  |  PPR score ${Number(d.score || 0).toExponential(3)}` +
          `${d.inA ? "  |  A: " + d.inA + " full views" : ""}${d.inB ? "  |  B: " + d.inB + " full views" : ""}${d.rec ? "  |  Top-5 rank " + d.rec : ""}`;
      node.on("mouseenter", (ev) => showTip(ev, tip)).on("mousemove", (ev) => showTip(ev, tip)).on("mouseleave", hideTip);
    });
  }

  function showTip(ev, text) {
    let el = document.getElementById(THEME.tooltipId);
    if (!el) {
      el = document.createElement("div");
      el.id = THEME.tooltipId;
      el.setAttribute("role", "tooltip");
      Object.assign(el.style, { position: "absolute", pointerEvents: "none", background: "#1c2128", color: "#ffffff", padding: "0.35rem 0.5rem",
        borderRadius: "6px", fontSize: "0.78rem", lineHeight: "1.35", zIndex: "10", whiteSpace: "nowrap", display: "none" });
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.style.display = "block";
    el.style.left = (ev.pageX || 0) + 12 + "px";
    el.style.top = (ev.pageY || 0) + 12 + "px";
  }
  function hideTip() { const el = document.getElementById(THEME.tooltipId); if (el) el.style.display = "none"; }

  function legend(view, st) {
    const sw = (c, extra = "") => `<span class="legend-swatch" style="background:${c};${extra}"></span>`;
    const item = (s, t) => `<span class="legend-item">${s}<span>${esc(t)}</span></span>`;
    const items = [item(sw(THEME.userColor), "user node (left column)"), item(sw(THEME.movieColor), "programme node (right column)")];
    if (st.colorBy === "score") items.push(item(sw("", `background:linear-gradient(90deg, ${ramp(0)}, ${ramp(1)})`), `fill = PPR score from seed ${st.scoreSeed} (single-hue ramp)`));
    items.push(item("", st.colorBy === "score" ? "radius scales with score" : "radius scales with degree"));
    items.push(item(sw(THEME.edgeA), "edge of user A"), item(sw(THEME.edgeB), "edge of user B"), item(sw(THEME.edgeColor), "edge of another viewer"));
    items.push(item("", st.edgeWidth === "weight" ? "edge width = edge weight" : "uniform edge width"));
    items.push(item(sw("#ffffff", `border:2px solid ${THEME.recStroke};border-radius:50%`), `dark ring + label = PPR Top-5 of seed ${st.scoreSeed}`));
    items.push(item("", "programmes top-to-bottom: shared by A and B, A only, B only, recommended only"));
    if (view.omitted > 0) items.push(item("", `${view.omitted} reachable user node(s) omitted by the ${st.maxNodes}-node cap`));
    return items.join("");
  }

  /**
   * Mount the controls, canvas and legend into `container`.
   * @param {HTMLElement} container
   * @param {{ctx: object, getOpts: function(): object, defaults: {a:number, b:number}}} env
   */
  let LAST = null;                                             // last drawn graph: {ctx, opts, st, sc, view}

  /** Stand-alone SVG of the last graph (white background, legend and caption) in the given colour mode. */
  function snapshotSvg(colorBy) {
    if (!LAST) return null;
    const NS = "http://www.w3.org/2000/svg";
    const st = { ...LAST.st, colorBy };
    const view = buildView(LAST.ctx, LAST.opts, st, LAST.sc);
    const E = LAST.ctx.E;
    const svg = document.createElementNS(NS, "svg");
    draw(svg, view, st, E);
    const H = THEME.height + 92;
    svg.setAttribute("xmlns", NS);
    svg.setAttribute("width", THEME.width); svg.setAttribute("height", H);
    svg.setAttribute("viewBox", `0 0 ${THEME.width} ${H}`);
    svg.setAttribute("font-family", "Helvetica, Arial, sans-serif");
    const bg = document.createElementNS(NS, "rect");
    bg.setAttribute("width", THEME.width); bg.setAttribute("height", H); bg.setAttribute("fill", "#ffffff");
    svg.insertBefore(bg, svg.firstChild);
    const add = (tag, attrs, text) => { const el = document.createElementNS(NS, tag); for (const [k, x] of Object.entries(attrs)) el.setAttribute(k, x); if (text !== undefined) el.textContent = text; svg.appendChild(el); return el; };
    const y0 = THEME.height + 14;
    let x = 20;
    const item = (draw, text) => { draw(x); add("text", { x: x + 22, y: y0 + 4, "font-size": 12, fill: THEME.labelColor }, text); x += 30 + text.length * 6.4; };
    item((x) => add("circle", { cx: x + 7, cy: y0, r: 6, fill: THEME.userColor }), "user");
    item((x) => add("circle", { cx: x + 7, cy: y0, r: 6, fill: THEME.movieColor }), "programme");
    item((x) => add("line", { x1: x, y1: y0, x2: x + 16, y2: y0, stroke: THEME.edgeA, "stroke-width": 3 }), "edge of A");
    item((x) => add("line", { x1: x, y1: y0, x2: x + 16, y2: y0, stroke: THEME.edgeB, "stroke-width": 3 }), "edge of B");
    item((x) => add("line", { x1: x, y1: y0, x2: x + 16, y2: y0, stroke: THEME.edgeColor, "stroke-width": 3 }), "edge of another viewer");
    item((x) => add("circle", { cx: x + 7, cy: y0, r: 6, fill: "#ffffff", stroke: THEME.recStroke, "stroke-width": 2.5 }), `PPR Top-5 of seed ${st.scoreSeed}`);
    const enc = st.colorBy === "score" ? `fill and radius = PPR score from seed ${st.scoreSeed} (single-hue ramp)` : "fill = node kind, radius = degree";
    const ew = st.edgeWidth === "weight" ? `edge width = edge weight (${LAST.opts.weight})` : "uniform edge width";
    add("text", { x: 20, y: y0 + 30, "font-size": 12, fill: THEME.noteColor }, `${enc}; ${ew}; ${view.nodes.length} nodes, ${view.edges.length} edges` + (view.omitted > 0 ? `, ${view.omitted} reachable users omitted by the ${st.maxNodes}-node cap` : ""));
    add("text", { x: 20, y: y0 + 52, "font-size": 12, fill: THEME.noteColor },
      `Training graph, weeks 11–20. PPR α = ${LAST.opts.alpha}. A = ${E.users[st.a]}, B = ${E.users[st.b]}. Scores: ${view.source}.`);
    return { svg: new XMLSerializer().serializeToString(svg), width: THEME.width, height: H };
  }

  /** Data behind the last graph: state, nodes, edges and both users' PPR Top-5. */
  function exportData() {
    if (!LAST) return null;
    const { ctx, st, sc, view, opts } = LAST, E = ctx.E;
    const top = [];
    for (const [role, u] of [["A", st.a], ["B", st.b]]) {
      const t = C.topK(ctx, sc.get(u).scores), act = new Set(E.gtValid[u] || []);
      const seen = new Set(C.history(ctx, u).filter((h) => h.n > 0).map((h) => h.item));
      t.items.forEach((i, r) => top.push({ role, user_id: E.users[u], rank: r + 1, tv_show_id: E.items.id[i], title: E.items.title[i] || "", category: E.items.category[i] || "",
        score: r < t.fromMethod ? sc.get(u).scores[i] : "", slot: r < t.fromMethod ? "method" : "fallback", in_actual_21_30: act.has(i) ? 1 : 0, in_training_history: seen.has(i) ? 1 : 0 }));
    }
    return { state: { ...st, a: E.users[st.a], b: E.users[st.b] }, opts, source: view.source, conv: view.conv, shared: view.shared, omitted: view.omitted,
      nodes: view.nodes, edges: view.edges, top };
  }

  function mount(container, env) {
    const E = env.ctx.E, ctx = env.ctx;
    const views = new Map();
    for (const u of ctx.cohort) { let s = 0; for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) s += ctx.aN[k]; views.set(u, s); }
    const cohort = ctx.cohort.slice().sort((x, y) => views.get(y) - views.get(x) || x - y);
    const opt = (u, sel) => `<option value="${u}"${u === sel ? " selected" : ""}>${E.users[u]} · ${views.get(u)} full views</option>`;
    const userControl = (id, labelText, sel) => `<div class="control"><label for="${id}">${labelText}</label>
      <input type="search" id="${id}-search" placeholder="Search user ID" autocomplete="off" aria-controls="${id}">
      <select id="${id}">${cohort.map((u) => opt(u, sel)).join("")}</select></div>`;
    container.innerHTML = `
      <p class="hint">Choose two users of the evaluation cohort (default: the heavy and the light example user). The graph is the weeks 11–20 training graph
        with the PPR settings of step 2 (α and edge weight); PPR is run from both users. Display conventions follow week 6.</p>
      <div class="inline-controls">
        ${userControl("gv-a", "User A", env.defaults.a)}
        ${userControl("gv-b", "User B", env.defaults.b)}
        <div class="control"><label for="gv-color">Colour by</label><select id="gv-color"><option value="kind">node kind / degree</option><option value="score">PPR score</option></select></div>
        <div class="control"><label for="gv-seed">Score seed</label><select id="gv-seed"><option value="A">user A</option><option value="B">user B</option></select></div>
        <div class="control"><label for="gv-edge">Edge width</label><select id="gv-edge"><option value="weight">by edge weight</option><option value="uniform">uniform (week 6)</option></select></div>
        <div class="control"><label for="gv-per">Programmes per user</label><input type="number" id="gv-per" min="1" max="40" step="1" value="12"></div>
        <div class="control"><label for="gv-max">Max nodes</label><input type="number" id="gv-max" min="10" max="400" step="10" value="60"></div>
      </div>
      <div class="button-row"><button type="button" id="gv-draw">Draw graph for A and B</button></div>
      <div id="gv-controls-note" class="graph-controls"></div>
      <svg id="gv-svg" class="graph-view graph-view--tall" role="img" aria-label="Bipartite training subgraph of two users and their programmes"></svg>
      <div id="gv-legend" class="graph-legend"><p class="empty-state">Press “Draw graph for A and B”.</p></div>
      <div id="gv-table"></div>`;
    for (const id of ["gv-a", "gv-b"]) {
      container.querySelector(`#${id}-search`).addEventListener("input", (e) => {
        const q = e.target.value.trim();
        const sel = container.querySelector(`#${id}`), cur = Number(sel.value);
        const list = cohort.filter((u) => !q || E.users[u].includes(q) || u === cur);
        sel.innerHTML = list.slice(0, 300).map((u) => opt(u, cur)).join("");
      });
    }
    const state = () => ({
      a: Number(container.querySelector("#gv-a").value), b: Number(container.querySelector("#gv-b").value),
      colorBy: container.querySelector("#gv-color").value, scoreSeed: container.querySelector("#gv-seed").value,
      edgeWidth: container.querySelector("#gv-edge").value,
      perUser: Math.max(1, Number(container.querySelector("#gv-per").value) || 12), maxNodes: Math.max(10, Number(container.querySelector("#gv-max").value) || 60),
    });
    let cache = null;
    async function render(recompute) {
      const st = state();
      const note = container.querySelector("#gv-controls-note");
      if (!root.d3) { note.innerHTML = '<p class="empty-state">d3 (vendor/d3.min.js) did not load.</p>'; return; }
      if (st.a === st.b) { note.innerHTML = '<p class="empty-state">Choose two different users.</p>'; return; }
      const opts = env.getOpts();
      const key = JSON.stringify([st.a, st.b, opts.alpha, opts.weight, opts.tol, opts.maxIter]);
      if (recompute || !cache || cache.key !== key) {
        note.innerHTML = '<span class="hint">Running PPR from both users…</span>';
        cache = { key, sc: await seedScores(ctx, opts, [st.a, st.b]), opts };
      }
      const view = buildView(ctx, cache.opts, st, cache.sc);
      draw(container.querySelector("#gv-svg"), view, st, E);
      LAST = { ctx, opts: cache.opts, st, sc: cache.sc, view };
      container.querySelector("#gv-legend").innerHTML = legend(view, st);
      const c = view.conv || {};
      note.innerHTML = `<span class="hint">PPR α = ${cache.opts.alpha}, edge weight = ${esc(cache.opts.weight)}; scores from the ${esc(view.source)};
        converged ${c.runs - c.notConverged}/${c.runs}, mean iterations ${(c.iterSum / Math.max(1, c.runs)).toFixed(1)}.
        Drawn ${view.nodes.length} nodes and ${view.edges.length} edges; A and B share ${view.shared} of the drawn training programmes.</span>`;
      container.querySelector("#gv-table").innerHTML = topTable(st, cache.sc);
    }
    function topTable(st, sc) {
      const row = (u) => {
        const t = C.topK(ctx, sc.get(u).scores), act = new Set(E.gtValid[u] || []);
        const seen = new Set(C.history(ctx, u).filter((h) => h.n > 0).map((h) => h.item));
        return t.items.map((i, r) => `${act.has(i) ? '<span class="mark-hit">✓</span> ' : ""}${esc(E.items.title[i] || "")} <span class="hint">#${E.items.id[i]}${seen.has(i) ? " · ↺ seen" : ""}${r >= t.fromMethod ? " · fallback" : ""}</span>`);
      };
      const A = row(st.a), B = row(st.b);
      const ia = C.topK(ctx, sc.get(st.a).scores).items, ib = C.topK(ctx, sc.get(st.b).scores).items;
      const common = ia.filter((i) => ib.includes(i)).length;
      const apA = C.apk(E.gtValid[st.a] || [], ia), apB = C.apk(E.gtValid[st.b] || [], ib);
      return `<div class="scroll-x"><table class="data-table"><thead><tr><th>#</th><th>PPR Top-5 of A (AP@5 ${apA.toFixed(3)})</th><th>PPR Top-5 of B (AP@5 ${apB.toFixed(3)})</th></tr></thead><tbody>` +
        [0, 1, 2, 3, 4].map((r) => `<tr><td>${r + 1}</td><td>${A[r]}</td><td>${B[r]}</td></tr>`).join("") +
        `</tbody></table></div><p class="hint">${common} programme(s) appear in both lists. ✓ = in the user's actual weeks 21–30 top-5; ↺ seen = watched ≥ 80 % in weeks 11–20.</p>`;
    }
    container.querySelector("#gv-draw").addEventListener("click", () => render(true).catch((e) => { container.querySelector("#gv-controls-note").innerHTML = `<p class="error">${esc(e.message)}</p>`; }));
    for (const id of ["gv-color", "gv-seed", "gv-edge", "gv-per", "gv-max"]) container.querySelector(`#${id}`).addEventListener("change", () => { if (cache) render(false); });
    return { render, hasRender: () => LAST !== null && LAST.ctx === ctx, state };
  }

  root.ExamGraphView = { mount, buildView, snapshotSvg, exportData, THEME };
})(typeof window !== "undefined" ? window : globalThis);

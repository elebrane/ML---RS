/**
 * app.js — page wiring for the sweet.tv exam application (no framework).
 *
 * Tabs: 1 Data & protocol · 2 Popularity · 3 MF (TF.js) · 4 PPR (TF.js) · 5 Summary.
 * (Content-based filtering and association rules stay in js/content.js and js/rules.js as secondary methods; their
 * results are in results/summary.csv, they are not shown on the page.) Every method tab is laid out as numbered steps: what the method takes
 * and from where → settings → run on validation → the two fixed example users → validation sweep →
 * refit for the test weeks → export. The page calls exactly the functions tools/run_experiments.js
 * calls, so a browser run reproduces the saved batch results in results/ (the "Saved batch run" blocks).
 */
(function () {
  "use strict";
  const C = window.ExamCore, M = window.ExamMethods, E = window.EXAM;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const fmt = (x, d = 5) => (Number.isFinite(Number(x)) && x !== "" && x !== null ? Number(x).toFixed(d) : "—");
  const canon = (o) => JSON.stringify(Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b))));
  const yieldUI = () => new Promise((r) => setTimeout(r, 0));
  const S = { runs: {}, sweeps: {}, examples: {}, tests: {}, saved: { summary: {}, examples: {}, sweeps: {} }, checks: null, tfReady: false,
    models: {}, savedExamples: {}, exChoice: "default: P90 / P10 of training full views", exLists: null };
  const METHOD_KEYS = ["popularity", "mf", "ppr"];

  // ------------------------------------------------------------------ helpers
  function setStatus(msg, cls = "") { const el = $("status"); el.textContent = msg; el.className = "status " + cls; }

  function parseCsv(text) {
    const rows = []; let row = [], f = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true;
      else if (c === ",") { row.push(f); f = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(f); rows.push(row); row = []; f = ""; }
      else f += c;
    }
    if (f !== "" || row.length) { row.push(f); rows.push(row); }
    const [header, ...body] = rows.filter((r) => r.length > 1 || r[0] !== "");
    return body.map((r) => Object.fromEntries(header.map((h, j) => [h, r[j]])));
  }
  async function fetchCsv(path) {
    try { const r = await fetch(path, { cache: "no-store" }); return r.ok ? parseCsv(await r.text()) : null; } catch { return null; }
  }
  function table(header, rows, cls = "data-table") {
    return `<div class="scroll-x"><table class="${cls}"><thead><tr>${header.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table></div>`;
  }
  const tr = (cells, cls = "") => `<tr${cls ? ` class="${cls}"` : ""}>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  const label = (i) => C.itemLabel(E, i);

  async function saveFile(filename, content, encoding = "utf-8") {
    try {
      const r = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename, content, encoding }) });
      return r.ok ? (await r.json()).saved : null;
    } catch { return null; }
  }
  async function exportFiles(files, container) {
    const parts = [];
    const MIME = { json: "application/json", csv: "text/csv", svg: "image/svg+xml", png: "image/png" };
    for (const [name, content, encoding = "utf-8"] of files) {
      const saved = await saveFile(name, content, encoding);
      const body = encoding === "base64" ? Uint8Array.from(atob(content), (c) => c.charCodeAt(0)) : content;
      const url = URL.createObjectURL(new Blob([body], { type: MIME[name.split(".").pop()] || "application/octet-stream" }));
      parts.push(`<a href="${url}" download="${name}">${esc(name)}</a>${saved ? ` <span class="hint">saved to ${esc(saved)}</span>` : ` <span class="hint">(download only — start server.py to save into results/browser/)</span>`}`);
    }
    container.innerHTML = `<p class="download-links small">${parts.join("<br>")}</p>`;
  }

  // ------------------------------------------------------------------ method specs (steps 1-2)
  const sel = (name, labelText, options, hint, num = false) => ({ name, label: labelText, type: "select", options, hint, num });
  const numc = (name, labelText, step, hint, min = 0, max = undefined) => ({ name, label: labelText, type: "number", step, hint, min, max });
  const SPECS = {
    popularity: {
      what: `<p><b>Takes:</b> the training rows of weeks 11–20 from <code>window.EXAM.log</code> — per (user, programme, week) the number of airings the user watched for at least 80&nbsp;% of their length
        (built by <code>tools/build_interactions.py</code> from <code>dataset11-30.csv</code> joined to <code>export_arh_11-20-final.csv</code> on channel + interval overlap).
        <b>Candidates:</b> labelled programmes that air in weeks 21–30 (<code>export_arh_21-30-final.csv</code>, schedule only).</p>
        <p><b>Score → five ids:</b> one popularity count per programme; every user gets the same five highest counts (ties by programme id).
        This is the required reference, not one of the methods, and its validation-selected setting is the shared fallback of every method.</p>`,
      controls: [sel("count", "Count", ["views", "viewers"], "views = total full views; viewers = distinct users with a full view"), sel("recentWeeks", "Training weeks used", ["0", "2", "4"], "0 = all ten weeks; W = only the last W weeks", true)],
    },
    content: {
      what: `<p><b>Takes:</b> programme attributes from the schedule files — <code>tv_show_category</code>, <code>tv_show_genre_1..3</code>, decade of <code>year_of_production</code>,
        <code>director</code>/<code>actors</code> — plus two broadcast attributes from the same schedule: the channels a programme airs on and the share of its airings in each 3-hour slot
        (training + target window; schedule only, never who watched). The user profile is built from the user's full views in weeks 11–20.</p>
        <p><b>Score → five ids:</b> each attribute group is L2-normalised and multiplied by its weight, the item vector is L2-normalised;
        profile = (weighted) average of the vectors of watched programmes (course week 2); score = cosine(profile, programme) for every candidate; Top-5 by cosine,
        ties by the shared popularity rank. A user with no full view gets the fallback list.</p>`,
      controls: [numc("meta", "Weight: category / genres / decade", 0.5, "catalogue metadata"), numc("people", "Weight: directors / actors", 0.5, "people metadata"),
        numc("channel", "Weight: channels", 0.5, "where the programme airs"), numc("time", "Weight: time of day", 0.5, "3-hour airing slots"),
        sel("profile", "Profile weighting", ["log", "linear"], "watched programme weight = log(1+n) or n full views")],
    },
    rules: {
      what: `<p><b>Takes:</b> one transaction per (user, week) of weeks 11–20 = the set of programmes the user watched ≥ 80&nbsp;% that week (course week 4: a basket is a set).
        <b>same-week</b> = classic rules A → B inside one basket; <b>next-week</b> = time-window rules A(week t) → B(week t+1) of the same user, which also yields A → A (“watched it this week → watches it next week”).</p>
        <p><b>Score → five ids:</b> Apriori up to pairs with downward closure; keep rules with support ≥ min support, confidence ≥ min confidence and lift &gt; 1.
        For a user, the antecedents are the programmes watched in the last <i>lookback</i> training weeks; a candidate's score is the max (or sum) confidence of the rules that point to it.
        Candidates no rule reaches get no score, and the free slots are filled from the fallback list.</p>`,
      controls: [sel("mode", "Rule family", ["same-week", "next-week"], "transaction definition"), sel("minSupport", "Min support", ["0.0005", "0.001", "0.002", "0.005"], "share of transactions", true),
        sel("minConfidence", "Min confidence", ["0.05", "0.1", "0.2"], "P(B | A)", true), sel("agg", "Aggregate", ["max", "sum"], "over a user's antecedents"), sel("lookback", "Lookback (weeks)", ["1", "4", "10"], "antecedent window", true)],
    },
    mf: {
      what: `<p><b>Takes:</b> the user × programme matrix of weeks 11–20: preference p = 1 if the user watched the programme ≥ 80&nbsp;% at least once, confidence c = 1 + α·r
        (r = number of full views, or their hours). Model: user and programme embedding tables trained with <code>tf.train.adam</code> in TensorFlow.js (course week 5),
        but on the implicit-feedback loss Σ c(p − u·v)² + λ(‖U‖² + ‖V‖²) over <i>all</i> cells (Hu, Koren &amp; Volinsky, 2008) — an unwatched cell is a 0 with the lowest confidence, not a missing rating.
        Training controls as in HW5: <b>epochs</b>, <b>learning rate</b>, <b>L2 (λ)</b> and <b>batch size</b> — a mini-batch is a set of users (whole rows of the matrix); the penalty is applied to the batch's user rows only
        (HW5 lesson) plus the batch's share of ‖V‖², so one epoch's batch losses add up to the full loss.</p>
        <p><b>Score → five ids:</b> dot product u·v for every candidate that has a factor (watched by someone in training); Top-5 by dot product; programmes without a factor and users without views fall back to popularity.</p>`,
      controls: [sel("factors", "Factors k", ["16", "32", "64"], "embedding size", true), numc("lambda", "λ (L2)", 10, "penalty on all factors"), numc("alpha", "α (confidence)", 0.5, "c = 1 + α·r"),
        sel("signal", "Signal r", ["count", "hours"], "full views or their hours"), numc("epochs", "Epochs", 5, "passes over all users", 1),
        numc("batchSize", "Batch size (users)", 256, "users per mini-batch; ≥ number of users (≈4.7k) = one full batch per epoch", 1),
        numc("lr", "Learning rate", 0.01, "Adam", 0.001), numc("seed", "Seed", 1, "initialisation and batch order", 0)],
    },
    ppr: {
      what: `<p><b>Takes:</b> the bipartite user–programme graph of weeks 11–20 (edge = at least one full view; weight binary, log(1+n) or n), so two programmes are connected through their common viewers.
        TensorFlow.js propagation as in course week 6: two dense blocks (user→programme and programme→user, rows normalised), restart one-hot at the target user,
        p ← α·Qᵀp + (1 − α)·v with α = probability of following a link; stop at L1 change &lt; tol or maxIter.</p>
        <p><b>Score → five ids:</b> the programme block of the stationary vector (a walk probability, not a rating); Top-5 among candidates with positive mass.
        Users with no edge get the fallback list.</p>`,
      controls: [numc("alpha", "α (follow a link)", 0.05, "1 − α = restart probability", 0.05, 0.95), sel("weight", "Edge weight", ["binary", "log", "count"], "per (user, programme)"),
        numc("tol", "Tolerance (L1)", 1e-6, "per user row"), numc("maxIter", "Max iterations", 50, "cap; convergence is reported", 1)],
    },
  };

  function defaultsFor(key) {
    const saved = S.saved.summary[key];
    if (saved && saved.selected_settings) { try { return JSON.parse(saved.selected_settings); } catch { /* fall through */ } }
    return M.byKey(key).defaults;
  }
  function readOpts(key) {
    const opts = { ...M.byKey(key).defaults };
    for (const c of SPECS[key].controls) {
      const el = $(`${key}-${c.name}`);
      opts[c.name] = c.type === "number" || c.num ? Number(el.value) : el.value;
    }
    return opts;
  }
  function setOpts(key, opts) {
    for (const c of SPECS[key].controls) if (opts[c.name] !== undefined) $(`${key}-${c.name}`).value = String(opts[c.name]);
  }

  // ------------------------------------------------------------------ rendering a method tab
  function renderMethodPanel(key) {
    const m = M.byKey(key), spec = SPECS[key];
    const controls = spec.controls.map((c) => {
      const id = `${key}-${c.name}`;
      const input = c.type === "select"
        ? `<select id="${id}">${c.options.map((o) => `<option value="${o}">${o}</option>`).join("")}</select>`
        : `<input type="number" id="${id}" step="${c.step}" ${c.min !== undefined ? `min="${c.min}"` : ""} ${c.max !== undefined ? `max="${c.max}"` : ""}>`;
      return `<div class="control"><label for="${id}">${esc(c.label)}</label>${input}<p class="hint">${esc(c.hint)}</p></div>`;
    }).join("");
    let stepNo = 0;
    const n = () => ++stepNo;
    const graphStep = () => (key === "ppr" ? `
        <div class="step"><h3><span class="step-no">${n()}</span>Graph view for two users (week-6 display)</h3><div id="ppr-graph"></div></div>` : "");
    const tfNote = key === "mf" || key === "ppr" ? `<p class="hint">TensorFlow.js backend: <span class="tf-backend">—</span>. Full-matrix training / propagation can take from a few seconds (WebGL) to a few minutes (CPU).</p>` : "";
    $(`panel-${key}`).innerHTML = `
      <div class="panel">
        <h2>${esc(m.title)}</h2>
        <div class="step"><h3><span class="step-no">${n()}</span>What the method takes, from where, and how it produces exactly five ids</h3>${spec.what}<div id="${key}-input"></div></div>
        <div class="step"><h3><span class="step-no">${n()}</span>Settings</h3>
          <p class="hint">Pre-filled with the setting selected on validation by the saved batch run (highest MAP@5 on the 1,150-user cohort).</p>
          <div class="inline-controls">${controls}</div>${tfNote}
          <div class="button-row"><button type="button" id="${key}-reset" class="secondary">Reset to selected setting</button></div></div>
        <div class="step"><h3><span class="step-no">${n()}</span>Fit on weeks 11–20, evaluate MAP@5 on weeks 21–30</h3>
          <div class="button-row"><button type="button" id="${key}-run">Run on validation</button></div>
          <div id="${key}-result"><p class="empty-state">Not run in this session yet.</p></div><div id="${key}-diag"></div></div>
        <div class="step"><h3><span class="step-no">${n()}</span>Two example users (heavy and light viewer)</h3>
          <div id="${key}-expick"></div>
          <div id="${key}-examples"><p class="empty-state">Loading the saved batch run…</p></div></div>${graphStep()}
        <div class="step"><h3><span class="step-no">${n()}</span>Validation sweep (setting selection)</h3>
          <p class="hint">Same grid as <code>tools/run_experiments.js</code>; the best setting is the highest cohort MAP@5. The saved batch sweep is shown until you run one here.</p>
          <div class="button-row"><button type="button" id="${key}-sweep" class="secondary">Run the sweep in the browser</button></div>
          <div id="${key}-sweep-out"><p class="empty-state">Loading the saved sweep…</p></div></div>
        <div class="step"><h3><span class="step-no">${n()}</span>Export</h3>
          <div class="button-row"><button type="button" id="${key}-export" class="secondary">Save this tab's results (CSV)</button>${key === "ppr" ? `<button type="button" id="ppr-report">Save PPR for the report (settings, data, figures)</button>` : ""}</div><div id="${key}-export-out"></div></div>
        <div class="step"><h3><span class="step-no">${n()}</span>Kaggle submission — step by step</h3>
          <p class="hint">Rebuilds <code>submission_${key}.csv</code> from scratch with the settings of step 2: every stage is shown with its numbers, the file is checked
            against the template's rules, saved to <code>results/browser/</code> and compared with the batch-run file <code>results/submissions/submission_${key}.csv</code>.</p>
          <div class="button-row"><button type="button" id="${key}-test">Build submission_${key}.csv step by step</button></div>
          <ol id="${key}-test-out" class="pipeline"></ol></div>
      </div>`;
    setOpts(key, defaultsFor(key));
    $(`${key}-reset`).addEventListener("click", () => setOpts(key, defaultsFor(key)));
    $(`${key}-run`).addEventListener("click", () => guard(() => runValidation(key)));
    $(`${key}-sweep`).addEventListener("click", () => guard(() => runSweep(key)));
    $(`${key}-test`).addEventListener("click", () => guard(() => buildSubmissionStepwise(key)));
    $(`${key}-export`).addEventListener("click", () => guard(() => exportMethod(key)));
    $(`${key}-input`).innerHTML = inputSummary(key);
    $(`${key}-expick`).innerHTML = examplePicker(key);
    wireExamplePicker(key);
    if (key === "ppr" && window.ExamGraphView) {
      S.graph = window.ExamGraphView.mount($("ppr-graph"), { ctx: S.vctx, getOpts: () => readOpts("ppr"), defaults: { a: S.ex.heavy.u, b: S.ex.light.u } });
      $("ppr-report").addEventListener("click", () => guard(exportPprReport));
    }
    renderSavedSweep(key);
    renderExamples(key);
  }

  function inputSummary(key) {
    const v = S.vctx;
    const rows = v.rowItem.length, users = new Set(); let full = 0;
    for (let u = 0; u < v.nUsers; u++) if (v.rowPtr[u + 1] > v.rowPtr[u]) users.add(u);
    for (const n of v.rowN) full += n;
    return `<div class="metric-row">
      <div class="metric"><b>${rows.toLocaleString("en-US")}</b><span>(user, programme, week) rows, weeks 11–20</span></div>
      <div class="metric"><b>${full.toLocaleString("en-US")}</b><span>full views (≥ 80 %) in weeks 11–20</span></div>
      <div class="metric"><b>${users.size.toLocaleString("en-US")}</b><span>users with training rows</span></div>
      <div class="metric"><b>${v.candidateList.length.toLocaleString("en-US")}</b><span>candidates (air in weeks 21–30)</span></div>
      <div class="metric"><b>${v.cohort.length.toLocaleString("en-US")}</b><span>evaluation cohort users</span></div></div>`;
  }

  let busy = false;
  async function guard(fn) {
    if (busy) { setStatus("Another computation is running — please wait.", "warning"); return; }
    busy = true;
    document.querySelectorAll("button").forEach((b) => { if (!b.classList.contains("tab-button")) b.disabled = true; });
    try { await fn(); } catch (e) { console.error(e); setStatus(`Error: ${e.message}`, "error"); }
    finally { busy = false; document.querySelectorAll("button").forEach((b) => { b.disabled = false; }); refreshTfButtons(); }
  }

  async function buildModel(key, ctx, opts, users, onProgress) {
    const m = M.byKey(key);
    let model;
    if (key === "mf") model = await window.ExamMFTF.trainMFTF(ctx, opts, onProgress ? (s, l) => { onProgress(`epoch ${s}/${opts.epochs}, loss ${l.toFixed(1)}`); return yieldUI(); } : null);
    else model = await m.make(ctx, opts);
    if (model.prepare) await model.prepare(users);
    return model;
  }

  function exampleData(model, ctx) {
    const out = {};
    for (const [role, { u }] of Object.entries(S.ex)) {
      const s = model.scoreUser(u);
      const { items, fromMethod } = C.topK(ctx, s);
      out[role] = { u, recs: items.map((i, r) => ({ i, score: s && s[i] !== C.NO_SCORE ? s[i] : null, src: r < fromMethod ? "method" : "fallback" })),
        actual: E.gtValid[u], hist: C.history(ctx, u).filter((h) => h.n > 0).slice(0, 5) };
    }
    return out;
  }

  // ---------------------------------------------------------------- example-user pair (shared by every tab)
  function viewsOf(u) { let s = 0; for (let k = S.vctx.aggPtr[u]; k < S.vctx.aggPtr[u + 1]; k++) s += S.vctx.aN[k]; return s; }
  /** Heavy = top 25 % of the cohort by weeks 11–20 full views, light = bottom 25 % (includes users with no history). */
  function buildExampleLists() {
    const all = S.vctx.cohort.map((u) => ({ u, s: viewsOf(u) })).sort((x, y) => x.s - y.s || x.u - y.u);
    const q = Math.ceil(all.length * 0.25);
    S.exLists = { light: all.slice(0, q), heavy: all.slice(all.length - q).reverse(), lightMax: all[q - 1].s, heavyMin: all[all.length - q].s };
  }
  const pairKey = () => `${S.ex.heavy.u}/${S.ex.light.u}`;
  function examplePicker(key) {
    const opt = (x, sel) => `<option value="${x.u}"${x.u === sel ? " selected" : ""}>${E.users[x.u]} · ${x.s} full views</option>`;
    return `<div class="inline-controls">
      <div class="control"><label for="${key}-ex-heavy">Heavy viewer — top 25 % (≥ ${S.exLists.heavyMin} full views)</label><select id="${key}-ex-heavy" class="ex-heavy">${S.exLists.heavy.map((x) => opt(x, S.ex.heavy.u)).join("")}</select></div>
      <div class="control"><label for="${key}-ex-light">Light viewer — bottom 25 % (≤ ${S.exLists.lightMax} full views)</label><select id="${key}-ex-light" class="ex-light">${S.exLists.light.map((x) => opt(x, S.ex.light.u)).join("")}</select></div></div>
      <div class="button-row"><button type="button" class="secondary ex-rand" data-role="heavy">Random heavy</button>
        <button type="button" class="secondary ex-rand" data-role="light">Random light</button>
        <button type="button" class="secondary ex-rand" data-role="both">Random pair</button>
        <button type="button" class="secondary ex-default">Default (P90 / P10)</button></div>
      <p class="hint">One pair for every tab (exam §7: the same two users for all methods). Lists: the ${S.vctx.cohort.length.toLocaleString("en-US")}-user cohort ranked by full views in weeks 11–20. Current choice: <span class="ex-choice">${esc(S.exChoice)}</span>.</p>`;
  }
  function wireExamplePicker(key) {
    const box = $(`${key}-expick`);
    box.querySelector(".ex-heavy").addEventListener("change", (e) => guard(() => setExampleUsers(Number(e.target.value), S.ex.light.u, "chosen from the list")));
    box.querySelector(".ex-light").addEventListener("change", (e) => guard(() => setExampleUsers(S.ex.heavy.u, Number(e.target.value), "chosen from the list")));
    box.querySelectorAll(".ex-rand").forEach((b) => b.addEventListener("click", () => guard(() => {
      const pick = (list) => list[Math.floor(Math.random() * list.length)].u;
      const role = b.dataset.role;
      const h = role === "light" ? S.ex.heavy.u : pick(S.exLists.heavy);
      const l = role === "heavy" ? S.ex.light.u : pick(S.exLists.light);
      return setExampleUsers(h, l, "random draw from the heavy / light lists");
    })));
    box.querySelector(".ex-default").addEventListener("click", () => guard(() => {
      const d = C.exampleUsers(S.vctx);
      return setExampleUsers(d.heavy.u, d.light.u, "default: P90 / P10 of training full views");
    }));
  }
  async function setExampleUsers(h, l, choice) {
    S.ex = { heavy: { u: h, s: viewsOf(h) }, light: { u: l, s: viewsOf(l) } };
    S.exChoice = choice;
    for (const k of METHOD_KEYS) {
      const box = $(`${k}-expick`);
      if (!box) continue;
      box.querySelector(".ex-heavy").value = String(h);
      box.querySelector(".ex-light").value = String(l);
      box.querySelector(".ex-choice").textContent = choice;
    }
    const ga = document.getElementById("gv-a"), gb = document.getElementById("gv-b");   // keep the PPR graph view in sync
    if (ga && [...ga.options].some((o) => o.value === String(h))) ga.value = String(h);
    if (gb && [...gb.options].some((o) => o.value === String(l))) gb.value = String(l);
    for (const k of METHOD_KEYS) { await computeExamples(k, false); renderExamples(k); }
    renderSummary();
    setStatus(`Example users: heavy ${E.users[h]} (${S.ex.heavy.s} full views), light ${E.users[l]} (${S.ex.light.s}) — ${choice}.`, "success");
  }
  /** Top-5 of one method for the current pair: reuse this session's model, else the saved batch rows, else fit now. */
  async function computeExamples(key, allowTrain) {
    const pk = pairKey();
    if (S.examples[key] && S.examples[key].pair === pk && !S.examples[key].pending) return;
    if (!S.models[key] && S.savedExamples[key] && S.savedExamples[key].pair === pk) { S.examples[key] = S.savedExamples[key]; return; }
    let cached = S.models[key];
    if (!cached) {
      if ((key === "mf" && !allowTrain) || ((key === "mf" || key === "ppr") && !S.tfReady)) { S.examples[key] = { pending: true, pair: pk }; return; }
      setStatus(`${key}: fitting on weeks 11–20 for the example users…`);
      await yieldUI();
      const opts = readOpts(key);
      const model = await buildModel(key, S.vctx, opts, [S.ex.heavy.u, S.ex.light.u], (msg) => setStatus(`${key}: ${msg}`));
      cached = S.models[key] = { opts, model };
    } else if (cached.model.prepare) await cached.model.prepare([S.ex.heavy.u, S.ex.light.u]);
    S.examples[key] = { source: `this session, settings ${JSON.stringify(cached.opts)}`, pair: pk, data: exampleData(cached.model, S.vctx) };
  }
  function cacheModel(key, opts, model) {
    const old = S.models[key];
    if (old && old.model !== model && old.model.dispose) old.model.dispose();
    S.models[key] = { opts, model };
  }

  async function runValidation(key) {
    const opts = readOpts(key);
    setStatus(`${key}: fitting on weeks 11–20…`);
    await yieldUI();
    const t0 = C.nowMs();
    const model = await buildModel(key, S.vctx, opts, S.vctx.cohort, (msg) => setStatus(`${key}: ${msg}`));
    const t1 = C.nowMs();
    const ev = C.evaluate(S.vctx, (u) => model.scoreUser(u));
    const t2 = C.nowMs();
    const m = M.byKey(key);
    S.runs[key] = { opts, map: ev.map, users: ev.users, padded: ev.paddedSlots, usersPadded: ev.usersPadded, fitS: (t1 - t0) / 1000, scoreS: (t2 - t1) / 1000, stats: m.stats ? m.stats(model) : {}, at: new Date().toISOString() };
    cacheModel(key, opts, model);                         // kept so a new example pair needs no refit
    S.examples[key] = { source: `this session, settings ${JSON.stringify(opts)}`, pair: pairKey(), data: exampleData(model, S.vctx) };
    renderResult(key);
    $(`${key}-diag`).innerHTML = diagnostics(key, model);
    renderExamples(key);
    renderSummary();
    setStatus(`${key}: MAP@5 = ${ev.map.toFixed(5)} on ${ev.users} users (${((t2 - t0) / 1000).toFixed(1)} s).`, "success");
  }

  function renderResult(key) {
    const r = S.runs[key], saved = S.saved.summary[key];
    const same = saved && canon(JSON.parse(saved.selected_settings)) === canon(r.opts);
    $(`${key}-result`).innerHTML = `<div class="metric-row">
      <div class="metric"><b>${fmt(r.map)}</b><span>MAP@5, ${r.users} cohort users</span></div>
      <div class="metric"><b>${r.padded}</b><span>slots filled by fallback (${r.usersPadded} users)</span></div>
      <div class="metric"><b>${r.fitS.toFixed(1)} s / ${r.scoreS.toFixed(1)} s</b><span>fit / score (${esc(window.tf ? tf.getBackend() : "no tf")})</span></div>
      ${saved ? `<div class="metric"><b>${fmt(saved.valid_map5_cohort_1150)}</b><span>saved batch run (selected setting)</span></div>` : ""}</div>
      <p class="hint">Settings: <code>${esc(JSON.stringify(r.opts))}</code>${same ? ` — identical to the saved batch run's selected setting, so the two MAP@5 values should agree (float32 TF.js backends may differ in the last digits).` : ""}</p>`;
  }

  function diagnostics(key, model) {
    if (key === "popularity") {
      const s = window.ExamPopularity.popularityScores(S.vctx, readOpts(key));
      const top = C.topK(S.vctx, s).items;
      return table(["Rank", "Programme", "Category", "Count"], top.map((i, r) => tr([r + 1, `${esc(label(i).title)} <span class="hint">#${label(i).id}</span>`, esc(label(i).category), s[i]])));
    }
    if (key === "rules") {
      const st = model.mined.stats;
      const all = [...model.mined.rules.values()].flat();
      const byLift = all.slice().sort((a, b) => b.lift - a.lift).slice(0, 8);
      const byConf = all.slice().sort((a, b) => b.confidence - a.confidence).slice(0, 8);
      const rows = (list) => list.map((r) => tr([`${esc(label(r.a).title)} → ${esc(label(r.b).title)}${r.a === r.b ? ' <span class="hint">(self)</span>' : ""}`, r.countA, r.count, r.countB, fmt(r.support, 4), fmt(r.confidence, 3), fmt(r.lift, 2)]));
      const head = ["Rule", "count(A)", "count(A,B)", "count(B)", "Support", "Confidence", "Lift"];
      return `<p class="small">Transactions N = ${st.transactions.toLocaleString("en-US")}, min count = ${st.minCount}, frequent antecedents ${st.frequentItemsA}, frequent consequents ${st.frequentItemsB}, frequent pairs ${st.frequentPairs}, rules kept (lift &gt; 1) ${st.rules}.</p>
        <h4>Highest-lift rules</h4>${table(head, rows(byLift))}<h4>Highest-confidence rules</h4>${table(head, rows(byConf))}`;
    }
    if (key === "content") {
      const u = S.ex.heavy.u;
      model.profile(u);
      const P = model.getProfile();
      const names = [...model.vectors.featIndex.entries()].map(([k, j]) => ({ k, w: P[j] })).filter((x) => x.w > 0).sort((a, b) => b.w - a.w).slice(0, 12);
      return `<p class="small">${model.vectors.nFeat.toLocaleString("en-US")} features. Strongest profile features of the heavy example user:</p>` +
        table(["Feature", "Profile weight"], names.map((x) => tr([esc(x.k), fmt(x.w, 4)])));
    }
    if (key === "mf") {
      return `<p class="small">Materialised ${model.nU.toLocaleString("en-US")} users × ${model.nI.toLocaleString("en-US")} programmes, ${model.positives.toLocaleString("en-US")} positive cells;
        ${model.fullBatch ? "one full batch per epoch" : `${model.batches} mini-batches of ≤ ${model.batchSize} users per epoch`}, ${model.epochs} epochs = ${model.updates} Adam updates; ‖V‖ = ${fmt(model.factorNorm, 3)}.</p>` +
        table(["Epoch", "Loss (sum over the epoch's batches)"], model.history.map((h) => tr([h.epoch ?? h.step, h.loss.toFixed(1)])));
    }
    if (key === "ppr") {
      const c = model.conv;
      return `<p class="small">Graph: ${model.blocks.nU.toLocaleString("en-US")} user nodes + ${model.blocks.nI.toLocaleString("en-US")} programme nodes, ${model.blocks.edges.toLocaleString("en-US")} edges, no zero-degree node.
        Converged for ${c.runs - c.notConverged} of ${c.runs} users; mean iterations ${(c.iterSum / Math.max(1, c.runs)).toFixed(1)}; largest final L1 residual ${c.maxResidual.toExponential(2)}.</p>`;
    }
    return "";
  }

  function exampleTables(data) {
    const blocks = [];
    for (const role of ["heavy", "light"]) {
      const d = data[role];
      if (!d) continue;
      const act = new Set(d.actual), hist = new Set(d.hist.map((h) => h.item));
      const rec = d.recs.map((x, r) => tr([r + 1, `${esc(label(x.i).title)} <span class="hint">#${label(x.i).id}</span>`, esc(label(x.i).category),
        x.score === null ? "—" : Number(x.score).toPrecision(4), (act.has(x.i) ? '<span class="mark-hit">✓ hit</span> ' : "") + (hist.has(x.i) ? '<span class="mark-hist">↺ seen</span>' : "") + (x.src === "fallback" ? ' <span class="hint">fallback</span>' : "")], act.has(x.i) ? "hit" : ""));
      const ap = C.apk(d.actual, d.recs.map((x) => x.i));
      blocks.push(`<div><h4>${role === "heavy" ? "Heavy" : "Light"} viewer <code>${E.users[d.u]}</code> — AP@5 = ${ap.toFixed(3)}</h4>
        ${table(["#", "Recommended", "Category", "Score", ""], rec)}
        <p class="small"><b>Actually watched most, weeks 21–30:</b> ${d.actual.map((i) => `${esc(label(i).title)} <span class="hint">#${label(i).id}</span>`).join("; ")}</p>
        <p class="small"><b>Most watched in training, weeks 11–20:</b> ${d.hist.map((h) => `${esc(label(h.item).title)} (${h.n})`).join("; ")}</p></div>`);
    }
    return `<div class="examples-grid">${blocks.join("")}</div>`;
  }

  function renderExamples(key) {
    const ex = S.examples[key];
    const el = $(`${key}-examples`);
    if (!el) return;
    if (ex && ex.pending) {
      el.innerHTML = `<p class="empty-state">${key === "mf" ? "MF has not been trained in this session for this pair; computing it trains the model on weeks 11–20 (seconds on WebGL, a few minutes on WASM)." : "TensorFlow.js is not available."}</p>
        <div class="button-row"><button type="button" class="secondary" id="${key}-ex-compute">Compute for these users</button></div>`;
      $(`${key}-ex-compute`).addEventListener("click", () => guard(async () => { await computeExamples(key, true); renderExamples(key); renderSummary(); }));
      return;
    }
    el.innerHTML = ex ? `<p class="hint">Source: ${esc(ex.source)}. ✓ hit = in the user's actual weeks 21–30 top-5; ↺ seen = watched ≥ 80 % in weeks 11–20.</p>${exampleTables(ex.data)}`
      : `<p class="empty-state">No saved batch examples found (open the page through server.py) — press “Run on validation”.</p>`;
  }

  function examplesFromSaved(rows) {
    const idx = new Map(E.items.id.map((id, i) => [String(id), i]));
    const out = {};
    for (const role of ["heavy", "light"]) {
      const rs = rows.filter((r) => r.user_role === role);
      if (!rs.length) continue;
      const u = E.users.indexOf(rs[0].user_id);
      out[role] = { u, recs: rs.filter((r) => r.list === "recommended").map((r) => ({ i: idx.get(r.tv_show_id), score: r.score === "" ? null : Number(r.score), src: r.slot_source })),
        actual: E.gtValid[u], hist: rs.filter((r) => r.list === "training_top5_weeks_11_20").map((r) => ({ item: idx.get(r.tv_show_id), n: Number(r.score) })) };
    }
    return out;
  }

  // ------------------------------------------------------------------ sweep (same staged logic as the runner)
  async function runSweep(key) {
    const m = M.byKey(key);
    const results = new Map();
    const order = [];
    const out = $(`${key}-sweep-out`);
    for (let st = 1; st <= (m.stages || 1); st++) {
      let prev = null;
      if (st > 1) prev = order.filter((o) => o.stage === st - 1).map((o) => results.get(o.id)).reduce((a, b) => (b.map > a.map ? b : a));
      const grid = m.grid(st === 1 ? null : { ...prev.opts, stage: st });
      for (const opts of grid) {
        const id = canon(opts);
        if (!order.some((o) => o.id === id)) order.push({ id, stage: st });
        if (results.has(id)) continue;
        setStatus(`${key} sweep: stage ${st}, setting ${order.length} — ${JSON.stringify(opts)}`);
        await yieldUI();
        const t0 = C.nowMs();
        const model = await buildModel(key, S.vctx, opts, S.vctx.cohort);
        const ev = C.evaluate(S.vctx, (u) => model.scoreUser(u));
        results.set(id, { opts, stage: st, map: ev.map, padded: ev.paddedSlots, seconds: (C.nowMs() - t0) / 1000, stats: m.stats ? m.stats(model) : {} });
        if (model.dispose) model.dispose();
        out.innerHTML = sweepTable(order.filter((o) => results.has(o.id)).map((o) => results.get(o.id)), "this session (running…)");
      }
    }
    const list = order.map((o) => results.get(o.id));
    const best = list.reduce((a, b) => (b.map > a.map ? b : a));
    S.sweeps[key] = { list, best };
    out.innerHTML = sweepTable(list, "this session", best);
    setOpts(key, best.opts);
    setStatus(`${key} sweep done: best MAP@5 ${best.map.toFixed(5)} with ${JSON.stringify(best.opts)} (settings updated).`, "success");
  }

  function sweepTable(list, source, best = null) {
    const keys = [...new Set(list.flatMap((r) => Object.keys(r.opts)))];
    const top = list.slice().sort((a, b) => b.map - a.map);
    const rows = top.map((r) => tr([r.stage ?? "", ...keys.map((k) => esc(r.opts[k])), fmt(r.map), r.padded ?? "", r.seconds !== undefined ? r.seconds.toFixed(1) : ""], best && r === best ? "hit" : ""));
    return `<p class="hint">Source: ${esc(source)} — ${list.length} settings, sorted by MAP@5 (highlighted = selected).</p>
      <details${list.length <= 12 ? " open" : ""}><summary>Show all ${list.length} settings</summary>${table(["Stage", ...keys, "MAP@5", "Fallback slots", "Seconds"], rows)}</details>`;
  }

  async function renderSavedSweep(key) {
    const rows = S.saved.sweeps[key] || await fetchCsv(`results/validation/${key}_sweep.csv`);
    const el = $(`${key}-sweep-out`);
    if (!rows) { el.innerHTML = `<p class="empty-state">Saved sweep not available (open the page through server.py).</p>`; return; }
    S.saved.sweeps[key] = rows;
    const optCols = Object.keys(rows[0]).filter((c) => !["method", "config", "stage", "map5_cohort_1150", "users", "padded_slots", "users_padded", "fit_s", "score_s", "diagnostics", "selected", "on_final_search_path"].includes(c));
    const list = rows.map((r) => ({ stage: r.stage, opts: Object.fromEntries(optCols.map((c) => [c, r[c]])), map: Number(r.map5_cohort_1150), padded: r.padded_slots, seconds: Number(r.fit_s) + Number(r.score_s), selected: r.selected === "1" }));
    el.innerHTML = sweepTable(list, "saved batch run (results/validation/" + key + "_sweep.csv)", list.find((r) => r.selected));
  }

  // ------------------------------------------------------------------ test refit + export
  /**
   * Build the Kaggle file for one method, showing every stage with its numbers (for presenting the work):
   * template → training weeks 11–30 → candidates on air in weeks 31–42 → fit → Top-5 + fallback →
   * write + format checks → save + compare with the batch-run file. Same functions as tools/run_experiments.js.
   */
  async function buildSubmissionStepwise(key) {
    const m = M.byKey(key), opts = readOpts(key), ctx = S.tctx, out = $(`${key}-test-out`);
    const N = (x) => Number(x).toLocaleString("en-US");
    const T0 = C.nowMs();
    const titles = ["Template: who has to be predicted", "Training data: weeks 11–30 of the view log", "Candidates: programmes on air in weeks 31–42",
      `Fit — ${m.title}, weeks 11–30`, "Rank: Top-5 per user, free slots from the shared fallback", "Write the file in template order and check the format",
      "Save and compare with the batch-run file"];
    out.innerHTML = titles.map((t, i) => `<li id="${key}-ps-${i}"><span class="st">◻</span><b>${i + 1}. ${esc(t)}</b><div class="small ps-body"></div></li>`).join("");
    const set = async (i, state, html) => {
      const li = $(`${key}-ps-${i}`);
      li.className = state;
      li.querySelector(".st").textContent = { run: "⏳", ok: "✓", fail: "✗" }[state] || "◻";
      if (html !== undefined) li.querySelector(".ps-body").innerHTML = html;
      await yieldUI();
    };
    const hasHist = (u) => { for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) if (ctx.aN[k] > 0) return true; return false; };

    await set(0, "run");
    await set(0, "ok", `${N(E.testUsers.length)} users, in the row order of the competition's <code>submission.csv</code> (template rows <code>user_id,0 0 0 0 0</code>);
      every one of them is present in <code>dataset11-30.csv</code>.`);

    await set(1, "run");
    let full = 0;
    for (const x of ctx.rowN) full += x;
    const withHist = E.testUsers.filter(hasHist).length;
    await set(1, "ok", `${N(ctx.rowItem.length)} (user, programme, week) rows from weeks ${ctx.train[0]}–${ctx.train[1]} → ${N(full)} full views
      (an airing watched for ≥ 80 % of its length, the organiser's rule). ${N(withHist)} of ${N(E.testUsers.length)} test users have at least one full view;
      ${N(E.testUsers.length - withHist)} have none and will receive the fallback list.`);

    await set(2, "run");
    const seenItems = new Set();
    for (let k = 0; k < ctx.aItem.length; k++) if (ctx.aN[k] > 0) seenItems.add(ctx.aItem[k]);
    const candSeen = ctx.candidateList.filter((i) => seenItems.has(i)).length;
    await set(2, "ok", `${N(ctx.candidateList.length)} labelled programmes air in weeks 31–42 (<code>export_arh_31-42-final.csv</code>, <code>tv_show_id ≠ 0</code>);
      ${N(candSeen)} of them were watched by someone in weeks 11–30. Only the schedule of the test weeks is read — never who watched.`);

    await set(3, "run", `Settings: <code>${esc(JSON.stringify(opts))}</code>`);
    const t1 = C.nowMs();
    let model;
    if (key === "mf") model = await window.ExamMFTF.trainMFTF(ctx, opts, (st, l) => set(3, "run", `Settings: <code>${esc(JSON.stringify(opts))}</code> — epoch ${st}/${opts.epochs}, loss ${l.toFixed(1)}`));
    else model = await m.make(ctx, opts);
    const fitS = (C.nowMs() - t1) / 1000;
    const fitInfo = {
      popularity: () => "one popularity count per candidate programme (the same list for every user).",
      content: () => `${N(model.vectors.nFeat)} content features per programme (category, genres, decade, people, channels, time of day); one profile per user.`,
      rules: () => { const x = model.mined.stats; return `${N(x.transactions)} ${x.mode} transactions, ${N(x.frequentPairs)} frequent pairs, ${N(x.rules)} rules kept (lift &gt; 1).`; },
      mf: () => `${N(model.nU)} users × ${N(model.nI)} programmes, ${N(model.positives)} positive cells; ${model.epochs} epochs × ${model.batches} batch(es) = ${N(model.updates)} Adam updates; final epoch loss ${model.finalLoss.toFixed(1)}.`,
      ppr: () => `bipartite graph: ${N(model.blocks.nU)} user nodes + ${N(model.blocks.nI)} programme nodes, ${N(model.blocks.edges)} edges weighted by ${esc(opts.weight)}.`,
    }[key];
    await set(3, "ok", `Settings: <code>${esc(JSON.stringify(opts))}</code> → ${fitInfo()} (${fitS.toFixed(1)} s, ${esc(window.tf ? tf.getBackend() : "no tf")}).`);

    await set(4, "run", key === "ppr" ? `Propagating PPR from each of the ${N(E.testUsers.length)} users…` : "Scoring every candidate for every user…");
    const t2 = C.nowMs();
    if (model.prepare) await model.prepare(E.testUsers);
    const rows = [], lists = [];
    let padded = 0, allFallback = 0;
    for (const u of E.testUsers) {
      const s = model.scoreUser(u);
      const { items, fromMethod } = C.topK(ctx, s);
      lists.push(items);
      padded += 5 - fromMethod;
      if (fromMethod === 0) allFallback++;
      items.forEach((i, r) => rows.push([E.users[u], r + 1, ...Object.values(label(i)), s && s[i] !== C.NO_SCORE ? +s[i].toPrecision(6) : "", r < fromMethod ? "method" : "fallback"]));
    }
    const conv = key === "ppr" ? ` PPR converged for ${N(model.conv.runs - model.conv.notConverged)} of ${N(model.conv.runs)} users (mean ${(model.conv.iterSum / Math.max(1, model.conv.runs)).toFixed(1)} iterations, largest residual ${model.conv.maxResidual.toExponential(2)}).` : "";
    await set(4, "ok", `Per user: candidates sorted by the method's score (ties: shared popularity rank, then id), first five kept.
      ${N(E.testUsers.length * 5 - padded)} of ${N(E.testUsers.length * 5)} slots come from the method, ${N(padded)} from the fallback
      (popularity: distinct viewers in the last 4 training weeks); ${N(allFallback)} users are entirely on the fallback.${conv} (${((C.nowMs() - t2) / 1000).toFixed(1)} s)`);

    await set(5, "run");
    const text = C.toCsv(["user_id", "tv_show_id"], E.testUsers.map((u, j) => [E.users[u], lists[j].map((i) => E.items.id[i]).join(" ")]));
    const lines = text.trim().split("\n");
    const body = lines.slice(1).map((l) => l.split(","));
    const checks = [
      ["header is <code>user_id,tv_show_id</code>", lines[0] === "user_id,tv_show_id"],
      [`${N(E.testUsers.length)} data rows`, body.length === E.testUsers.length],
      ["users in the template's order", body.every((r, j) => r[0] === E.users[E.testUsers[j]])],
      ["exactly five space-separated ids per user", body.every((r) => r[1].split(" ").length === 5)],
      ["five different ids per user", body.every((r) => new Set(r[1].split(" ")).size === 5)],
      ["no placeholder id 0", body.every((r) => !r[1].split(" ").includes("0"))],
      ["every id airs in weeks 31–42", lists.every((l) => l.every((i) => ctx.isCandidate[i]))],
    ];
    const okAll = checks.every((c) => c[1]);
    await set(5, okAll ? "ok" : "fail", `<ul class="checks">${checks.map(([t, ok]) => `<li>${ok ? '<span class="mark-hit">✓</span>' : "✗"} ${t}</li>`).join("")}</ul>`);
    if (model.dispose) model.dispose();

    await set(6, "run", '<div class="ps-links"></div><div class="ps-compare"></div>');
    const li = $(`${key}-ps-6`);
    await exportFiles([[`submission_${key}.csv`, text], [`${key}_test_top5.csv`, C.toCsv(["user_id", "rank", "tv_show_id", "title", "category", "score", "slot_source"], rows)]], li.querySelector(".ps-links"));
    let cmp = "Batch-run file not available (open the page through server.py).";
    try {
      const r = await fetch(`results/submissions/submission_${key}.csv`, { cache: "no-store" });
      if (r.ok) {
        const ref = await r.text();
        const a = text.trim().split("\n"), b = ref.trim().split("\n");
        const same = a.filter((l, j) => l === b[j]).length - 1;
        const saved = S.saved.summary[key] ? S.saved.summary[key].selected_settings : null;
        const sameSettings = saved && canon(JSON.parse(saved)) === canon(opts);
        cmp = `Compared with <code>results/submissions/submission_${key}.csv</code> (built by <code>tools/run_experiments.js</code>): ${N(same)} of ${N(b.length - 1)} rows identical${text === ref ? " — byte-identical file" : ""}.` +
          (sameSettings ? "" : " The settings on screen differ from the batch run's selected setting, so differences are expected.");
      }
    } catch { /* keep the default message */ }
    const preview = E.testUsers.slice(0, 5).map((u, j) => tr([`<code>${E.users[u]}</code>`, lists[j].map((i) => `${esc(label(i).title)} <span class="hint">#${label(i).id}</span>`).join("; ")]));
    li.querySelector(".ps-compare").innerHTML = `<p class="small">${cmp}</p>${table(["user_id", "five programmes (rank order)"], preview)}`;
    await set(6, okAll ? "ok" : "fail");
    S.tests[key] = { opts, rows, submission: text, padded, seconds: (C.nowMs() - T0) / 1000 };
    setStatus(`${key}: submission_${key}.csv built step by step in ${((C.nowMs() - T0) / 1000).toFixed(1)} s — ${okAll ? "all format checks passed" : "a format check failed"}.`, okAll ? "success" : "error");
  }

  async function exportMethod(key) {
    const files = [];
    const r = S.runs[key];
    if (r) files.push([`${key}_validation.csv`, C.toCsv(["method", "settings", "map5_cohort", "users", "padded_slots", "fit_s", "score_s", "diagnostics", "backend", "run_at"],
      [[key, JSON.stringify(r.opts), r.map.toFixed(6), r.users, r.padded, r.fitS.toFixed(2), r.scoreS.toFixed(2), JSON.stringify(r.stats), window.tf ? tf.getBackend() : "", r.at]])]);
    if (S.sweeps[key]) files.push([`${key}_sweep.csv`, C.toCsv(["stage", "settings", "map5_cohort", "padded_slots", "seconds", "selected"], S.sweeps[key].list.map((x) => [x.stage, JSON.stringify(x.opts), x.map.toFixed(6), x.padded, x.seconds.toFixed(2), x === S.sweeps[key].best ? 1 : 0]))]);
    if (S.examples[key] && S.examples[key].data && S.examples[key].source.startsWith("this session")) files.push([`${key}_examples.csv`, examplesCsv(key, S.examples[key].data)]);
    if (S.tests[key]) {
      files.push([`${key}_test_top5.csv`, C.toCsv(["user_id", "rank", "tv_show_id", "title", "category", "score", "slot_source"], S.tests[key].rows)]);
      files.push([`submission_${key}.csv`, S.tests[key].submission]);
    }
    if (!files.length) { $(`${key}-export-out`).innerHTML = `<p class="empty-state">Nothing computed in this session yet.</p>`; return; }
    await exportFiles(files, $(`${key}-export-out`));
  }

  /** Rasterise a stand-alone SVG string to a PNG (base64, 2x scale). */
  function svgToPng(svgText, w, h, scale = 2) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement("canvas");
        c.width = w * scale; c.height = h * scale;
        const g = c.getContext("2d");
        g.fillStyle = "#ffffff"; g.fillRect(0, 0, c.width, c.height);
        g.drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/png").split(",")[1]);
      };
      img.onerror = () => reject(new Error("SVG could not be rasterised"));
      img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svgText);
    });
  }

  /** Everything the report needs from the PPR tab, exactly as it is on screen now. */
  async function exportPprReport() {
    const GV = window.ExamGraphView;
    if (!S.graph.hasRender()) { setStatus("ppr: drawing the graph view with the current controls…"); await S.graph.render(true); }
    const gd = GV.exportData();
    const opts = readOpts("ppr");
    const files = [];
    const settings = {
      exportedAt: new Date().toISOString(), page: location.href, tfjs: window.tf ? tf.version.tfjs : null, backend: window.tf ? tf.getBackend() : null, dataBuilt: E.meta.built,
      pprSettingsOnScreen: opts, pprSettingsSelectedOnValidation: S.saved.summary.ppr ? JSON.parse(S.saved.summary.ppr.selected_settings) : null,
      graphView: { ...gd.state, pprSettingsUsed: gd.opts, scoresFrom: gd.source, convergence: gd.conv, sharedProgrammesDrawn: gd.shared, omittedUsers: gd.omitted },
      examplePair: { heavy: E.users[S.ex.heavy.u], heavyFullViews: S.ex.heavy.s, light: E.users[S.ex.light.u], lightFullViews: S.ex.light.s, choice: S.exChoice },
      validationRunThisSession: S.runs.ppr || null, savedBatchRun: S.saved.summary.ppr || null,
    };
    files.push(["ppr_report_settings.json", JSON.stringify(settings, null, 2)]);
    const vrows = [];
    if (S.runs.ppr) { const r = S.runs.ppr; vrows.push(["this session", JSON.stringify(r.opts), r.map.toFixed(6), r.users, r.padded, r.fitS.toFixed(2), r.scoreS.toFixed(2), JSON.stringify(r.stats), settings.backend]); }
    if (S.saved.summary.ppr) { const r = S.saved.summary.ppr; vrows.push(["saved batch run (results/summary.csv)", r.selected_settings, r.valid_map5_cohort_1150, r.cohort_users, r.valid_padded_slots, "", r.valid_runtime_s, "", "wasm (Node)"]); }
    files.push(["ppr_report_validation.csv", C.toCsv(["source", "settings", "map5_cohort", "users", "padded_slots", "fit_s", "score_s", "diagnostics", "backend"], vrows)]);
    if (S.examples.ppr && S.examples.ppr.data) files.push(["ppr_report_examples.csv", examplesCsv("ppr", S.examples.ppr.data)]);
    if (S.sweeps.ppr) files.push(["ppr_report_sweep.csv", C.toCsv(["source", "stage", "settings", "map5_cohort", "padded_slots", "seconds", "selected"], S.sweeps.ppr.list.map((x) => ["this session", x.stage, JSON.stringify(x.opts), x.map.toFixed(6), x.padded, x.seconds.toFixed(2), x === S.sweeps.ppr.best ? 1 : 0]))]);
    else if (S.saved.sweeps.ppr) { const rows = S.saved.sweeps.ppr; files.push(["ppr_report_sweep.csv", C.toCsv(Object.keys(rows[0]), rows.map((r) => Object.values(r)))]); }
    files.push(["ppr_report_graph_nodes.csv", C.toCsv(["kind", "user_or_tv_show_id", "title", "category", "degree", "ppr_score", "a_full_views", "b_full_views", "top5_rank_of_score_seed", "role"],
      gd.nodes.map((n) => [n.kind === "movie" ? "programme" : "user", n.id, n.title || "", n.category || "", n.degree, n.score, n.inA || "", n.inB || "", n.rec || "", n.label || ""]))]);
    files.push(["ppr_report_graph_edges.csv", C.toCsv(["user_id", "tv_show_id", "full_views", "edge_weight", "seed_edge"],
      gd.edges.map((e) => [E.users[Number(e.source.slice(1))], E.items.id[Number(e.target.slice(1))], e.n, +e.w.toFixed(6), e.seed]))]);
    files.push(["ppr_report_graph_top5.csv", C.toCsv(Object.keys(gd.top[0]), gd.top.map((r) => Object.values(r)))]);
    for (const mode of ["kind", "score"]) {
      const snap = GV.snapshotSvg(mode);
      files.push([`ppr_report_graph_${mode}.svg`, snap.svg]);
      files.push([`ppr_report_graph_${mode}.png`, await svgToPng(snap.svg, snap.width, snap.height), "base64"]);
    }
    await exportFiles(files, $("ppr-export-out"));
    setStatus(`ppr: ${files.length} report files exported.`, "success");
  }

  function examplesCsv(key, data) {
    const rows = [];
    for (const [role, d] of Object.entries(data)) {
      d.recs.forEach((x, r) => rows.push([key, role, E.users[d.u], "recommended", r + 1, ...Object.values(label(x.i)), x.score === null ? "" : x.score, x.src, d.actual.includes(x.i) ? 1 : 0]));
      d.actual.forEach((i, r) => rows.push([key, role, E.users[d.u], "actual_weeks_21_30", r + 1, ...Object.values(label(i)), "", "", 1]));
    }
    return C.toCsv(["method", "user_role", "user_id", "list", "rank", "tv_show_id", "title", "category", "score", "slot_source", "in_actual_21_30"], rows);
  }

  // ------------------------------------------------------------------ data tab
  async function renderDataPanel() {
    const v = S.vctx;
    const eda = {};
    for (const f of ["schedule_summary", "join_summary", "test_coverage", "repeat_diagnostic", "clock_alignment", "screen_frac_hist", "views_summary"]) eda[f] = await fetchCsv(`results/eda/${f}.csv`);
    const kv = (rows, k = "metric", val = "value") => rows ? table(["Quantity", "Value"], rows.map((r) => tr([esc(r[k] ?? Object.values(r)[0]), esc(r[val] ?? Object.values(r)[1])]))) : `<p class="empty-state">Not available (open through server.py).</p>`;
    $("panel-data").innerHTML = `<div class="panel"><h2>Data &amp; shared protocol</h2>
      <div class="metric-row">
        <div class="metric"><b>${E.users.length.toLocaleString("en-US")}</b><span>users in the view log</span></div>
        <div class="metric"><b>${E.items.id.length.toLocaleString("en-US")}</b><span>labelled programmes (tv_show_id ≠ 0)</span></div>
        <div class="metric"><b>${E.meta.nLogRows.toLocaleString("en-US")}</b><span>(user, programme, week) rows</span></div>
        <div class="metric"><b>${Object.keys(E.gtValid).length.toLocaleString("en-US")}</b><span>users with a weeks 21–30 top-5</span></div>
        <div class="metric"><b>${v.cohort.length.toLocaleString("en-US")}</b><span>of them are competition test users</span></div>
        <div class="metric"><b>${E.testUsers.length.toLocaleString("en-US")}</b><span>test users to predict</span></div></div>
      <div class="step"><h3><span class="step-no">1</span>Join rule (view → programme)</h3>
        <p>A view <code>(user, vsetv_id, start, stop)</code> is matched to every <b>labelled</b> airing on the same channel (<code>vsetv_id = channel_id</code>) whose interval overlaps it.
        Per (user, airing) the overlapping seconds of all the user's views are summed and divided by the airing's <code>duration</code>: screen time. An airing counts as <b>watched</b> when screen time ≥ 0.8.
        This threshold is not tuned: it is the organiser's own label rule (Data Description: “Only TV shows with high screen time (≥ 0.8) were taken into account”), and the observed screen-time distribution is bimodal (zapping below 10 % vs. 90–100 %), so the cut sits in a sparse region.
        Rows with <code>tv_show_id = 0</code> are unlabelled airtime: they are dropped as items (they cannot be recommended or scored).</p></div>
      <div class="step"><h3><span class="step-no">2</span>Split, candidates, repeats, fallback, metric</h3>
        <ul class="discussion-list">
          <li><b>Validation:</b> fit on weeks 11–20, evaluate on weeks 21–30 (temporal order kept, nothing from weeks 21–30 is used to fit). <b>Final:</b> refit on weeks 11–30, predict weeks 31–42.</li>
          <li><b>Ground truth</b> (validation): the organiser's rule applied to weeks 21–30 — per user, count watched airings per programme, keep users with ≥ 5 programmes, take the 5 most frequent (ties: watched seconds, then id).</li>
          <li><b>Evaluation cohort:</b> the ${v.cohort.length.toLocaleString("en-US")} competition test users that have such a top-5 (the population Kaggle scores); a robustness column uses all ${Object.keys(E.gtValid).length.toLocaleString("en-US")} users.</li>
          <li><b>Candidates:</b> labelled programmes that air in the target window (${v.candidateList.length.toLocaleString("en-US")} in weeks 21–30, ${S.tctx.candidateList.length.toLocaleString("en-US")} in weeks 31–42) — the test schedule is used only for what airs, never for who watched.</li>
          <li><b>Repeats allowed</b> for every method: half of the validation top-5 programmes were already watched by the same user in weeks 11–20.</li>
          <li><b>Fallback</b> (users with no history and unfilled slots, identical for all methods): popularity = distinct viewers in the last 4 training weeks among candidates.</li>
          <li><b>MAP@5</b> exactly as in the course slides: AP@5 = Σ precision at relevant ranks / min(5, |R_u|); here |R_u| = 5 for every evaluated user.</li></ul></div>
      <div class="step"><h3><span class="step-no">3</span>Exploration results (results/eda/)</h3>
        <div class="examples-grid"><div><h4>Schedule files</h4>${eda.schedule_summary ? table(Object.keys(eda.schedule_summary[0]), eda.schedule_summary.map((r) => tr(Object.values(r).map(esc)))) : kv(null)}</div></div>
        <div class="examples-grid"><div><h4>Join</h4>${kv(eda.join_summary)}</div><div><h4>Test users and pool</h4>${kv(eda.test_coverage, "", "value")}</div>
          <div><h4>Repeat viewing (weeks 11–20 → 21–30)</h4>${kv(eda.repeat_diagnostic, "", "value")}</div>
          <div><h4>Clock check: view starts within 2 min of an airing start</h4>${eda.clock_alignment ? table(["Offset (h)", "Share", "Watched pairs (sample)"], eda.clock_alignment.map((r) => tr(Object.values(r).map(esc)))) : kv(null)}</div>
          <div><h4>Screen-time distribution (user, airing)</h4>${eda.screen_frac_hist ? table(["Screen time", "Pairs"], eda.screen_frac_hist.map((r) => tr(Object.values(r).map(esc)))) : kv(null)}</div></div></div>
      <div class="step"><h3><span class="step-no">4</span>Self-checks</h3>
        <p class="hint">Hand-checkable fixtures (MAP@5 slide example, lecture-style rule counts, the week-6 PPR reference graph, Cholesky) plus data checks and the TF.js-vs-float64 comparison.</p>
        <div class="button-row"><button type="button" id="run-tests">Run tests</button><button type="button" id="save-tests" class="secondary">Save verification.json</button></div>
        <pre id="test-log" class="test-log" tabindex="0">Press “Run tests”.</pre><div id="tests-export-out"></div></div></div>`;
    $("run-tests").addEventListener("click", () => guard(runChecks));
    $("save-tests").addEventListener("click", () => guard(async () => {
      if (!S.checks) await runChecks();
      await exportFiles([["verification.json", JSON.stringify({ ranAt: S.checks.at, backend: S.checks.backend, results: S.checks.results }, null, 2)]], $("tests-export-out"));
    }));
  }

  async function runChecks() {
    const log = $("test-log");
    log.textContent = "Running…";
    await yieldUI();
    const a = window.ExamSelfChecks.run({ withRealData: true });
    const b = S.tfReady ? await window.ExamSelfChecks.runTF() : { results: [{ name: "TF.js checks", ok: false, detail: "TensorFlow.js not loaded" }], passed: 0, failed: 1 };
    const results = [...a.results, ...b.results];
    S.checks = { at: new Date().toISOString(), backend: window.tf ? tf.getBackend() : null, results };
    log.textContent = results.map((r) => `${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.detail ? "\n      " + r.detail : ""}`).join("\n") + `\n\n${a.passed + b.passed} passed, ${a.failed + b.failed} failed`;
    setStatus(`Self-checks: ${a.passed + b.passed} passed, ${a.failed + b.failed} failed.`, a.failed + b.failed ? "error" : "success");
  }

  // ------------------------------------------------------------------ summary tab
  function renderSummary() {
    const el = $("panel-summary");
    if (!el) return;
    const keys = [...METHOD_KEYS];
    const rows = keys.map((k) => {
      const m = M.byKey(k), sv = S.saved.summary[k], r = S.runs[k];
      return `<tr>${[esc(m.title), k === "popularity" ? "required baseline" : (m.isMethod ? "method" : "reference"),
        sv ? `<b>${fmt(sv.valid_map5_cohort_1150)}</b>` : "—", sv ? fmt(sv.valid_map5_all_4030) : "—", r ? fmt(r.map) : "—", sv ? sv.valid_padded_slots : "—", sv ? `${sv.valid_runtime_s} / ${sv.test_runtime_s}` : "—",
        sv ? `<a href="results/submissions/submission_${k}.csv">submission_${k}.csv</a>` : ""].map((c) => `<td>${c}</td>`).join("")}<td class="settings-cell"><code class="small">${esc(sv ? sv.selected_settings : "")}</code></td></tr>`;
    });
    const best = METHOD_KEYS.filter((k) => M.byKey(k).isMethod && S.saved.summary[k]).sort((a, b) => Number(S.saved.summary[b].valid_map5_cohort_1150) - Number(S.saved.summary[a].valid_map5_cohort_1150))[0];
    const exRows = (role) => {
      const cols = METHOD_KEYS.map((k) => S.examples[k] && S.examples[k].data && S.examples[k].pair === pairKey() && S.examples[k].data[role]).filter(Boolean);
      if (!cols.length) return "";
      const d0 = cols[0];
      const act = new Set(d0.actual);
      const lines = [0, 1, 2, 3, 4].map((r) => tr([r + 1, ...METHOD_KEYS.map((k) => {
        const d = S.examples[k] && S.examples[k].data && S.examples[k].pair === pairKey() && S.examples[k].data[role];
        if (!d) return "—";
        const x = d.recs[r];
        return `${act.has(x.i) ? '<span class="mark-hit">✓</span> ' : ""}${esc(label(x.i).title)} <span class="hint">#${label(x.i).id} · ${esc(label(x.i).category)}</span>`;
      }), `${esc(label(d0.actual[r]).title)} <span class="hint">#${label(d0.actual[r]).id}</span>`]));
      return `<h4>${role === "heavy" ? "Heavy" : "Light"} viewer <code>${E.users[d0.u]}</code></h4>` + table(["#", ...METHOD_KEYS.map((k) => esc(M.byKey(k).title)), "Actual weeks 21–30"], lines);
    };
    el.innerHTML = `<div class="panel"><h2>Summary — method comparison on one hold-out</h2>
      <p class="hint">Validation: fit on weeks 11–20, MAP@5 on weeks 21–30 for the ${S.vctx.cohort.length.toLocaleString("en-US")}-user cohort. “Saved batch” = <code>results/summary.csv</code> written by <code>tools/run_experiments.js</code>; “this session” = runs made in this browser.
        Selection used the same validation data, so the selected values are optimistic; the Kaggle public score is the independent check.</p>
      ${table(["Method", "Role", "MAP@5 cohort (saved)", "MAP@5 all 4,030 (saved)", "MAP@5 this session", "Fallback slots", "Runtime valid / test (s)", "Kaggle file", "Selected setting (saved batch)"], rows)}
      <div class="step"><h3><span class="step-no">★</span>Current best single method</h3>
        <p>${best ? `<b>${esc(M.byKey(best).title)}</b> — validation MAP@5 ${fmt(S.saved.summary[best].valid_map5_cohort_1150)} vs. ${fmt(S.saved.summary.popularity && S.saved.summary.popularity.valid_map5_cohort_1150)} for the popularity baseline. Submission: <code>results/submissions/submission_${best}.csv</code>.` : "Run the methods first."}
        The final strategy (single method or hybrid) is decided after the Kaggle check and documented in the report.</p></div>
      <div class="step"><h3><span class="step-no">◆</span>Same two users, every method side by side</h3>
        <p class="hint">Pair: heavy <code>${E.users[S.ex.heavy.u]}</code> (${S.ex.heavy.s} full views), light <code>${E.users[S.ex.light.u]}</code> (${S.ex.light.s}) — ${esc(S.exChoice)}. Change it in step 4 of any method tab.</p>${exRows("heavy")}${exRows("light")}</div>
      <div class="button-row"><button type="button" id="summary-export" class="secondary">Save summary and examples (CSV)</button></div><div id="summary-export-out"></div></div>`;
    $("summary-export").addEventListener("click", () => guard(async () => {
      const files = [["summary_browser.csv", C.toCsv(["method", "map5_cohort_this_session", "settings", "saved_map5_cohort", "saved_map5_all_4030"], keys.map((k) => [k, S.runs[k] ? S.runs[k].map.toFixed(6) : "", S.runs[k] ? JSON.stringify(S.runs[k].opts) : "", S.saved.summary[k] ? S.saved.summary[k].valid_map5_cohort_1150 : "", S.saved.summary[k] ? S.saved.summary[k].valid_map5_all_4030 : ""]))]];
      const exAll = METHOD_KEYS.filter((k) => S.examples[k] && S.examples[k].data && S.examples[k].pair === pairKey()).map((k) => examplesCsv(k, S.examples[k].data).trim().split("\n").slice(1).join("\n")).join("\n");
      files.push(["examples_all_methods.csv", "method,user_role,user_id,list,rank,tv_show_id,title,category,score,slot_source,in_actual_21_30\n" + exAll + "\n"]);
      await exportFiles(files, $("summary-export-out"));
    }));
  }

  // ------------------------------------------------------------------ tabs, TF, start-up
  function wireTabs() {
    document.querySelectorAll(".tab-button").forEach((btn) => btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((b) => { const on = b === btn; b.classList.toggle("is-active", on); b.setAttribute("aria-selected", String(on)); });
      document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `panel-${btn.dataset.tab}`; });
      if (btn.dataset.tab === "summary") renderSummary();
    }));
  }
  function refreshTfButtons() {
    for (const k of ["mf", "ppr"]) for (const s of ["run", "sweep", "test"]) { const b = $(`${k}-${s}`); if (b && !S.tfReady) b.disabled = true; }
    document.querySelectorAll(".tf-backend").forEach((e) => { e.textContent = S.tfReady ? tf.getBackend() : "not available"; });
  }
  /** TF.js backend: ?backend=webgl (default) | wasm | cpu. All three run the same code; WebGL is fastest on a GPU. */
  async function initTf() {
    if (!window.tf) { $("tf-status").textContent = "TensorFlow.js failed to load (offline?) — MF and PPR tabs are disabled; saved results are still shown."; return; }
    const wanted = new URLSearchParams(location.search).get("backend") || "webgl";
    try {
      if (wanted === "wasm" && tf.wasm) tf.wasm.setWasmPaths("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/");
      const ok = await tf.setBackend(wanted).catch(() => false);
      if (!ok) await tf.setBackend("webgl").catch(() => tf.setBackend("cpu"));
      await tf.ready();
      S.tfReady = true;
      const opts = ["webgl", "wasm", "cpu"].map((b) => `<option value="${b}"${b === tf.getBackend() ? " selected" : ""}>${b}</option>`).join("");
      $("tf-status").innerHTML = `TensorFlow.js ${esc(tf.version.tfjs)} ready — backend: <b>${esc(tf.getBackend())}</b>.
        <label class="small">Switch backend <select id="backend-select">${opts}</select></label>
        <span class="hint">(WebGL uses the GPU; WASM is the CPU fallback — PPR and MF then take a few minutes each.)</span>`;
      $("backend-select").addEventListener("change", (e) => { const u = new URL(location.href); u.searchParams.set("backend", e.target.value); location.href = u.toString(); });
    } catch (e) { $("tf-status").textContent = `TensorFlow.js error: ${e.message}`; }
  }

  async function init() {
    if (!E || !C) { setStatus("Failed to load data/exam_data.js or js/*.js.", "error"); return; }
    setStatus("Building the validation and test contexts…");
    await yieldUI();
    S.vctx = C.buildContext(E, "valid");
    S.tctx = C.buildContext(E, "test");
    S.ex = C.exampleUsers(S.vctx);
    buildExampleLists();
    wireTabs();
    await initTf();
    const summary = await fetchCsv("results/summary.csv");
    if (summary) for (const r of summary) S.saved.summary[r.method] = r;
    for (const k of METHOD_KEYS) {
      const rows = await fetchCsv(`results/examples/${k}_examples.csv`);
      if (rows && rows.length) {
        const data = examplesFromSaved(rows);
        if (data.heavy && data.light) S.examples[k] = S.savedExamples[k] = { source: `saved batch run (results/examples/${k}_examples.csv)`, pair: `${data.heavy.u}/${data.light.u}`, data };
      }
    }
    await renderDataPanel();
    for (const k of METHOD_KEYS) renderMethodPanel(k);
    renderSummary();
    refreshTfButtons();
    setStatus(summary ? "Ready. Saved batch results loaded; every tab can recompute its numbers in the browser." : "Ready. Saved results not found (serve the folder with server.py to load them).", "success");
  }
  window.addEventListener("load", () => { init().catch((e) => { console.error(e); setStatus(`Start-up error: ${e.message}`, "error"); }); });
})();

/**
 * Batch runner (Node, TensorFlow.js WASM backend) — runs exactly the code the page runs.
 *
 * For every method:
 *   1. validation sweep: fit on weeks 11-20, MAP@5 on the 1,150-user cohort (weeks 21-30);
 *   2. select the setting with the highest cohort MAP@5 (ties: earlier grid entry = simpler);
 *   3. jobs for the selected setting: "examples" (deterministic refit + the two fixed example users),
 *      "robust:c" (MAP@5 on all 4,030 users with a weeks 21-30 top-5, in chunks), and "test"
 *      (refit on weeks 11-30, Top-5 for the 1,260 test users, Kaggle submission file).
 * Every finished unit of work is cached in results/validation/cache_<key>[.s<shard>].json, so a run
 * stopped by --budget (seconds) resumes where it stopped, and --shard=i/n splits the pending units
 * across n parallel processes. Outputs (Solution/results/): validation/<key>_sweep.csv,
 * examples/<key>_examples.csv, test/<key>_test_top5.csv, submissions/submission_<key>.csv,
 * summary.csv, environment.json.
 * Usage: node tools/run_experiments.js [keys...] [--budget=SECONDS] [--shard=i/n]
 */
const fs = require("fs");
const os = require("os");
const path = require("path");
const { loadEnv, initTf, SOL } = require("./load_env.js");
const { fileCheckpoint } = require("./checkpoint.js");

(async () => {
  const tf = await initTf("wasm");
  const g = loadEnv(["content.js", "rules.js", "mf.js", "ppr.js", "mf_tf.js", "ppr_tf.js", "methods.js"]);
  const C = g.ExamCore, E = g.EXAM, M = g.ExamMethods;
  const OUT = path.join(SOL, "results"), VAL = path.join(OUT, "validation");
  for (const d of ["validation", "examples", "test", "submissions"]) fs.mkdirSync(path.join(OUT, d), { recursive: true });

  const args = process.argv.slice(2);
  const keys = args.filter((a) => !a.startsWith("--")).length ? args.filter((a) => !a.startsWith("--")) : M.METHODS.filter((m) => !m.reference).map((m) => m.key);
  const budget = Number((args.find((a) => a.startsWith("--budget=")) || "=Infinity").split("=")[1]);
  const [si, sn] = (args.find((a) => a.startsWith("--shard=")) || "--shard=0/1").split("=")[1].split("/").map(Number);
  const START = C.nowMs();
  const elapsed = () => (C.nowMs() - START) / 1000;
  const log = (...a) => console.log(`[${elapsed().toFixed(0)}s shard ${si}/${sn}]`, ...a);

  const vctx = C.buildContext(E, "valid");
  const allUsers = C.allValidUsers(vctx);
  const tctx = C.buildContext(E, "test");
  const ex = C.exampleUsers(vctx);
  const canon = (o) => JSON.stringify(Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b))));
  const settingCols = (m) => [...new Set([...Object.keys(m.defaults), ...m.grid(null).flatMap((o) => Object.keys(o))])];

  function loadCache(key) {
    const merged = { configs: {}, jobs: {} };
    const re = new RegExp(`^cache_${key}(\\.s\\d+)?\\.json$`);
    for (const f of fs.readdirSync(VAL).filter((f) => re.test(f)).sort()) {
      const raw = JSON.parse(fs.readFileSync(path.join(VAL, f), "utf8"));
      const configs = raw.configs ? raw.configs : raw;            // legacy files are a flat {id: entry}
      for (const r of Object.values(configs)) if (r && r.opts && !merged.configs[canon(r.opts)]) merged.configs[canon(r.opts)] = r;
      Object.assign(merged.jobs, raw.jobs || {});
    }
    return merged;
  }
  const mineFile = (key) => path.join(VAL, `cache_${key}.s${si}.json`);
  function saveMine(key, mine) { fs.writeFileSync(mineFile(key), JSON.stringify(mine)); }

  async function fit(m, ctx, opts, users, ckptName = null) {
    const t0 = C.nowMs();
    const run = ckptName ? { checkpoint: fileCheckpoint(path.join(SOL, "cache", "ckpt", ckptName)), deadline: Date.now() + (budget - elapsed() - 15) * 1000 } : null;
    const model = await m.make(ctx, opts, run);
    const t1 = C.nowMs();
    if (model.prepare) await model.prepare(users);
    return { model, fitS: (t1 - t0) / 1000, t1 };
  }
  const release = (model) => { if (model && model.dispose) model.dispose(); };

  function examplesRows(model, key) {
    const rows = [];
    const label = (i) => Object.values(C.itemLabel(E, i));
    for (const [role, { u }] of Object.entries(ex)) {
      const s = model.scoreUser(u);
      const { items, fromMethod } = C.topK(vctx, s);
      const actual = E.gtValid[u];
      const hist = C.history(vctx, u).filter((h) => h.n > 0);
      const histSet = new Set(hist.map((h) => h.item));
      items.forEach((i, r) => rows.push([key, role, E.users[u], "recommended", r + 1, ...label(i),
        s && s[i] !== C.NO_SCORE ? +s[i].toPrecision(6) : "", r < fromMethod ? "method" : "fallback", actual.includes(i) ? 1 : 0, histSet.has(i) ? 1 : 0]));
      actual.forEach((i, r) => rows.push([key, role, E.users[u], "actual_weeks_21_30", r + 1, ...label(i), "", "", 1, histSet.has(i) ? 1 : 0]));
      hist.slice(0, 5).forEach((h, r) => rows.push([key, role, E.users[u], "training_top5_weeks_11_20", r + 1, ...label(h.item), h.n, "", actual.includes(h.item) ? 1 : 0, 1]));
    }
    return rows;
  }

  const finished = [];
  for (const key of keys) {
    const m = M.byKey(key);
    if (!m) { log(`unknown method ${key}`); continue; }
    const cache = loadCache(key);
    const mine = fs.existsSync(mineFile(key)) ? JSON.parse(fs.readFileSync(mineFile(key), "utf8")) : { configs: {}, jobs: {} };
    if (!mine.configs) { mine.configs = {}; mine.jobs = {}; }
    const order = [];
    let complete = true;
    const est = () => Math.max(0, ...Object.values(cache.configs).map((r) => r.fitS + r.scoreS));
    for (let st = 1; st <= (m.stages || 1); st++) {
      let prev = null;
      if (st > 1) prev = order.filter((o) => o.stage === st - 1).map((o) => cache.configs[o.id]).reduce((a, b) => (b.map > a.map ? b : a));
      const grid = m.grid(st === 1 ? null : { ...prev.opts, stage: st });
      grid.forEach((opts, j) => {
        const id = canon(opts);
        if (!order.some((o) => o.id === id)) order.push({ id, stage: st, j });
      });
      for (const [j, opts] of grid.entries()) {
        const id = canon(opts);
        if (cache.configs[id] || j % sn !== si) continue;
        if (elapsed() + 1.15 * est() > budget) break;
        const { model, fitS, t1 } = await fit(m, vctx, opts, vctx.cohort);
        const ev = C.evaluate(vctx, (u) => model.scoreUser(u));
        const r = { opts, stage: st, map: ev.map, users: ev.users, padded: ev.paddedSlots, usersPadded: ev.usersPadded, fitS, scoreS: (C.nowMs() - t1) / 1000, stats: m.stats ? m.stats(model) : {} };
        release(model);
        cache.configs[id] = r; mine.configs[id] = r; saveMine(key, mine);
        log(`${key} stage${st} ${id} MAP@5=${r.map.toFixed(5)} (${(r.fitS + r.scoreS).toFixed(1)}s)`);
      }
      if (order.filter((o) => o.stage === st).some((o) => !cache.configs[o.id])) { complete = false; break; }
    }
    if (!complete) { log(`INCOMPLETE ${key}: sweep units pending (other shards or next run)`); continue; }

    const inGrid = new Set(order.map((o) => o.id));
    const tried = [...order.map((o) => cache.configs[o.id]), ...Object.entries(cache.configs).filter(([id]) => !inGrid.has(id)).map(([, r]) => ({ ...r, offPath: true }))];
    const best = tried.filter((r) => !r.offPath).reduce((a, b) => (b.map > a.map ? b : a));
    const R = m.robustChunks || 1;
    const jobs = ["examples", "test", ...Array.from({ length: R }, (_, c) => `robust:${c}`)];
    const bestId = canon(best.opts);
    for (const [j, job] of jobs.entries()) {
      if ((cache.jobs[job] && cache.jobs[job].for === bestId) || j % sn !== si) continue;
      // a checkpointable training (TF.js MF test refit) can always start; other units need a full refit
      const jobEst = job === "test" && key === "mf" ? 0 : 1.05 * (best.fitS + best.scoreS);
      if (elapsed() + jobEst > budget) continue;
      const t0 = C.nowMs();
      let res;
      if (job === "examples") {
        const { model } = await fit(m, vctx, best.opts, vctx.cohort);
        const refit = C.evaluate(vctx, (u) => model.scoreUser(u)).map;
        res = { rows: examplesRows(model, key), refitMap: refit };
        if (R === 1) {                                   // one refit serves both units
          if (model.prepare) await model.prepare(allUsers);
          const ev = C.evaluate(vctx, (u) => model.scoreUser(u), allUsers);
          const rb = { sum: ev.map * ev.users, n: ev.users, for: bestId, seconds: 0 };
          cache.jobs["robust:0"] = rb; mine.jobs["robust:0"] = rb;
        }
        release(model);
      } else if (job === "test") {
        let model;
        try { ({ model } = await fit(m, tctx, best.opts, E.testUsers, `${key}_test_${bestId.replace(/[^a-z0-9]+/gi, "_")}`)); }
        catch (e) { if (e.checkpointed) { log(`${key} job test: ${e.message} — run again to resume`); break; } throw e; }
        const rows = [];
        let padded = 0;
        for (const u of E.testUsers) {
          const s = model.scoreUser(u);
          const { items, fromMethod } = C.topK(tctx, s);
          padded += 5 - fromMethod;
          items.forEach((i, r) => rows.push([E.users[u], r + 1, ...Object.values(C.itemLabel(E, i)), s && s[i] !== C.NO_SCORE ? +s[i].toPrecision(6) : "", r < fromMethod ? "method" : "fallback"]));
        }
        res = { rows, padded, submission: C.submissionCsv(tctx, (u) => model.scoreUser(u)) };
        release(model);
      } else {
        const c = Number(job.split(":")[1]);
        const users = allUsers.filter((_, q) => q % R === c);
        const { model } = await fit(m, vctx, best.opts, users);
        const ev = C.evaluate(vctx, (u) => model.scoreUser(u), users);
        res = { sum: ev.map * ev.users, n: ev.users };
        release(model);
      }
      res.for = bestId; res.seconds = (C.nowMs() - t0) / 1000;
      cache.jobs[job] = res; mine.jobs[job] = res; saveMine(key, mine);
      log(`${key} job ${job} done in ${res.seconds.toFixed(1)}s`);
    }
    if (jobs.some((job) => !(cache.jobs[job] && cache.jobs[job].for === bestId))) { log(`INCOMPLETE ${key}: final jobs pending`); continue; }

    const cols = settingCols(m);
    fs.writeFileSync(path.join(VAL, `${key}_sweep.csv`), C.toCsv(
      ["method", "config", "stage", ...cols, "map5_cohort_1150", "users", "padded_slots", "users_padded", "fit_s", "score_s", "diagnostics", "selected", "on_final_search_path"],
      tried.map((r, j) => [key, j + 1, r.stage, ...cols.map((c) => r.opts[c] ?? ""), r.map.toFixed(6), r.users, r.padded, r.usersPadded,
        r.fitS.toFixed(2), r.scoreS.toFixed(2), JSON.stringify(r.stats), r === best ? 1 : 0, r.offPath ? 0 : 1])));
    fs.writeFileSync(path.join(OUT, "examples", `${key}_examples.csv`), C.toCsv(
      ["method", "user_role", "user_id", "list", "rank", "tv_show_id", "title", "category", "score", "slot_source", "in_actual_21_30", "in_training_history"], cache.jobs.examples.rows));
    fs.writeFileSync(path.join(OUT, "test", `${key}_test_top5.csv`), C.toCsv(["user_id", "rank", "tv_show_id", "title", "category", "score", "slot_source"], cache.jobs.test.rows));
    fs.writeFileSync(path.join(OUT, "submissions", `submission_${key}.csv`), cache.jobs.test.submission);
    let sum = 0, n = 0;
    for (let c = 0; c < R; c++) { sum += cache.jobs[`robust:${c}`].sum; n += cache.jobs[`robust:${c}`].n; }
    const refitDiff = Math.abs(cache.jobs.examples.refitMap - best.map);
    finished.push([key, m.title, m.isMethod ? "method" : (m.reference ? "reference implementation" : "reference"), tried.length, JSON.stringify(best.opts), best.map.toFixed(6), (sum / n).toFixed(6), best.users,
      best.padded, cache.jobs.test.padded, (best.fitS + best.scoreS).toFixed(2), cache.jobs.test.seconds.toFixed(2), refitDiff.toExponential(1)]);
    log(`== ${key}: selected ${JSON.stringify(best.opts)} cohort MAP@5=${best.map.toFixed(5)} all-4030=${(sum / n).toFixed(5)} refit |diff|=${refitDiff.toExponential(1)}`);
  }

  if (finished.length) {
    const sumPath = path.join(OUT, "summary.csv");
    const header = ["method", "title", "role", "configs_tried", "selected_settings", "valid_map5_cohort_1150", "valid_map5_all_4030", "cohort_users", "valid_padded_slots", "test_padded_slots", "valid_runtime_s", "test_runtime_s", "refit_map_abs_diff"];
    const done = finished.map((r) => r[0]);
    const prev = fs.existsSync(sumPath) ? fs.readFileSync(sumPath, "utf8").trim().split("\n").slice(1).filter((l) => l && !done.includes(l.split(",")[0])) : [];
    const orderKeys = M.METHODS.map((x) => x.key);
    const lines = [...prev, ...finished.map((r) => C.toCsv([], [r]).trim().split("\n").pop())].sort((a, b) => orderKeys.indexOf(a.split(",")[0]) - orderKeys.indexOf(b.split(",")[0]));
    fs.writeFileSync(sumPath, [header.join(","), ...lines].join("\n") + "\n");
  }
  fs.writeFileSync(path.join(OUT, "environment.json"), JSON.stringify({ node: process.version, tfjs: tf.version.tfjs, tfBackend: tf.getBackend(), platform: `${os.platform()} ${os.arch()}`, cpus: os.cpus().length, memGB: +(os.totalmem() / 2 ** 30).toFixed(1), dataBuilt: E.meta.built, lastRun: new Date().toISOString(), exampleUsers: { heavy: E.users[ex.heavy.u], heavyTrainFullViews: ex.heavy.s, light: E.users[ex.light.u], lightTrainFullViews: ex.light.s } }, null, 2));
})().catch((e) => { console.error(e); process.exit(1); });

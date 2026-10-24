/**
 * List profile of every method's selected setting on the validation cohort (fit weeks 11-20, evaluate weeks 21-30):
 * how many Top-5 slots are programmes the user already watched (>= 80 %) in weeks 11-20 ("familiar") versus new ones,
 * where the hits come from, and how many distinct programmes the method shows. Also writes per-user AP@5 files used for
 * the hand check in the report. One method per call is fine: results are cached in results/analysis/cache_<key>.json,
 * and long TF.js fits are checkpointed (run again to resume).
 * Usage: node tools/analyze_lists.js [keys...] [--budget=SECONDS]
 */
const fs = require("fs");
const path = require("path");
const { loadEnv, initTf, SOL } = require("./load_env.js");
const { fileCheckpoint } = require("./checkpoint.js");

(async () => {
  const T0 = Date.now();
  const budget = Number((process.argv.find((a) => a.startsWith("--budget=")) || "=Infinity").split("=")[1]);
  await initTf("wasm");
  const g = loadEnv(["content.js", "rules.js", "mf.js", "ppr.js", "mf_tf.js", "ppr_tf.js", "methods.js"]);
  const C = g.ExamCore, E = g.EXAM, M = g.ExamMethods;
  const OUT = path.join(SOL, "results", "analysis");
  fs.mkdirSync(OUT, { recursive: true });
  // selected settings = results/summary.csv
  const sum = fs.readFileSync(path.join(SOL, "results", "summary.csv"), "utf8").trim().split("\n");
  const head = sum[0].split(",");
  const parseCsvLine = (l) => { const out = []; let cur = "", q = false; for (let i = 0; i < l.length; i++) { const c = l[i]; if (q) { if (c === '"' && l[i + 1] === '"') { cur += '"'; i++; } else if (c === '"') q = false; else cur += c; } else if (c === '"') q = true; else if (c === ",") { out.push(cur); cur = ""; } else cur += c; } out.push(cur); return out; };
  const selected = {}, reported = {};
  for (const l of sum.slice(1)) { const r = parseCsvLine(l); selected[r[0]] = JSON.parse(r[head.indexOf("selected_settings")]); reported[r[0]] = +r[head.indexOf("valid_map5_cohort_1150")]; }

  const keys = process.argv.slice(2).filter((a) => !a.startsWith("--"));
  const order = keys.length ? keys : ["popularity", "content", "rules", "mf", "ppr", "own"];
  const vctx = C.buildContext(E, "valid");
  const cohort = vctx.cohort;
  const familiarSet = (u) => new Set(C.history(vctx, u).filter((h) => h.n > 0).map((h) => h.item));

  for (const key of order) {
    const cacheFile = path.join(OUT, `cache_${key}.json`);
    if (fs.existsSync(cacheFile)) { console.log(`${key}: cached`); continue; }
    const m = M.METHODS.find((x) => x.key === key);
    const opts = selected[key];
    const name = `analysis_${key}_valid_` + Object.keys(opts).sort().map((k) => `${k}_${opts[k]}`).join("_").replace(/[^a-z0-9]+/gi, "_");
    const run = { checkpoint: fileCheckpoint(path.join(SOL, "cache", "ckpt", name)), deadline: T0 + (budget - 20) * 1000 };
    let model;
    try { model = await m.make(vctx, opts, run); }
    catch (e) { if (e.checkpointed) { console.log(`${key}: ${e.message} — run again to resume`); return; } throw e; }
    if (model.prepare) await model.prepare(cohort);
    const ev = C.evaluate(vctx, (u) => model.scoreUser(u), cohort);
    if (model.dispose) model.dispose();
    let slots = 0, fam = 0, hits = 0, hitsFam = 0, gtFam = 0, gtN = 0, fromFallback = 0;
    const shown = new Set(), perUser = [];
    for (const r of ev.perUser) {
      const F = familiarSet(r.u), gt = E.gtValid[r.u], G = new Set(gt);
      r.items.forEach((i, k) => {
        slots++; shown.add(i);
        const f = F.has(i), h = G.has(i);
        if (f) fam++; if (h) hits++; if (f && h) hitsFam++;
        if (k >= r.fromMethod) fromFallback++;
      });
      gt.forEach((i) => { gtN++; if (F.has(i)) gtFam++; });
      perUser.push([E.users[r.u], key, r.items.map((i) => E.items.id[i]).join(" "), gt.map((i) => E.items.id[i]).join(" "),
        r.items.map((i) => (G.has(i) ? 1 : 0)).join(""), r.fromMethod, +r.ap.toFixed(6)]);
    }
    const res = {
      key, settings: opts, users: cohort.length, map5: +ev.map.toFixed(6), map5_reported: reported[key],
      familiar_slot_share: +(fam / slots).toFixed(4), new_slot_share: +(1 - fam / slots).toFixed(4),
      hits_per_user: +(hits / cohort.length).toFixed(4), hit_share_familiar: hits ? +(hitsFam / hits).toFixed(4) : 0,
      precision_familiar_slots: fam ? +(hitsFam / fam).toFixed(4) : 0, precision_new_slots: slots - fam ? +((hits - hitsFam) / (slots - fam)).toFixed(4) : 0,
      fallback_slots: fromFallback, distinct_programmes_shown: shown.size, gt_familiar_share: +(gtFam / gtN).toFixed(4),
      seconds: +((Date.now() - T0) / 1000).toFixed(1),
    };
    fs.writeFileSync(cacheFile, JSON.stringify({ res, perUser }));
    console.log(JSON.stringify(res));
  }
  // assemble outputs from whatever is cached
  const done = ["popularity", "content", "rules", "mf", "ppr", "own"].filter((k) => fs.existsSync(path.join(OUT, `cache_${k}.json`)));
  const all = done.map((k) => JSON.parse(fs.readFileSync(path.join(OUT, `cache_${k}.json`), "utf8")));
  const cols = ["key", "users", "map5", "map5_reported", "familiar_slot_share", "new_slot_share", "hits_per_user", "hit_share_familiar", "precision_familiar_slots", "precision_new_slots", "fallback_slots", "distinct_programmes_shown", "gt_familiar_share"];
  fs.writeFileSync(path.join(OUT, "list_profile.csv"), C.toCsv(cols, all.map((a) => cols.map((c) => a.res[c]))));
  // AP@5 recomputed from the hit flags at full precision (the cached value is rounded to 6 decimals), so that the
  // mean of the ap5 column reproduces MAP@5 exactly; every cohort user has |actual| = 5 = K.
  const apFromFlags = (flags, nActual) => { let h = 0, s = 0; [...flags].forEach((f, k) => { if (f === "1") { h++; s += h / (k + 1); } }); return s / Math.min(5, nActual); };
  const rowsAp = all.flatMap((a) => a.perUser.map((r) => [...r.slice(0, 6), apFromFlags(r[4], r[3].split(" ").length)]));
  fs.writeFileSync(path.join(OUT, "per_user_ap_valid.csv"), C.toCsv(["user_id", "method", "top5", "actual_top5_weeks_21_30", "hit_flags", "slots_from_method", "ap5"], rowsAp));
  console.log(`assembled ${done.join(", ")} in ${((Date.now() - T0) / 1000).toFixed(0)}s`);
})();

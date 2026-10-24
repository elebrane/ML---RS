/**
 * Refit one method with explicit settings on weeks 11-30 and write its Kaggle submission file.
 * Used to check which settings produced a file that was already uploaded (provenance check).
 * Long TF.js fits are checkpointed in cache/ckpt/<name>, so a run stopped by --budget resumes.
 * Usage: node tools/refit_submission.js <key> '<settings json>' <out.csv> [--budget=SECONDS]
 */
const fs = require("fs");
const path = require("path");
const { loadEnv, initTf, SOL } = require("./load_env.js");
const { fileCheckpoint } = require("./checkpoint.js");

(async () => {
  const [key, json, out] = process.argv.slice(2).filter((a) => !a.startsWith("--"));
  const budget = Number((process.argv.find((a) => a.startsWith("--budget=")) || "=Infinity").split("=")[1]);
  const T0 = Date.now();
  await initTf("wasm");
  const g = loadEnv(["content.js", "rules.js", "mf.js", "ppr.js", "mf_tf.js", "ppr_tf.js", "methods.js"]);
  const C = g.ExamCore, E = g.EXAM, M = g.ExamMethods;
  const m = M.METHODS.find((x) => x.key === key);
  const opts = JSON.parse(json);
  const name = `refit_${key}_` + Object.keys(opts).sort().map((k) => `${k}_${opts[k]}`).join("_").replace(/[^a-z0-9]+/gi, "_");
  const run = { checkpoint: fileCheckpoint(path.join(SOL, "cache", "ckpt", name)), deadline: T0 + (budget - 15) * 1000 };
  const tctx = C.buildContext(E, "test");
  let model;
  try { model = await m.make(tctx, opts, run); }
  catch (e) { if (e.checkpointed) { console.log(`stopped: ${e.message} — run again to resume`); return; } throw e; }
  if (model.prepare) await model.prepare(E.testUsers);
  fs.writeFileSync(out, C.submissionCsv(tctx, (u) => model.scoreUser(u)));
  console.log(`written ${out} in ${((Date.now() - T0) / 1000).toFixed(0)}s`, JSON.stringify(opts));
})();

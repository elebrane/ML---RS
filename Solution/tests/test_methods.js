/**
 * Self-checks for the shared core, the four methods and the TF.js implementations
 * (run: node tests/test_methods.js --real). The page runs the same checks from "Run tests".
 */
const { loadEnv, initTf } = require("../tools/load_env.js");
(async () => {
  await initTf("wasm");
  const g = loadEnv(["content.js", "rules.js", "mf.js", "ppr.js", "mf_tf.js", "ppr_tf.js", "selfchecks.js"]);
  const a = g.ExamSelfChecks.run({ withRealData: process.argv.includes("--real") });
  const b = process.argv.includes("--real") ? await g.ExamSelfChecks.runTF() : { results: [], passed: 0, failed: 0 };
  for (const r of [...a.results, ...b.results]) console.log(`${r.ok ? "PASS" : "FAIL"}  ${r.name}${r.detail ? "  — " + r.detail : ""}`);
  const failed = a.failed + b.failed;
  console.log(`\n${a.passed + b.passed} passed, ${failed} failed (TF.js backend: ${g.tf.getBackend()})`);
  process.exit(failed ? 1 : 0);
})();

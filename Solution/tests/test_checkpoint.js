/**
 * Resume check: 6 straight Adam steps vs. the same 6 steps split over 4 processes with checkpoints.
 * Not bit-identical: on restore TF.js recomputes Adam's bias-correction powers as beta^t instead of the
 * running product, a float32 rounding difference (observed max |diff| 3e-6). Tolerance: 1e-4.
 */
const path = require("path"), fs = require("fs"), os = require("os");
const { loadEnv, initTf } = require("../tools/load_env.js");
const { fileCheckpoint } = require("../tools/checkpoint.js");
(async () => {
  await initTf("wasm");
  const g = loadEnv(["mf_tf.js"]);
  const ctx = g.ExamCore.buildContext(g.EXAM, "valid");
  const opts = { factors: 8, lambda: 50, alpha: 1, signal: "count", epochs: 6, batchSize: 8192, lr: 0.05, seed: 42 };
  const a = await g.ExamMFTF.trainMFTF(ctx, opts);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "ckpt-"));
  const ck = fileCheckpoint(dir);
  let stopped = false;
  try { await g.ExamMFTF.trainMFTF(ctx, opts, null, { checkpoint: { load: ck.load, save: ck.save }, deadline: 1 }); }
  catch (e) { stopped = e.checkpointed; }
  // first call stops after step 1 (deadline already passed); resume twice more to reach step 6
  let b = null, calls = 1;
  while (!b) {
    calls++;
    try { b = await g.ExamMFTF.trainMFTF(ctx, opts, null, { checkpoint: ck, deadline: calls < 4 ? 1 : null }); }
    catch (e) { if (!e.checkpointed) throw e; }
  }
  let maxDiff = 0;
  for (let j = 0; j < a.V.length; j++) maxDiff = Math.max(maxDiff, Math.abs(a.V[j] - b.V[j]));
  for (let j = 0; j < a.U.length; j++) maxDiff = Math.max(maxDiff, Math.abs(a.U[j] - b.U[j]));
  const ok = stopped && maxDiff < 1e-4;
  console.log(`${ok ? "PASS" : "FAIL"}  MF checkpoint/resume: ${calls} calls, max |factor diff| = ${maxDiff}, final loss ${a.finalLoss.toFixed(3)} vs ${b.finalLoss.toFixed(3)}`);
  process.exit(ok ? 0 : 1);
})();

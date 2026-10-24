const t00 = Date.now(); const log = (...a) => console.log(((Date.now() - t00) / 1000).toFixed(1) + "s", ...a);
const { loadEnv, initTf } = require(process.cwd() + "/tools/load_env.js");
(async () => {
  const tf = await initTf("wasm");
  const g = loadEnv(["mf.js", "mf_tf.js"]);
  const C = g.ExamCore; const ctx = C.buildContext(g.EXAM, "valid");
  const opts = { factors: 32, lambda: 50, alpha: 1, signal: "hours", steps: 150, lr: 0.05, seed: 42 };
  const m = await g.ExamMFTF.trainMFTF(ctx, opts, (s, l) => log("step", s, "loss", l.toFixed(1)));
  log("MF-TF MAP@5", C.evaluate(ctx, (u) => m.scoreUser(u)).map.toFixed(5), "nU", m.nU, "nI", m.nI, "tensors left", tf.memory().numTensors);
  const a = g.ExamMF.trainALS(ctx, { factors: 32, lambda: 50, alpha: 1, signal: "hours", confidence: "linear", iterations: 10, seed: 42, trackLoss: true });
  log("ALS same objective: final loss", a.history[a.history.length - 1].lossAfterItemStep.toFixed(1), "MAP@5", C.evaluate(ctx, (u) => a.scoreUser(u)).map.toFixed(5));
})().catch((e) => log("ERR", e.stack));

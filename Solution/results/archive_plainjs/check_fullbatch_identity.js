const { loadEnv, initTf } = require("./load_env.js");
(async () => {
  await initTf("wasm");
  const g = loadEnv(["mf_tf.js"]);
  const C = g.ExamCore, ctx = C.buildContext(g.EXAM, "valid");
  const t0 = Date.now();
  const m = await g.ExamMFTF.trainMFTF(ctx, { factors: 16, lambda: 50, alpha: 1, signal: "count", lr: 0.05, epochs: 100, batchSize: 8192, seed: 42 });
  const ev = C.evaluate(ctx, (u) => m.scoreUser(u));
  console.log("full batch (epochs 100, batch 8192):", ev.map.toFixed(6), "fullBatch", m.fullBatch, "updates", m.updates, ((Date.now() - t0) / 1000).toFixed(0) + "s");
  const t1 = Date.now();
  const mb = await g.ExamMFTF.trainMFTF(ctx, { factors: 16, lambda: 50, alpha: 1, signal: "count", lr: 0.05, epochs: 20, batchSize: 1024, seed: 42 });
  console.log("mini-batch (epochs 20, batch 1024):", C.evaluate(ctx, (u) => mb.scoreUser(u)).map.toFixed(6), "batches/epoch", mb.batches, "updates", mb.updates, ((Date.now() - t1) / 1000).toFixed(0) + "s", "loss", mb.history.map((h) => h.epoch + ":" + h.loss.toFixed(0)).slice(0, 4).join(" "), "...", mb.finalLoss.toFixed(0));
})();

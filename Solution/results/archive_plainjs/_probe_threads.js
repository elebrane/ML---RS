const path = require("path");
(async () => {
  const tf = require("@tensorflow/tfjs");
  const wasm = require("@tensorflow/tfjs-backend-wasm");
  wasm.setWasmPaths(path.join(process.cwd(), "node_modules/@tensorflow/tfjs-backend-wasm/dist") + path.sep);
  const n = Number(process.argv[2] || 1);
  if (wasm.setThreadsCount) wasm.setThreadsCount(n);
  await tf.setBackend("wasm"); await tf.ready();
  const threads = wasm.getThreadsCount ? wasm.getThreadsCount() : "n/a";
  const U = tf.randomNormal([4737, 64]), V = tf.randomNormal([2500, 64]), Cm = tf.ones([4737, 2500]);
  const t0 = Date.now();
  for (let s = 0; s < 3; s++) { const l = tf.tidy(() => { const e = tf.sub(1, tf.matMul(U, V, false, true)); return tf.sum(tf.mul(Cm, tf.square(e))); }); await l.data(); l.dispose(); }
  console.log("requested", n, "threads reported", threads, "ms per forward", ((Date.now() - t0) / 3).toFixed(0));
})();

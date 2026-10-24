const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-backend-wasm");
const { setWasmPaths } = require("@tensorflow/tfjs-backend-wasm");
setWasmPaths(require("path").join(process.cwd(), "node_modules/@tensorflow/tfjs-backend-wasm/dist/"));
(async () => {
  for (const be of ["cpu", "wasm"]) {
    await tf.setBackend(be); await tf.ready();
    const a = tf.randomNormal([1150, 4700]), b = tf.randomNormal([4700, 2500]);
    const t0 = Date.now(); const c = tf.matMul(a, b); await c.data(); const dt = (Date.now() - t0) / 1000;
    console.log(be, "matMul 1150x4700x2500:", dt.toFixed(2), "s =>", (2 * 1150 * 4700 * 2500 / dt / 1e9).toFixed(1), "GFLOP/s");
    tf.dispose([a, b, c]);
  }
})();

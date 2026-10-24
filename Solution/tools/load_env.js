/** Node helper: load the packed data, TensorFlow.js (WASM backend) and the page's scripts into globalThis. */
const path = require("path");
const SOL = path.resolve(__dirname, "..");
async function initTf(backend = "wasm") {
  const tf = require("@tensorflow/tfjs");
  if (backend === "wasm") {
    const wasm = require("@tensorflow/tfjs-backend-wasm");
    wasm.setWasmPaths(path.join(SOL, "node_modules", "@tensorflow", "tfjs-backend-wasm", "dist") + path.sep);
  }
  await tf.setBackend(backend);
  await tf.ready();
  globalThis.tf = tf;
  return tf;
}
function loadEnv(extra = []) {
  require(path.join(SOL, "data", "exam_data.js"));
  for (const f of ["core.js", "popularity.js", ...extra]) require(path.join(SOL, "js", f));
  return globalThis;
}
module.exports = { loadEnv, initTf, SOL };

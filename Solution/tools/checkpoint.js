/** File checkpoint for long TF.js trainings in Node (binary Float32 arrays + JSON metadata). */
const fs = require("fs");
const path = require("path");
function fileCheckpoint(dir) {
  return {
    async load() {
      const meta = path.join(dir, "meta.json");
      if (!fs.existsSync(meta)) return null;
      const m = JSON.parse(fs.readFileSync(meta, "utf8"));
      const rd = (f) => new Float32Array(new Uint8Array(fs.readFileSync(path.join(dir, f))).buffer);
      return { step: m.step, history: m.history, U: rd("U.bin"), V: rd("V.bin"),
        opt: m.opt.map((w, j) => ({ name: w.name, shape: w.shape, data: w.dtype === "int32" ? new Int32Array(new Uint8Array(fs.readFileSync(path.join(dir, `opt${j}.bin`))).buffer) : rd(`opt${j}.bin`) })) };
    },
    async save(st) {
      fs.mkdirSync(dir, { recursive: true });
      const wr = (f, a) => fs.writeFileSync(path.join(dir, f), Buffer.from(a.buffer, a.byteOffset, a.byteLength));
      wr("U.bin", st.U); wr("V.bin", st.V);
      st.opt.forEach((w, j) => wr(`opt${j}.bin`, w.data));
      fs.writeFileSync(path.join(dir, "meta.json"), JSON.stringify({ step: st.step, history: st.history,
        opt: st.opt.map((w) => ({ name: w.name, shape: w.shape, dtype: w.data instanceof Int32Array ? "int32" : "float32" })) }));
    },
  };
}
module.exports = { fileCheckpoint };

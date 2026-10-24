const { initTf } = require("./load_env.js");
(async () => {
  const tf = await initTf("wasm");
  const U = tf.variable(tf.randomNormal([4737, 64], 0, 0.01, "float32", 1)), V = tf.variable(tf.randomNormal([2500, 64], 0, 0.01, "float32", 2));
  const C = tf.ones([4737, 2500]);
  const opt = tf.train.adam(0.05);
  const B = 1024;
  const Cb = tf.slice(C, [0, 0], [B, 2500]);
  const t0 = Date.now();
  for (let s = 0; s < 3; s++) {
    const l = opt.minimize(() => tf.tidy(() => {
      const Ub = tf.slice(U, [B, 0], [B, 64]);
      const err = tf.sub(1, tf.matMul(Ub, V, false, true));
      return tf.add(tf.sum(tf.mul(Cb, tf.square(err))), tf.mul(50, tf.add(tf.sum(tf.square(Ub)), tf.mul(B / 4737, tf.sum(tf.square(V))))));
    }), true);
    console.log("step", s, (await l.data())[0].toFixed(1), (Date.now() - t0) + "ms");
    l.dispose();
  }
})().catch((e) => console.log("ERR", e.message));

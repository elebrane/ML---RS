/**
 * mf_tf.js — Method 3: matrix factorization for implicit feedback in TensorFlow.js
 * (course week 5: user and programme embedding tables, dot-product score, Adam optimizer, epochs and
 * mini-batches — the same four training controls as HW5: epochs, learning rate, L2 regularisation, batch size).
 *
 * Implicit data has no ratings, so the week-5 "fit observed ratings with MSE" recipe is replaced by the
 * implicit-feedback objective of Hu, Koren & Volinsky (2008) over ALL user x programme cells:
 *     loss = sum_{u,i} c_ui (p_ui - u_u . v_i)^2 + lambda (||U||^2 + ||V||^2)
 *     p_ui = 1 if user u watched programme i >= 80 % at least once in the training weeks, else 0
 *     c_ui = 1 + alpha * r_ui,  r = full views ("count") or hours inside full views ("hours"); c = 1 elsewhere
 * An unobserved cell is not a dislike: it is a 0 with the lowest confidence.
 * Mini-batches are batches of USERS (rows of the matrix) with every programme in each row. One epoch = one
 * pass over all users; the user order is a seeded permutation and the batch order is reshuffled every epoch.
 * Per batch b the loss is  sum_{u in b, i} c (p - u.v)^2 + lambda (||U_b||^2 + |b| / n_users * ||V||^2):
 * the penalty touches only the batch's user rows (lesson of HW5) and the shared programme table gets its
 * share, so the batch losses of one epoch add up to the full objective.
 * batchSize >= number of users means one full batch per epoch (no permutation) — exactly the full-matrix
 * training used for the saved batch run.
 * Only users / programmes with at least one training interaction are materialised (the others have no
 * signal: their optimal factors are exactly 0), so the dense prediction U V^T stays small.
 * Score for user u and programme i = u_u . v_i; programmes with no interaction get no score.
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  /**
   * @param {object} opts {factors, lambda, alpha, signal, lr, epochs, batchSize, seed} (legacy "steps" = epochs)
   * @param {function(number, number)} [onEpoch] called with (epoch, epoch loss) on recorded epochs
   * @param {object} run optional {deadline: ms timestamp, checkpoint: {load(), save(state)}} — lets a long
   *   training be split across several processes (Node runner); the page trains in one go and passes nothing.
   */
  async function trainMFTF(ctx, opts, onEpoch, run = null) {
    const tf = root.tf;
    const k = opts.factors, lambda = opts.lambda, alpha = opts.alpha;
    const epochs = Math.max(1, Math.round(opts.epochs ?? opts.steps ?? 100)), lr = opts.lr ?? 0.05, seed = opts.seed ?? 42;
    const uMap = new Int32Array(ctx.nUsers).fill(-1), iMap = new Int32Array(ctx.nItems).fill(-1);
    const users = [], items = [];
    for (let u = 0; u < ctx.nUsers; u++) for (let q = ctx.aggPtr[u]; q < ctx.aggPtr[u + 1]; q++) {
      if (ctx.aN[q] === 0) continue;
      if (uMap[u] < 0) { uMap[u] = users.length; users.push(u); }
      const i = ctx.aItem[q];
      if (iMap[i] < 0) { iMap[i] = items.length; items.push(i); }
    }
    const nU = users.length, nI = items.length;
    const batchSize = Math.max(1, Math.round(opts.batchSize ?? nU));
    const fullBatch = batchSize >= nU;
    if (!fullBatch) {                                    // seeded row permutation: contiguous slices = random batches
      const rp = C.mulberry32(seed + 7), perm = Int32Array.from({ length: nU }, (_, j) => j);
      for (let j = nU - 1; j > 0; j--) { const r = Math.floor(rp() * (j + 1)); const t = perm[j]; perm[j] = perm[r]; perm[r] = t; }
      for (const u of users) uMap[u] = perm[uMap[u]];
    }
    const Cd = new Float32Array(nU * nI).fill(1), Pd = new Float32Array(nU * nI);
    let positives = 0;
    for (let u = 0; u < ctx.nUsers; u++) for (let q = ctx.aggPtr[u]; q < ctx.aggPtr[u + 1]; q++) {
      if (ctx.aN[q] === 0) continue;
      const r = opts.signal === "hours" ? ctx.aM[q] / 60 : ctx.aN[q];
      const cell = uMap[u] * nI + iMap[ctx.aItem[q]];
      Cd[cell] = 1 + alpha * r; Pd[cell] = 1; positives++;
    }
    const nb = fullBatch ? 1 : Math.ceil(nU / batchSize);
    const bounds = Array.from({ length: nb }, (_, b) => (fullBatch ? [0, nU] : [b * batchSize, Math.min(batchSize, nU - b * batchSize)]));
    const Cb = [], Pb = [];
    for (const [s0, len] of bounds) {
      Cb.push(tf.tensor2d(Cd.subarray(s0 * nI, (s0 + len) * nI), [len, nI]));
      Pb.push(tf.tensor2d(Pd.subarray(s0 * nI, (s0 + len) * nI), [len, nI]));
    }
    const orderRng = C.mulberry32(seed + 13);             // batch order of every epoch, drawn up front (resumable)
    const orders = Array.from({ length: epochs }, () => {
      const o = Array.from({ length: nb }, (_, j) => j);
      if (!fullBatch) for (let j = nb - 1; j > 0; j--) { const r = Math.floor(orderRng() * (j + 1)); [o[j], o[r]] = [o[r], o[j]]; }
      return o;
    });
    // tf.variable copies its initial value, so the random initialisers are disposed straight away
    const U = tf.tidy(() => tf.variable(tf.randomNormal([nU, k], 0, 0.01, "float32", seed), true, "mf_U"));
    const V = tf.tidy(() => tf.variable(tf.randomNormal([nI, k], 0, 0.01, "float32", seed + 1), true, "mf_V"));
    const optimizer = tf.train.adam(lr);
    let startEpoch = 1;
    let history = [];
    if (run && run.checkpoint) {
      const st = await run.checkpoint.load();
      if (st) {                                          // resume: weights + Adam moments + iteration count
        tf.tidy(() => { U.assign(tf.tensor2d(st.U, [nU, k])); V.assign(tf.tensor2d(st.V, [nI, k])); });
        const restored = st.opt.map((w) => ({ name: w.name, tensor: tf.tensor(w.data, w.shape) }));
        await optimizer.setWeights(restored);               // Adam copies them into its own variables
        tf.dispose(restored.map((w) => w.tensor));
        startEpoch = st.step + 1; history = st.history;
      }
    }
    // full batch: the original full-matrix loss (bit-identical to the saved run); otherwise the batch loss
    const lossFull = () => tf.tidy(() => {
      const err = tf.sub(Pb[0], tf.matMul(U, V, false, true));
      return tf.add(tf.sum(tf.mul(Cb[0], tf.square(err))), tf.mul(lambda, tf.add(tf.sum(tf.square(U)), tf.sum(tf.square(V)))));
    });
    const lossBatch = (b) => () => tf.tidy(() => {
      const [s0, len] = bounds[b];
      const Ub = tf.slice(U, [s0, 0], [len, k]);
      const err = tf.sub(Pb[b], tf.matMul(Ub, V, false, true));
      const reg = tf.add(tf.sum(tf.square(Ub)), tf.mul(len / nU, tf.sum(tf.square(V))));
      return tf.add(tf.sum(tf.mul(Cb[b], tf.square(err))), tf.mul(lambda, reg));
    });
    const recEvery = Math.max(1, Math.round(epochs / 20));
    const releaseData = () => tf.dispose([...Cb, ...Pb]);
    for (let epoch = startEpoch; epoch <= epochs; epoch++) {
      const record = epoch === 1 || epoch === epochs || epoch % recEvery === 0;
      let epochLoss = 0;
      for (const b of orders[epoch - 1]) {
        const loss = optimizer.minimize(fullBatch ? lossFull : lossBatch(b), true);
        if (record) epochLoss += (await loss.data())[0];
        loss.dispose();
      }
      if (record) {
        history.push({ epoch, loss: epochLoss });
        if (onEpoch) await onEpoch(epoch, epochLoss);
      }
      if (run && run.checkpoint && run.deadline && Date.now() > run.deadline && epoch < epochs) {
        const ow = await optimizer.getWeights();
        const opt = [];
        for (const w of ow) opt.push({ name: w.name, shape: w.tensor.shape, data: await w.tensor.data() });
        await run.checkpoint.save({ step: epoch, U: await U.data(), V: await V.data(), opt, history });
        optimizer.dispose(); releaseData(); tf.dispose([U, V]);
        const err = new Error(`MF_CHECKPOINTED at epoch ${epoch}/${epochs}`);
        err.checkpointed = true;
        throw err;
      }
    }
    optimizer.dispose();
    releaseData();
    const Ud = await U.data(), Vd = await V.data();
    tf.dispose([U, V]);
    let factorNorm = 0;
    for (const x of Vd) factorNorm += x * x;
    return {
      k, history, factorNorm: Math.sqrt(factorNorm), positives, nU, nI, U: Ud, V: Vd, uMap, iMap,
      epochs, batchSize: fullBatch ? nU : batchSize, batches: nb, updates: epochs * nb, fullBatch,
      finalLoss: history.length ? history[history.length - 1].loss : null,
      describe: `factors=${k}, lambda=${lambda}, alpha=${alpha}, signal=${opts.signal}, epochs=${epochs}, batch=${fullBatch ? "full" : batchSize}, lr=${lr}, seed=${seed}`,
      scoreUser(u) {
        const ru = uMap[u];
        if (ru < 0) return null;                         // no training views -> shared fallback
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (const i of ctx.candidateList) {
          const ri = iMap[i];
          if (ri < 0) continue;
          let d = 0;
          for (let a = 0; a < k; a++) d += Ud[ru * k + a] * Vd[ri * k + a];
          s[i] = d;
        }
        return s;
      },
    };
  }

  root.ExamMFTF = { trainMFTF };
})(typeof window !== "undefined" ? window : globalThis);

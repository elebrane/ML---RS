/**
 * ppr_tf.js — Method 4: Personalized PageRank with TensorFlow.js tensor operations (course week 6).
 *
 * Bipartite user-programme graph from the training weeks (edge = watched >= 80 % at least once),
 * edge weight "binary" | "log" (log(1+n)) | "count" (n full views). Only nodes with at least one edge
 * are materialised, so there is no zero-degree node and no dangling mass to re-inject (asserted);
 * users without an edge receive the shared fallback.
 * As in week 6 the square adjacency is never built: two dense blocks are used,
 *     T_ui = row-normalised W (user -> programme),  T_iu = row-normalised W^T (programme -> user),
 * and a batch of B users is propagated at once (one restart one-hot row per user):
 *     P_items <- alpha * P_users T_ui
 *     P_users <- alpha * P_items T_iu + (1 - alpha) * V        (alpha = link-following probability)
 * starting from P_users = V, P_items = 0; iterate until every row's L1 change < tol or maxIter.
 * Score = the programme block of the stationary vector. A plain-JS float64 implementation
 * (ppr.js) is kept as an independent oracle; the self-checks compare the two.
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  function buildBlocks(ctx, weight) {
    const wOf = weight === "count" ? (n) => n : weight === "log" ? (n) => Math.log1p(n) : () => 1;
    const uMap = new Int32Array(ctx.nUsers).fill(-1), iMap = new Int32Array(ctx.nItems).fill(-1);
    const users = [], items = [];
    for (let u = 0; u < ctx.nUsers; u++) for (let q = ctx.aggPtr[u]; q < ctx.aggPtr[u + 1]; q++) {
      if (ctx.aN[q] === 0) continue;
      if (uMap[u] < 0) { uMap[u] = users.length; users.push(u); }
      const i = ctx.aItem[q];
      if (iMap[i] < 0) { iMap[i] = items.length; items.push(i); }
    }
    const nU = users.length, nI = items.length;
    const W = new Float32Array(nU * nI);
    let edges = 0;
    for (let u = 0; u < ctx.nUsers; u++) for (let q = ctx.aggPtr[u]; q < ctx.aggPtr[u + 1]; q++) {
      if (ctx.aN[q] === 0) continue;
      W[uMap[u] * nI + iMap[ctx.aItem[q]]] = wOf(ctx.aN[q]); edges++;
    }
    return { uMap, iMap, users, items, nU, nI, W, edges };
  }

  async function makePPRTF(ctx, opts) {
    const tf = root.tf;
    const B = buildBlocks(ctx, opts.weight);
    const alpha = opts.alpha, tol = opts.tol ?? 1e-6, maxIter = opts.maxIter ?? 500;
    // tf.tidy disposes every intermediate; only the two transition blocks and the check survive
    const [Tui, Tiu, zeroDegT] = tf.tidy(() => {
      const Wt = tf.tensor2d(B.W, [B.nU, B.nI]);
      const rowU = tf.sum(Wt, 1, true), rowI = tf.sum(Wt, 0, true);
      const zero = tf.add(tf.sum(tf.equal(rowU, 0)), tf.sum(tf.equal(rowI, 0)));
      return [tf.div(Wt, rowU), tf.transpose(tf.div(Wt, rowI)), zero];   // nU x nI and nI x nU, rows sum to 1
    });
    const zeroDeg = (await zeroDegT.data())[0];
    zeroDegT.dispose();
    if (zeroDeg !== 0) { tf.dispose([Tui, Tiu]); throw new Error("zero-degree node in the materialised graph"); }
    const scores = new Map();
    const conv = { runs: 0, notConverged: 0, iterSum: 0, maxResidual: 0, batches: 0 };

    return {
      blocks: { nU: B.nU, nI: B.nI, edges: B.edges, uMap: B.uMap, iMap: B.iMap }, conv,
      userScores: new Map(),                               // user-block rows, kept only when prepare(..., keepUsers)
      describe: `alpha=${alpha}, weight=${opts.weight}, tol=${tol}, maxIter=${maxIter}`,
      async prepare(users, batchSize = 1200, keepUsers = false) {
        const todo = users.filter((u) => B.uMap[u] >= 0 && !scores.has(u));
        for (let b = 0; b < todo.length; b += batchSize) {
          const part = todo.slice(b, b + batchSize);
          const V = tf.tidy(() => tf.oneHot(tf.tensor1d(part.map((u) => B.uMap[u]), "int32"), B.nU).toFloat());
          let pU = V.clone(), pI = tf.zeros([part.length, B.nI]);
          let it = 0, residual = Infinity;
          while (it < maxIter) {
            it++;
            const [nU, nI, res] = tf.tidy(() => {
              const nI_ = tf.mul(alpha, tf.matMul(pU, Tui));
              const nU_ = tf.add(tf.mul(alpha, tf.matMul(pI, Tiu)), tf.mul(1 - alpha, V));
              const r = tf.max(tf.add(tf.sum(tf.abs(tf.sub(nU_, pU)), 1), tf.sum(tf.abs(tf.sub(nI_, pI)), 1)));
              return [nU_, nI_, r];
            });
            residual = (await res.data())[0];
            tf.dispose([pU, pI, res]);
            pU = nU; pI = nI;
            if (residual < tol) break;
          }
          const data = await pI.data();
          part.forEach((u, j) => scores.set(u, data.subarray(j * B.nI, (j + 1) * B.nI)));
          if (keepUsers) {
            const du = await pU.data();
            part.forEach((u, j) => this.userScores.set(u, du.subarray(j * B.nU, (j + 1) * B.nU)));
          }
          conv.runs += part.length; conv.batches++; conv.iterSum += it * part.length;
          if (residual >= tol) conv.notConverged += part.length;
          conv.maxResidual = Math.max(conv.maxResidual, residual);
          tf.dispose([V, pU, pI]);
        }
      },
      scoreUser(u) {
        if (B.uMap[u] < 0) return null;                    // no training edge -> shared fallback
        const row = scores.get(u);
        if (!row) throw new Error("call prepare(users) before scoreUser");
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (const i of ctx.candidateList) { const j = B.iMap[i]; if (j >= 0 && row[j] > 0) s[i] = row[j]; }
        return s;
      },
      /** Raw programme-block row (compact programme index) for a prepared user. */
      itemRow(u) { return scores.get(u) || null; },
      dispose() { tf.dispose([Tui, Tiu]); scores.clear(); this.userScores.clear(); },
    };
  }

  root.ExamPPRTF = { buildBlocks, makePPRTF };
})(typeof window !== "undefined" ? window : globalThis);

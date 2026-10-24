/**
 * mf.js — Method 3: matrix factorization for implicit feedback (course week 5: user and item
 * embeddings scored by a dot product), trained with alternating least squares (ALS) as in
 * Hu, Koren & Volinsky (ICDM 2008).
 *
 * Why not the week-5 Adam/MSE recipe: there are no ratings here, only views. Fitting only the
 * observed cells would make every observed target "1" and the trivial solution wins; implicit MF
 * therefore uses ALL user x item cells with a preference p_ui (1 if watched, else 0) and a
 * confidence c_ui that grows with the amount of watching:
 *     loss = sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda (sum ||x_u||^2 + sum ||y_i||^2)
 *     r_ui = full views ("count"), or hours inside full views ("hours")
 *     c_ui = 1 + alpha * log(1 + r_ui)  ("log")   or   1 + alpha * r_ui  ("linear")
 * ALS fixes Y and solves every x_u exactly (k x k Cholesky), then fixes X and solves every y_i;
 * each half-step can only lower the loss, which the self-check verifies.
 * Items with no training interaction get no factor and therefore no score (shared fallback).
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  /** In-place Cholesky solve of A x = b for a symmetric positive-definite k x k matrix. */
  function choleskySolve(A, b, k) {
    for (let j = 0; j < k; j++) {
      let d = A[j * k + j];
      for (let p = 0; p < j; p++) d -= A[j * k + p] * A[j * k + p];
      if (d <= 0) throw new Error("matrix not positive definite");
      d = Math.sqrt(d);
      A[j * k + j] = d;
      for (let i = j + 1; i < k; i++) {
        let s = A[i * k + j];
        for (let p = 0; p < j; p++) s -= A[i * k + p] * A[j * k + p];
        A[i * k + j] = s / d;
      }
    }
    for (let i = 0; i < k; i++) { let s = b[i]; for (let p = 0; p < i; p++) s -= A[i * k + p] * b[p]; b[i] = s / A[i * k + i]; }
    for (let i = k - 1; i >= 0; i--) { let s = b[i]; for (let p = i + 1; p < k; p++) s -= A[p * k + i] * b[p]; b[i] = s / A[i * k + i]; }
    return b;
  }

  /** Build the sparse confidence matrix (CSR by user and by item) from the context. */
  function buildMatrix(ctx, opts) {
    const rOf = opts.signal === "hours" ? (k) => ctx.aM[k] / 60 : (k) => ctx.aN[k];
    const cOf = opts.confidence === "linear" ? (r) => 1 + opts.alpha * r : (r) => 1 + opts.alpha * Math.log1p(r);
    const uPtr = new Int32Array(ctx.nUsers + 1), uIdx = [], uConf = [];
    const itemHas = new Uint8Array(ctx.nItems);
    for (let u = 0; u < ctx.nUsers; u++) {
      for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) {
        if (ctx.aN[k] === 0) continue;               // preference = watched >= 80 % at least once
        uIdx.push(ctx.aItem[k]); uConf.push(cOf(rOf(k))); itemHas[ctx.aItem[k]] = 1;
      }
      uPtr[u + 1] = uIdx.length;
    }
    const nnz = uIdx.length;
    const iPtr = new Int32Array(ctx.nItems + 1);
    for (const i of uIdx) iPtr[i + 1]++;
    for (let i = 0; i < ctx.nItems; i++) iPtr[i + 1] += iPtr[i];
    const fill = iPtr.slice(0, ctx.nItems), iIdx = new Int32Array(nnz), iConf = new Float64Array(nnz);
    for (let u = 0; u < ctx.nUsers; u++) for (let q = uPtr[u]; q < uPtr[u + 1]; q++) { const i = uIdx[q]; iIdx[fill[i]] = u; iConf[fill[i]] = uConf[q]; fill[i]++; }
    return { uPtr, uIdx: Int32Array.from(uIdx), uConf: Float64Array.from(uConf), iPtr, iIdx, iConf, nnz, itemHas };
  }

  function gram(F, n, k) {
    const G = new Float64Array(k * k);
    for (let r = 0; r < n; r++) {
      const o = r * k;
      for (let a = 0; a < k; a++) { const fa = F[o + a]; if (fa === 0) continue; for (let b = a; b < k; b++) G[a * k + b] += fa * F[o + b]; }
    }
    for (let a = 0; a < k; a++) for (let b = 0; b < a; b++) G[a * k + b] = G[b * k + a];
    return G;
  }

  /** One ALS half-step: solve every row of `out` given the fixed factors `fixed`. */
  function solveSide(out, nOut, fixed, nFixed, ptr, idx, conf, k, lambda) {
    const G = gram(fixed, nFixed, k);
    const A = new Float64Array(k * k), b = new Float64Array(k);
    for (let r = 0; r < nOut; r++) {
      const o = r * k;
      if (ptr[r + 1] === ptr[r]) { for (let a = 0; a < k; a++) out[o + a] = 0; continue; }
      A.set(G);
      for (let a = 0; a < k; a++) A[a * k + a] += lambda;
      b.fill(0);
      for (let q = ptr[r]; q < ptr[r + 1]; q++) {
        const f = idx[q] * k, c = conf[q];
        for (let a = 0; a < k; a++) {
          const ya = fixed[f + a];
          b[a] += c * ya;                            // p_ui = 1 on observed cells
          const w = (c - 1) * ya;
          for (let bb = a; bb < k; bb++) A[a * k + bb] += w * fixed[f + bb];
        }
      }
      for (let a = 0; a < k; a++) for (let bb = 0; bb < a; bb++) A[a * k + bb] = A[bb * k + a];
      choleskySolve(A, b, k);
      for (let a = 0; a < k; a++) out[o + a] = b[a];
    }
  }

  /** Exact training loss using sum_all (x.y)^2 = <X'X, Y'Y>_F (no dense matrix needed). */
  function loss(M, X, Y, nU, nI, k, lambda) {
    const GX = gram(X, nU, k), GY = gram(Y, nI, k);
    let total = 0;
    for (let a = 0; a < k * k; a++) total += GX[a] * GY[a];
    for (let u = 0; u < nU; u++) for (let q = M.uPtr[u]; q < M.uPtr[u + 1]; q++) {
      const i = M.uIdx[q], c = M.uConf[q];
      let s = 0;
      for (let a = 0; a < k; a++) s += X[u * k + a] * Y[i * k + a];
      total += c * (1 - s) * (1 - s) - s * s;
    }
    let reg = 0;
    for (let a = 0; a < nU * k; a++) reg += X[a] * X[a];
    for (let a = 0; a < nI * k; a++) reg += Y[a] * Y[a];
    return total + lambda * reg;
  }

  /**
   * Train implicit ALS. opts: {factors, lambda, alpha, iterations, signal, confidence, seed}
   * @returns model with scoreUser(u) and a per-iteration history of the loss
   */
  function trainALS(ctx, opts, onIteration) {
    const k = opts.factors, lambda = opts.lambda;
    const M = buildMatrix(ctx, opts);
    const rand = C.mulberry32(opts.seed ?? 42);
    const X = new Float64Array(ctx.nUsers * k), Y = new Float64Array(ctx.nItems * k);
    for (let i = 0; i < ctx.nItems; i++) if (M.itemHas[i]) for (let a = 0; a < k; a++) Y[i * k + a] = 0.01 * C.gaussian(rand);
    const history = [];
    for (let it = 1; it <= opts.iterations; it++) {
      solveSide(X, ctx.nUsers, Y, ctx.nItems, M.uPtr, M.uIdx, M.uConf, k, lambda);
      const afterUsers = opts.trackLoss ? loss(M, X, Y, ctx.nUsers, ctx.nItems, k, lambda) : null;
      solveSide(Y, ctx.nItems, X, ctx.nUsers, M.iPtr, M.iIdx, M.iConf, k, lambda);
      const afterItems = opts.trackLoss ? loss(M, X, Y, ctx.nUsers, ctx.nItems, k, lambda) : null;
      history.push({ iteration: it, lossAfterUserStep: afterUsers, lossAfterItemStep: afterItems });
      if (onIteration) onIteration(it);
    }
    let factorNorm = 0;
    for (let a = 0; a < Y.length; a++) factorNorm += Y[a] * Y[a];
    return {
      X, Y, k, M, history, factorNorm: Math.sqrt(factorNorm),
      describe: `factors=${k}, lambda=${lambda}, alpha=${opts.alpha}, signal=${opts.signal}, confidence=${opts.confidence}, iterations=${opts.iterations}, seed=${opts.seed ?? 42}`,
      scoreUser(u) {
        if (M.uPtr[u + 1] === M.uPtr[u]) return null;  // no training views -> shared fallback
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (const i of ctx.candidateList) {
          if (!M.itemHas[i]) continue;
          let d = 0;
          for (let a = 0; a < k; a++) d += X[u * k + a] * Y[i * k + a];
          s[i] = d;
        }
        return s;
      },
    };
  }

  root.ExamMF = { choleskySolve, buildMatrix, trainALS, loss };
})(typeof window !== "undefined" ? window : globalThis);

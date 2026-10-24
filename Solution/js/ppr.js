/**
 * ppr.js — Method 4: Personalized PageRank (course week 6).
 *
 * Graph: bipartite user-programme graph from the training weeks; an edge u <-> i exists when user u
 * watched programme i >= 80 % at least once. Two programmes are linked through their common viewers,
 * so the walk follows the co-view structure. Edge weight w_ui: "binary" (1), "log" (log(1+n)) or
 * "count" (n full views). Each node's out-edges are normalised to probabilities.
 * Update (week-6 convention, alpha = probability of FOLLOWING a link, 1 - alpha = restart):
 *     p_next = alpha * (Q^T p + d * v) + (1 - alpha) * v
 * Q = row-normalised transitions (zero rows for dangling nodes), d = mass on dangling nodes,
 * v = one-hot restart at the target user. Start p = v; stop when ||p_next - p||_1 < tol or after
 * maxIter iterations; convergence is reported, never assumed. Score = programme-node probability.
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  /**
   * Generic power iteration on a CSR transition graph (used by the bipartite PPR and by the week-6
   * reference-graph test). out: {ptr, idx, prob}; restart: Float64Array summing to 1.
   */
  function powerIteration(g, n, alpha, restart, tol = 1e-6, maxIter = 500) {
    let p = Float64Array.from(restart), next = new Float64Array(n);
    let it = 0, residual = Infinity;
    while (it < maxIter) {
      it++;
      let d = 0;
      next.fill(0);
      for (let j = 0; j < n; j++) {
        const pj = p[j];
        if (pj === 0) continue;
        if (g.ptr[j + 1] === g.ptr[j]) { d += pj; continue; }
        for (let q = g.ptr[j]; q < g.ptr[j + 1]; q++) next[g.idx[q]] += alpha * pj * g.prob[q];
      }
      residual = 0;
      for (let j = 0; j < n; j++) {
        if (restart[j] !== 0) next[j] += (alpha * d + (1 - alpha)) * restart[j];
        residual += Math.abs(next[j] - p[j]);
      }
      const t = p; p = next; next = t;
      if (residual < tol) break;
    }
    return { p, iterations: it, residual, converged: residual < tol };
  }

  /** Bipartite graph: nodes 0..nUsers-1 are users, nUsers + i is programme i. */
  function buildGraph(ctx, weight) {
    const wOf = weight === "count" ? (n) => n : weight === "log" ? (n) => Math.log1p(n) : () => 1;
    const n = ctx.nUsers + ctx.nItems;
    const adj = Array.from({ length: n }, () => []);
    let edges = 0;
    for (let u = 0; u < ctx.nUsers; u++) for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) {
      if (ctx.aN[k] === 0) continue;
      const w = wOf(ctx.aN[k]), it = ctx.nUsers + ctx.aItem[k];
      adj[u].push([it, w]); adj[it].push([u, w]); edges++;
    }
    const ptr = new Int32Array(n + 1), idx = [], prob = [];
    let dangling = 0;
    for (let j = 0; j < n; j++) {
      const tot = adj[j].reduce((s, e) => s + e[1], 0);
      if (adj[j].length === 0) dangling++;
      for (const [k, w] of adj[j]) { idx.push(k); prob.push(w / tot); }
      ptr[j + 1] = idx.length;
    }
    return { n, ptr, idx: Int32Array.from(idx), prob: Float64Array.from(prob), edges, dangling };
  }

  function makePPR(ctx, opts) {
    const g = buildGraph(ctx, opts.weight);
    const tol = opts.tol ?? 1e-6, maxIter = opts.maxIter ?? 500;
    const restart = new Float64Array(g.n);
    const conv = { runs: 0, notConverged: 0, iterSum: 0, maxResidual: 0 };
    return {
      graph: g, conv,
      describe: `alpha=${opts.alpha}, weight=${opts.weight}, tol=${tol}, maxIter=${maxIter}`,
      run(u) {
        restart.fill(0); restart[u] = 1;
        const r = powerIteration(g, g.n, opts.alpha, restart, tol, maxIter);
        conv.runs++; conv.iterSum += r.iterations; if (!r.converged) conv.notConverged++;
        conv.maxResidual = Math.max(conv.maxResidual, r.residual);
        return r;
      },
      scoreUser(u) {
        if (g.ptr[u + 1] === g.ptr[u]) return null;    // no edges: PPR would equal the restart -> fallback
        const r = this.run(u);
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (const i of ctx.candidateList) { const v = r.p[ctx.nUsers + i]; if (v > 0) s[i] = v; }
        return s;
      },
    };
  }

  root.ExamPPR = { powerIteration, buildGraph, makePPR };
})(typeof window !== "undefined" ? window : globalThis);

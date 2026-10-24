/**
 * methods.js — one registry of every recommender with its validation grid, shared by the page
 * and tools/run_experiments.js so both search exactly the same settings.
 *
 * Each entry: key, title, make(ctx, opts) -> model (or Promise of a model; TF.js models also expose
 * async prepare(users) that scores a batch of users before scoreUser(u)), grid() -> list of opts,
 * stats(model) -> optional diagnostics, isMethod (false for the baseline / reference).
 */
(function (root) {
  "use strict";
  const cartesian = (spec) => Object.entries(spec).reduce(
    (acc, [k, vals]) => acc.flatMap((o) => vals.map((v) => ({ ...o, [k]: v }))), [{}]);

  const METHODS = [
    {
      key: "popularity", title: "Popularity baseline", isMethod: false,
      make: (ctx, o) => root.ExamPopularity.makePopularity(ctx, o),
      grid: () => cartesian({ count: ["views", "viewers"], recentWeeks: [0, 4, 2] }),
      defaults: { count: "views", recentWeeks: 0 },
    },
    {
      key: "content", title: "Content-based filtering", isMethod: true,
      make: (ctx, o) => root.ExamContent.makeContentBased(ctx, { weights: { meta: o.meta, people: o.people, channel: o.channel, time: o.time }, profile: o.profile }),
      grid: () => cartesian({ meta: [1], people: [0, 0.5], channel: [0, 1, 2], time: [0, 1, 2], profile: ["log", "linear"] }),
      defaults: { meta: 1, people: 0, channel: 1, time: 1, profile: "log" },
    },
    {
      key: "rules", title: "Association rules", isMethod: true,
      make: (ctx, o) => root.ExamRules.makeRules(ctx, o),
      stats: (m) => m.mined.stats,
      grid: () => cartesian({ mode: ["same-week", "next-week"], minSupport: [0.0005, 0.001, 0.002, 0.005], minConfidence: [0.05, 0.1, 0.2], agg: ["max", "sum"], lookback: [1, 4, 10] }),
      defaults: { mode: "next-week", minSupport: 0.002, minConfidence: 0.1, agg: "max", lookback: 4 },
    },
    {
      key: "mf", title: "Matrix factorization (TensorFlow.js, implicit)", isMethod: true,
      make: (ctx, o, run) => root.ExamMFTF.trainMFTF(ctx, o, null, run),
      stats: (m) => ({ users: m.nU, items: m.nI, positives: m.positives, batchesPerEpoch: m.batches, adamUpdates: m.updates, finalLoss: m.finalLoss && +m.finalLoss.toFixed(1), factorNorm: +m.factorNorm.toFixed(4) }),
      // staged search: 1) lambda x alpha at k=32 (they interact);  2) signal;  3) number of factors;
      // 4) HW5-style mini-batches (batch size x epochs) against the full batch selected so far
      grid: (best) => {
        const base = { lr: 0.05, epochs: 100, batchSize: 8192, seed: 42 };
        if (!best) return cartesian({ factors: [32], lambda: [10, 50, 100], alpha: [1, 5], signal: ["hours"], ...Object.fromEntries(Object.entries(base).map(([k, v]) => [k, [v]])) });
        const { stage, ...b } = best;
        if (stage === 2) return [{ ...b, signal: "count" }, { ...b }];
        if (stage === 3) return [16, 32, 64].map((f) => ({ ...b, factors: f }));
        return [{ ...b }, { ...b, batchSize: 1024, epochs: 20 }, { ...b, batchSize: 1024, epochs: 100 }, { ...b, batchSize: 256, epochs: 20 }];
      },
      stages: 4,
      robustChunks: 1,
      defaults: { factors: 32, lambda: 50, alpha: 1, signal: "hours", lr: 0.05, epochs: 100, batchSize: 8192, seed: 42 },
    },
    {
      key: "ppr", title: "Personalized PageRank (TensorFlow.js)", isMethod: true,
      make: (ctx, o) => root.ExamPPRTF.makePPRTF(ctx, o),
      stats: (m) => ({ users: m.blocks.nU, items: m.blocks.nI, edges: m.blocks.edges, runs: m.conv.runs, notConverged: m.conv.notConverged, meanIterations: m.conv.runs ? +(m.conv.iterSum / m.conv.runs).toFixed(1) : 0, maxResidual: m.conv.maxResidual }),
      grid: () => cartesian({ alpha: [0.15, 0.3, 0.5, 0.7, 0.85], weight: ["binary", "log", "count"], tol: [1e-6], maxIter: [500] }),
      robustChunks: 3,
      defaults: { alpha: 0.5, weight: "binary", tol: 1e-6, maxIter: 500 },
    },
    {
      key: "mf_als", title: "Reference implementation: implicit MF by ALS (plain JS, float64)", isMethod: false, reference: true,
      make: (ctx, o) => root.ExamMF.trainALS(ctx, o),
      stats: (m) => ({ nnz: m.M.nnz, factorNorm: +m.factorNorm.toFixed(4) }),
      grid: (best) => {
        if (!best) return cartesian({ factors: [32], lambda: [1], alpha: [1, 5, 20, 40], signal: ["count", "hours"], confidence: ["log", "linear"], iterations: [10], seed: [42] });
        const keep = { signal: [best.signal], confidence: [best.confidence], seed: [42] };
        if (best.stage === 2) return cartesian({ factors: [32], lambda: [0.1, 1, 10, 50, 100, 200, 500], alpha: [0.25, 1, 5], ...keep, iterations: [10] });
        if (best.stage === 3) return cartesian({ factors: [16, 32, 64], lambda: [best.lambda], alpha: [best.alpha], ...keep, iterations: [10] });
        const { stage, ...b } = best;
        return [5, 15, 20].map((it) => ({ ...b, iterations: it, seed: 42 })).concat([7, 2026].map((sd) => ({ ...b, seed: sd })));
      },
      stages: 4,
      defaults: { factors: 32, lambda: 1, alpha: 5, signal: "count", confidence: "log", iterations: 10, seed: 42 },
    },
    {
      key: "ppr_js", title: "Reference implementation: PPR by sparse power iteration (plain JS, float64)", isMethod: false, reference: true,
      make: (ctx, o) => root.ExamPPR.makePPR(ctx, o),
      stats: (m) => ({ edges: m.graph.edges, runs: m.conv.runs, notConverged: m.conv.notConverged, meanIterations: m.conv.runs ? +(m.conv.iterSum / m.conv.runs).toFixed(1) : 0, maxResidual: m.conv.maxResidual }),
      grid: () => cartesian({ alpha: [0.15, 0.3, 0.5, 0.7, 0.85], weight: ["binary", "log", "count"], tol: [1e-6], maxIter: [500] }),
      defaults: { alpha: 0.5, weight: "binary", tol: 1e-6, maxIter: 500 },
    },
    {
      key: "own", title: "Reference: own viewing history", isMethod: false,
      make: (ctx) => root.ExamPopularity.makeOwnHistory(ctx),
      grid: () => [{}],
      defaults: {},
    },
  ];

  root.ExamMethods = { METHODS, cartesian, byKey: (k) => METHODS.find((m) => m.key === k) };
})(typeof window !== "undefined" ? window : globalThis);

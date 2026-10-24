/**
 * rules.js — Method 2: association rules (course week 4: Apriori, support / confidence / lift).
 *
 * Transactions ("baskets"): one basket per (user, week) = the programmes the user watched
 * >= 80 % in that training week (a basket is a SET; counts within a week are ignored).
 * Two rule families, chosen on validation:
 *   "same-week": classic rules A -> B, A and B in the same basket.
 *   "next-week": time-window rules A(week t) -> B(week t+1) of the same user; this allows
 *                A -> A ("watched it this week -> watches it next week"), i.e. habit as a rule.
 * Definitions (N = number of transactions):
 *   support(A->B) = count(A and B) / N,  confidence = count(A and B) / count(A),
 *   lift = confidence / (count(B) / N).   Rules kept: support >= minSupport,
 *   confidence >= minConfidence, lift > 1 (as in HW4).
 * Frequent itemsets are mined Apriori-style up to size 2 (downward closure: a pair can only be
 * frequent if both items are); single-antecedent rules are what the scorer uses.
 * Scoring for user u: antecedents = programmes the user watched in the last `lookback` training
 * weeks; score(B) = max_A confidence(A->B) ("max") or sum_A confidence(A->B) ("sum").
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  /** Build transactions from the training rows of ctx. Returns {ante[], cons[]} arrays of Int32Array sets. */
  function buildTransactions(ctx, mode) {
    const ante = [], cons = [];
    for (let u = 0; u < ctx.nUsers; u++) {
      const byWeek = new Map();
      for (let k = ctx.rowPtr[u]; k < ctx.rowPtr[u + 1]; k++) {
        if (ctx.rowN[k] === 0) continue;
        const w = ctx.rowWeek[k];
        if (!byWeek.has(w)) byWeek.set(w, new Set());
        byWeek.get(w).add(ctx.rowItem[k]);
      }
      if (mode === "same-week") {
        for (const s of byWeek.values()) { const a = Int32Array.from([...s].sort((x, y) => x - y)); ante.push(a); cons.push(a); }
      } else {
        for (let w = ctx.train[0]; w < ctx.train[1]; w++) {
          const a = byWeek.get(w);
          if (!a) continue;                          // transaction exists when the user watched in week t
          ante.push(Int32Array.from([...a].sort((x, y) => x - y)));
          cons.push(Int32Array.from([...(byWeek.get(w + 1) || [])].sort((x, y) => x - y)));
        }
      }
    }
    return { ante, cons, N: ante.length };
  }

  /**
   * Apriori level 1 + level 2 with downward-closure pruning, then single-antecedent rules.
   * @returns {{rules: Map<number, Array>, stats: object}}  rules keyed by antecedent item
   */
  function mineRules(ctx, mode, minSupport, minConfidence) {
    const T = buildTransactions(ctx, mode);
    const minCount = Math.max(1, Math.ceil(minSupport * T.N));
    const cntA = new Int32Array(ctx.nItems), cntB = new Int32Array(ctx.nItems);
    for (let t = 0; t < T.N; t++) { for (const a of T.ante[t]) cntA[a]++; for (const b of T.cons[t]) cntB[b]++; }
    // Level 1: frequent single items (antecedent side and consequent side).
    const freqA = new Uint8Array(ctx.nItems), freqB = new Uint8Array(ctx.nItems);
    let nFreqA = 0, nFreqB = 0;
    for (let i = 0; i < ctx.nItems; i++) { if (cntA[i] >= minCount) { freqA[i] = 1; nFreqA++; } if (cntB[i] >= minCount) { freqB[i] = 1; nFreqB++; } }
    // Level 2: count only candidate pairs whose members are both frequent (downward closure).
    const pair = new Map();
    for (let t = 0; t < T.N; t++) {
      const A = T.ante[t].filter((a) => freqA[a]), B = T.cons[t].filter((b) => freqB[b]);
      for (const a of A) for (const b of B) {
        if (mode === "same-week" && a === b) continue;
        const key = a * 65536 + b;
        pair.set(key, (pair.get(key) || 0) + 1);
      }
    }
    const rules = new Map();
    let nFrequentPairs = 0, nRules = 0;
    for (const [key, c] of pair) {
      if (c < minCount) continue;
      nFrequentPairs++;
      const a = Math.floor(key / 65536), b = key % 65536;
      const support = c / T.N, confidence = c / cntA[a], lift = confidence / (cntB[b] / T.N);
      if (confidence < minConfidence || !(lift > 1)) continue;
      if (!rules.has(a)) rules.set(a, []);
      rules.get(a).push({ a, b, count: c, countA: cntA[a], countB: cntB[b], support, confidence, lift });
      nRules++;
    }
    return { rules, stats: { mode, transactions: T.N, minCount, frequentItemsA: nFreqA, frequentItemsB: nFreqB, frequentPairs: nFrequentPairs, rules: nRules } };
  }

  function makeRules(ctx, opts) {
    const mined = mineRules(ctx, opts.mode, opts.minSupport, opts.minConfidence);
    const lookback = opts.lookback || 4;
    const from = ctx.train[1] - lookback + 1;
    return {
      mined,
      describe: `mode=${opts.mode}, minSupport=${opts.minSupport}, minConfidence=${opts.minConfidence}, agg=${opts.agg}, lookback=${lookback}`,
      antecedents(u) {
        const set = new Set();
        for (let k = ctx.rowPtr[u]; k < ctx.rowPtr[u + 1]; k++) if (ctx.rowN[k] > 0 && ctx.rowWeek[k] >= from) set.add(ctx.rowItem[k]);
        return set;
      },
      scoreUser(u) {
        const A = this.antecedents(u);
        if (A.size === 0) return null;
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (const a of A) {
          const rs = mined.rules.get(a);
          if (!rs) continue;
          for (const r of rs) {
            if (!ctx.isCandidate[r.b]) continue;
            const cur = s[r.b] === C.NO_SCORE ? 0 : s[r.b];
            s[r.b] = opts.agg === "sum" ? cur + r.confidence : Math.max(cur, r.confidence);
          }
        }
        return s;
      },
    };
  }

  root.ExamRules = { buildTransactions, mineRules, makeRules };
})(typeof window !== "undefined" ? window : globalThis);

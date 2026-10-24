/**
 * content.js — Method 1: content-based filtering (course week 2: item vectors, cosine, profile).
 *
 * Item vector x_i = concatenation of feature groups, each group L2-normalised and multiplied by
 * its weight, then the whole vector L2-normalised (so the dot product is the cosine):
 *   meta    : category one-hot + genre multi-hot + decade of production (catalogue metadata)
 *   people  : directors / actors multi-hot
 *   channel : channels the programme airs on (training + target window schedule)
 *   time    : share of airings in each 3-hour slot of the day (training + target window schedule)
 * Only the schedule is used for channel/time — never who watched what in the target window.
 * User profile  P_u = sum_i f(n_ui) * x_i / sum_i f(n_ui)  over programmes watched >= 80 %,
 *   f = log(1+n) ("log") or n ("linear")  — the week-2 "average of liked items" profile.
 * Score s_ui = cos(P_u, x_i) for every candidate.
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;
  const GROUPS = ["meta", "people", "channel", "time"];

  /** Build sparse item vectors for the context's training + target schedule. */
  function buildItemVectors(ctx, weights) {
    const E = ctx.E, I = E.items;
    const featIndex = new Map();
    const fid = (key) => { let k = featIndex.get(key); if (k === undefined) { k = featIndex.size; featIndex.set(key, k); } return k; };
    const wins = ctx.mode === "valid" ? ["train", "valid"] : ["train", "valid", "test"];
    const ptr = [0], idx = [], val = [];
    for (let i = 0; i < ctx.nItems; i++) {
      const groups = { meta: new Map(), people: new Map(), channel: new Map(), time: new Map() };
      if (I.category[i]) groups.meta.set(fid("cat:" + I.category[i]), 1);
      for (const g of I.genres[i]) groups.meta.set(fid("gen:" + g.toLowerCase()), 1);
      if (I.year[i]) groups.meta.set(fid("dec:" + Math.floor(I.year[i] / 10) * 10), 1);
      for (const p of [...I.directors[i], ...I.actors[i]]) groups.people.set(fid("p:" + p.toLowerCase()), 1);
      for (const w of wins) {
        for (const c of E.chans[w][i]) groups.channel.set(fid("ch:" + c), 1);
        for (let b = 0; b < 8; b++) {
          const h = E.hours[w][i * 8 + b];
          if (h) { const f = fid("hr:" + b); groups.time.set(f, (groups.time.get(f) || 0) + h); }
        }
      }
      const entries = new Map();
      for (const g of GROUPS) {
        const wg = weights[g] || 0;
        if (wg === 0 || groups[g].size === 0) continue;
        let norm = 0;
        for (const v of groups[g].values()) norm += v * v;
        norm = Math.sqrt(norm);
        for (const [f, v] of groups[g]) entries.set(f, (wg * v) / norm);
      }
      let tot = 0;
      for (const v of entries.values()) tot += v * v;
      tot = Math.sqrt(tot) || 1;
      for (const [f, v] of entries) { idx.push(f); val.push(v / tot); }
      ptr.push(idx.length);
    }
    return { ptr: Int32Array.from(ptr), idx: Int32Array.from(idx), val: Float64Array.from(val), nFeat: featIndex.size, featIndex };
  }

  /**
   * @param {object} ctx  ExamCore context
   * @param {{weights:object, profile:"log"|"linear"}} opts
   */
  function makeContentBased(ctx, opts) {
    const V = buildItemVectors(ctx, opts.weights);
    const f = opts.profile === "linear" ? (n) => n : (n) => Math.log1p(n);
    const P = new Float64Array(V.nFeat);
    return {
      vectors: V,
      getProfile: () => P,
      describe: `weights=${JSON.stringify(opts.weights)}, profile=${opts.profile}`,
      profile(u) {
        P.fill(0);
        let wsum = 0;
        for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) {
          if (ctx.aN[k] === 0) continue;
          const i = ctx.aItem[k], w = f(ctx.aN[k]);
          wsum += w;
          for (let q = V.ptr[i]; q < V.ptr[i + 1]; q++) P[V.idx[q]] += w * V.val[q];
        }
        if (wsum === 0) return null;
        let norm = 0;
        for (let j = 0; j < P.length; j++) { P[j] /= wsum; norm += P[j] * P[j]; }
        return Math.sqrt(norm);
      },
      scoreUser(u) {
        const pn = this.profile(u);
        if (!pn) return null;                       // no full views in training -> shared fallback
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (const i of ctx.candidateList) {
          let dot = 0;
          for (let q = V.ptr[i]; q < V.ptr[i + 1]; q++) dot += P[V.idx[q]] * V.val[q];
          s[i] = V.ptr[i + 1] > V.ptr[i] ? dot / pn : C.NO_SCORE;   // item vectors are unit length
        }
        return s;
      },
    };
  }

  root.ExamContent = { GROUPS, buildItemVectors, makeContentBased };
})(typeof window !== "undefined" ? window : globalThis);

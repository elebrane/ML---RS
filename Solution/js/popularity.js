/**
 * popularity.js — the REQUIRED popularity baseline (not one of the four methods),
 * plus the "own history" reference used to read every method's numbers.
 *
 * Popularity: the same five candidates for every user.
 *   count  = "views"   → total airings watched >= 80 % in the training weeks
 *          = "viewers" → number of distinct users with at least one such airing
 *   recentWeeks = 0 → all training weeks; W > 0 → only the last W training weeks
 * Own history: user's own items scored by their full-view count (not a course method;
 *   a reference that shows how much of the target is plain repeat viewing).
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  const popularityScores = (ctx, opts = {}) => C.popularityVector(ctx, opts);

  /** Popularity recommender: same score vector for every user. */
  function makePopularity(ctx, opts) {
    const s = popularityScores(ctx, opts);
    return { scoreUser: () => s, describe: `count=${opts.count}, recentWeeks=${opts.recentWeeks || "all"}` };
  }

  /** Own-history reference: user's training full views of programmes that still air. */
  function makeOwnHistory(ctx) {
    return {
      scoreUser(u) {
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) if (ctx.aN[k] > 0) s[ctx.aItem[k]] = ctx.aN[k];
        return s;
      },
      describe: "own full-view counts, padded with popularity",
    };
  }

  root.ExamPopularity = { popularityScores, makePopularity, makeOwnHistory };
})(typeof window !== "undefined" ? window : globalThis);

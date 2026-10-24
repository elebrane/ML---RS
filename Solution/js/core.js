/**
 * core.js — shared, method-independent logic for the sweet.tv exam.
 *
 * Plain script (no modules): attaches `ExamCore` to `window` in the browser and to
 * `globalThis` in Node, so the page and tools/run_experiments.js run identical code.
 *
 * Conventions used by every method (identical across methods, as the exam requires):
 *  - Context `valid`: fit on weeks 11-20, predict weeks 21-30, compare with the host-rule top-5.
 *    Context `test`:  fit on weeks 11-30, predict weeks 31-42 (no labels; Kaggle scores it).
 *  - Candidate pool: labelled programmes that AIR in the target window (schedule only).
 *  - Repeats are allowed: a programme the user already watched may be recommended.
 *  - A method returns a score per item (Float64Array, -Infinity = "no score").
 *    Top-5 = candidates sorted by score desc, ties by the shared popularity rank, then id.
 *    Unfilled slots (and users with no history) are padded from the shared popularity list.
 */
(function (root) {
  "use strict";

  const K = 5;
  const NO_SCORE = -Infinity;

  /** Mulberry32 seeded PRNG (same generator as the HW6 starter). */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /** Standard normal sample from a uniform PRNG (Box–Muller). */
  function gaussian(rand) {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  /** AP@K as in the course slides: sum of precision at relevant ranks / min(K, |R_u|). */
  function apk(actual, predicted, k = K) {
    if (!actual || actual.length === 0) return 0;
    const rel = new Set(actual);
    const seen = new Set();
    let hits = 0;
    let sum = 0;
    for (let r = 0; r < Math.min(k, predicted.length); r++) {
      const p = predicted[r];
      if (rel.has(p) && !seen.has(p)) {
        hits += 1;
        sum += hits / (r + 1);
      }
      seen.add(p);
    }
    return sum / Math.min(k, rel.size);
  }

  /**
   * Build a fitting/prediction context.
   * @param {object} E     window.EXAM
   * @param {"valid"|"test"} mode
   */
  function buildContext(E, mode) {
    const train = mode === "valid" ? [11, 20] : [11, 30];
    const target = mode === "valid" ? "valid" : "test";
    const nUsers = E.users.length;
    const nItems = E.items.id.length;
    const L = E.log;

    // Raw per-week rows inside the training weeks, grouped by user (log is sorted by user, week).
    const keep = [];
    for (let r = 0; r < L.u.length; r++) if (L.w[r] >= train[0] && L.w[r] <= train[1]) keep.push(r);
    const rowPtr = new Int32Array(nUsers + 1);
    for (const r of keep) rowPtr[L.u[r] + 1] += 1;
    for (let u = 0; u < nUsers; u++) rowPtr[u + 1] += rowPtr[u];
    const rowItem = new Int32Array(keep.length), rowWeek = new Int16Array(keep.length);
    const rowN = new Int32Array(keep.length), rowP = new Int32Array(keep.length), rowM = new Int32Array(keep.length);
    keep.forEach((r, k) => { rowItem[k] = L.i[r]; rowWeek[k] = L.w[r]; rowN[k] = L.n[r]; rowP[k] = L.p[r]; rowM[k] = L.m[r]; });

    // Aggregated per (user, item) over the training weeks.
    const aggPtr = new Int32Array(nUsers + 1);
    const aItem = [], aN = [], aP = [], aM = [], aLast = [];
    for (let u = 0; u < nUsers; u++) {
      const acc = new Map();
      for (let k = rowPtr[u]; k < rowPtr[u + 1]; k++) {
        const it = rowItem[k];
        let o = acc.get(it);
        if (!o) { o = { n: 0, p: 0, m: 0, last: 0 }; acc.set(it, o); }
        o.n += rowN[k]; o.p += rowP[k]; o.m += rowM[k];
        if (rowN[k] > 0 && rowWeek[k] > o.last) o.last = rowWeek[k];
      }
      const ids = [...acc.keys()].sort((a, b) => a - b);
      for (const it of ids) { const o = acc.get(it); aItem.push(it); aN.push(o.n); aP.push(o.p); aM.push(o.m); aLast.push(o.last); }
      aggPtr[u + 1] = aItem.length;
    }

    const airTarget = E.air[target];
    const candidateList = [];
    for (let i = 0; i < nItems; i++) if (airTarget[i] > 0) candidateList.push(i);
    const isCandidate = new Uint8Array(nItems);
    for (const i of candidateList) isCandidate[i] = 1;
    const airTrain = new Int32Array(nItems);
    const trainWins = mode === "valid" ? ["train"] : ["train", "valid"];
    for (const w of trainWins) for (let i = 0; i < nItems; i++) airTrain[i] += E.air[w][i];

    const ctx = {
      E, mode, train, target, nUsers, nItems,
      rowPtr, rowItem, rowWeek, rowN, rowP, rowM,
      aggPtr, aItem: Int32Array.from(aItem), aN: Int32Array.from(aN), aP: Int32Array.from(aP),
      aM: Int32Array.from(aM), aLast: Int16Array.from(aLast),
      airTarget, airTrain, candidateList, isCandidate,
    };
    ctx.fallback = sharedPopularity(ctx);            // shared fallback list (identical for all methods)
    ctx.fallbackRank = new Int32Array(nItems).fill(nItems);
    ctx.fallback.forEach((it, r) => { ctx.fallbackRank[it] = r; });
    ctx.cohort = evaluationCohort(ctx);
    return ctx;
  }

  /** Popularity setting used as the shared fallback = the baseline's validation-selected setting. */
  const FALLBACK = { count: "viewers", recentWeeks: 4 };

  /**
   * Popularity score per item from the training rows (candidates only; -Infinity elsewhere).
   *   count "views"   = total airings watched >= 80 %;  "viewers" = distinct users with one such airing
   *   recentWeeks 0   = all training weeks; W > 0 = the last W training weeks only
   */
  function popularityVector(ctx, opts = FALLBACK) {
    const count = opts.count || "views";
    const W = opts.recentWeeks || 0;
    const from = W > 0 ? ctx.train[1] - W + 1 : ctx.train[0];
    const s = new Float64Array(ctx.nItems).fill(NO_SCORE);
    for (const i of ctx.candidateList) s[i] = 0;
    for (let u = 0; u < ctx.nUsers; u++) {
      const seen = new Set();
      for (let k = ctx.rowPtr[u]; k < ctx.rowPtr[u + 1]; k++) {
        const i = ctx.rowItem[k];
        if (ctx.rowWeek[k] < from || ctx.rowN[k] === 0 || !ctx.isCandidate[i]) continue;
        if (count === "views") s[i] += ctx.rowN[k];
        else if (!seen.has(i)) { s[i] += 1; seen.add(i); }
      }
    }
    return s;
  }

  /** Shared fallback list: candidates by the fallback popularity, ties by item id. */
  function sharedPopularity(ctx) {
    const s = popularityVector(ctx, FALLBACK);
    return ctx.candidateList.slice().sort((a, b) => s[b] - s[a] || ctx.E.items.id[a] - ctx.E.items.id[b]);
  }

  /** Evaluation cohort for `valid`: the competition's test users that have a weeks 21-30 top-5. */
  function evaluationCohort(ctx) {
    if (ctx.mode !== "valid") return ctx.E.testUsers.slice();
    return ctx.E.testUsers.filter((u) => ctx.E.gtValid[u]);
  }

  /** All users with a weeks 21-30 top-5 (wider cohort, used for robustness checks). */
  function allValidUsers(ctx) {
    return Object.keys(ctx.E.gtValid).map(Number).sort((a, b) => a - b);
  }

  /**
   * Turn a score vector into exactly K item indices.
   * @returns {{items:number[], fromMethod:number}} fromMethod = slots filled by the method itself
   */
  function topK(ctx, scores, k = K) {
    const ranked = [];
    if (scores) {
      for (const i of ctx.candidateList) {
        const s = scores[i];
        if (s !== NO_SCORE && Number.isFinite(s)) ranked.push(i);
      }
      ranked.sort((a, b) => scores[b] - scores[a] || ctx.fallbackRank[a] - ctx.fallbackRank[b] || ctx.E.items.id[a] - ctx.E.items.id[b]);
    }
    const out = ranked.slice(0, k);
    const fromMethod = out.length;
    const used = new Set(out);
    for (const it of ctx.fallback) {
      if (out.length >= k) break;
      if (!used.has(it)) { out.push(it); used.add(it); }
    }
    return { items: out, fromMethod };
  }

  /**
   * Evaluate a recommender on ctx.cohort (valid mode).
   * @param {(u:number)=>Float64Array|null} scoreUser
   */
  function evaluate(ctx, scoreUser, users = ctx.cohort) {
    let sum = 0, padded = 0, usersPadded = 0;
    const perUser = [];
    for (const u of users) {
      const { items, fromMethod } = topK(ctx, scoreUser(u));
      const ap = apk(ctx.E.gtValid[u], items);
      sum += ap;
      if (fromMethod < K) { padded += K - fromMethod; usersPadded += 1; }
      perUser.push({ u, items, ap, fromMethod });
    }
    return { map: users.length ? sum / users.length : 0, users: users.length, paddedSlots: padded, usersPadded, perUser };
  }

  /** Per-user training history as [{item, n, p, m, last}] sorted by full views desc. */
  function history(ctx, u) {
    const out = [];
    for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) {
      out.push({ item: ctx.aItem[k], n: ctx.aN[k], p: ctx.aP[k], m: ctx.aM[k], last: ctx.aLast[k] });
    }
    return out.sort((a, b) => b.n - a.n || b.m - a.m || a.item - b.item);
  }

  /** Two fixed example users (same for every method): a heavy and a light viewer of the cohort. */
  function exampleUsers(ctx) {
    const cohort = ctx.mode === "valid" ? ctx.cohort : ctx.E.testUsers.filter((u) => ctx.E.gtValid[u]);
    const views = cohort.map((u) => {
      let s = 0;
      for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) s += ctx.aN[k];
      return { u, s };
    }).sort((a, b) => a.s - b.s || a.u - b.u);
    const at = (q) => views[Math.min(views.length - 1, Math.floor(q * (views.length - 1)))];
    return { heavy: at(0.9), light: at(0.1) };
  }

  function itemLabel(E, i) {
    return { id: E.items.id[i], title: E.items.title[i] || "", category: E.items.category[i] || "" };
  }

  /** RFC 4180 CSV. */
  function csvField(v) {
    if (v === null || v === undefined) return "";
    const s = String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }
  function toCsv(header, rows) {
    return [header, ...rows].map((r) => r.map(csvField).join(",")).join("\n") + "\n";
  }

  /** Kaggle submission text for test mode: user_id, five space-separated tv_show_ids. */
  function submissionCsv(ctx, scoreUser) {
    const rows = ctx.E.testUsers.map((u) => {
      const { items } = topK(ctx, scoreUser(u));
      return [ctx.E.users[u], items.map((i) => ctx.E.items.id[i]).join(" ")];
    });
    return toCsv(["user_id", "tv_show_id"], rows);
  }

  function nowMs() {
    return typeof performance !== "undefined" ? performance.now() : Date.now();
  }

  root.ExamCore = {
    K, NO_SCORE, FALLBACK, popularityVector, mulberry32, gaussian, apk, buildContext, sharedPopularity, evaluationCohort, allValidUsers,
    topK, evaluate, history, exampleUsers, itemLabel, toCsv, submissionCsv, nowMs,
  };
})(typeof window !== "undefined" ? window : globalThis);

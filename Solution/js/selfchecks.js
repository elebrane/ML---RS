/**
 * selfchecks.js — hand-checkable fixtures shared by the page ("Run tests") and Node (tests/).
 * Every expected number below can be recomputed with pencil and paper.
 */
(function (root) {
  "use strict";
  const C = root.ExamCore;

  /** Minimal context for rule mining: one row per (user, week, item). */
  function tinyCtx(rows, nItems, train) {
    const nUsers = Math.max(...rows.map((r) => r[0])) + 1;
    rows.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    const rowPtr = new Int32Array(nUsers + 1);
    for (const r of rows) rowPtr[r[0] + 1]++;
    for (let u = 0; u < nUsers; u++) rowPtr[u + 1] += rowPtr[u];
    return {
      nUsers, nItems, train, rowPtr,
      rowWeek: Int16Array.from(rows.map((r) => r[1])), rowItem: Int32Array.from(rows.map((r) => r[2])),
      rowN: Int32Array.from(rows.map(() => 1)), isCandidate: new Uint8Array(nItems).fill(1),
    };
  }

  function run(opts = {}) {
    const results = [];
    const check = (name, fn) => {
      try { const detail = fn(); results.push({ name, ok: true, detail: detail || "" }); }
      catch (e) { results.push({ name, ok: false, detail: e.message }); }
    };
    const near = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;
    const assert = (cond, msg) => { if (!cond) throw new Error(msg); };

    // ---- MAP@5 (course slide) ----
    check("AP@5 slide example: X = (1+1+0.75)/3, Y = (0.5+0.5+0.6)/3", () => {
      const x = C.apk(["A", "B", "C"], ["A", "B", "D", "C", "E"]);
      const y = C.apk(["A", "B", "C"], ["D", "A", "E", "B", "C"]);
      assert(near(x, 2.75 / 3) && near(y, 1.6 / 3), `got ${x}, ${y}`);
      return `X=${x.toFixed(4)} Y=${y.toFixed(4)}`;
    });
    check("AP@5 denominator is min(5, |R|): 1 hit at rank 1 of 5 relevant = 0.2", () => {
      const v = C.apk([1, 2, 3, 4, 5], [1, 9, 9, 9, 9]);
      assert(near(v, 0.2), `got ${v}`);
    });

    // ---- Association rules (lecture-style baskets) ----
    // 8 baskets of week 11 (one user each): AB, AB, AB, A, B, B, ACD, CD   with A=0 B=1 C=2 D=3
    const baskets = [[0, 1], [0, 1], [0, 1], [0], [1], [1], [0, 2, 3], [2, 3]];
    const rows = [];
    baskets.forEach((b, u) => b.forEach((i) => rows.push([u, 11, i])));
    const ctxR = tinyCtx(rows, 4, [11, 11]);
    check("Rules: C->D support 2/8, confidence 2/2, lift 1/(2/8) = 4 is kept", () => {
      const m = root.ExamRules.mineRules(ctxR, "same-week", 0.2, 0.5);
      const r = (m.rules.get(2) || []).find((x) => x.b === 3);
      assert(r && near(r.support, 0.25) && near(r.confidence, 1) && near(r.lift, 4), JSON.stringify(r));
      return `support=${r.support} confidence=${r.confidence} lift=${r.lift}`;
    });
    check("Rules: A->B confidence 3/5 = 0.6 but lift 0.6/(5/8) = 0.96 < 1 is rejected", () => {
      const m = root.ExamRules.mineRules(ctxR, "same-week", 0.2, 0.5);
      const r = (m.rules.get(0) || []).find((x) => x.b === 1);
      assert(!r, "A->B should be filtered by lift <= 1");
      return `transactions=${m.stats.transactions}, frequentPairs=${m.stats.frequentPairs}, rules=${m.stats.rules}`;
    });
    check("Rules: next-week mode counts A(t)->A(t+1) as a rule", () => {
      // user 0: wk11 {A}, wk12 {A,B}; user 1: wk11 {A}, wk12 {B}; user 2: wk11 {B}, wk12 {} -> 3 transactions (wk11->wk12)
      const c = tinyCtx([[0, 11, 0], [0, 12, 0], [0, 12, 1], [1, 11, 0], [1, 12, 1], [2, 11, 1], [2, 12, 2]], 3, [11, 12]);
      const m = root.ExamRules.mineRules(c, "next-week", 0.01, 0.1);
      // N = transactions with a non-empty week-11 basket = 3 (wk12 baskets only act as consequents here)
      const aa = (m.rules.get(0) || []).find((x) => x.b === 0);
      // count(A_t)=2, count(A_t & A_t+1)=1 -> conf 0.5; count(A_t+1)=1 of N=3 -> lift 0.5/(1/3)=1.5
      assert(m.stats.transactions === 3 && aa && near(aa.confidence, 0.5) && near(aa.lift, 1.5), JSON.stringify({ s: m.stats, aa }));
      return `A->A confidence=${aa.confidence} lift=${aa.lift}`;
    });

    // ---- PPR: week-6 reference graph ----
    // A->B, A->C, B->C, C->A, D->C  (nodes A,B,C,D = 0..3)
    const G = { ptr: Int32Array.from([0, 2, 3, 4, 5]), idx: Int32Array.from([1, 2, 2, 0, 2]), prob: Float64Array.from([0.5, 0.5, 1, 1, 1]) };
    check("PPR reference graph, uniform restart, alpha=0.85 (week-6 values)", () => {
      const r = root.ExamPPR.powerIteration(G, 4, 0.85, Float64Array.from([0.25, 0.25, 0.25, 0.25]), 1e-12, 2000);
      const exp = [0.3725268513, 0.1958239118, 0.3941492369, 0.0375];
      exp.forEach((e, j) => assert(near(r.p[j], e, 1e-9), `node ${j}: ${r.p[j]} vs ${e}`));
      return `p=[${Array.from(r.p).map((v) => v.toFixed(10)).join(", ")}] iterations=${r.iterations}`;
    });
    check("PPR reference graph, restart at A, alpha=0.85 (week-6 values)", () => {
      const r = root.ExamPPR.powerIteration(G, 4, 0.85, Float64Array.from([1, 0, 0, 0]), 1e-12, 2000);
      const exp = [0.4522328999, 0.1921989825, 0.3555681176, 0];
      exp.forEach((e, j) => assert(near(r.p[j], e, 1e-9), `node ${j}: ${r.p[j]} vs ${e}`));
      return `p=[${Array.from(r.p).map((v) => v.toFixed(10)).join(", ")}]`;
    });
    check("PPR conserves probability mass with a dangling node", () => {
      // 0->1, 1->2, node 2 has no out-edge
      const D = { ptr: Int32Array.from([0, 1, 2, 2]), idx: Int32Array.from([1, 2]), prob: Float64Array.from([1, 1]) };
      const r = root.ExamPPR.powerIteration(D, 3, 0.85, Float64Array.from([1, 0, 0]), 1e-12, 5000);
      const mass = r.p.reduce((s, v) => s + v, 0);
      assert(near(mass, 1, 1e-9) && r.converged, `mass=${mass}`);
      return `mass=${mass.toFixed(12)} converged in ${r.iterations}`;
    });

    // ---- MF: Cholesky and ALS loss monotonicity ----
    check("MF: Cholesky solves [[4,2],[2,3]] x = [2,5] -> x = [-0.5, 2]", () => {
      const x = root.ExamMF.choleskySolve(Float64Array.from([4, 2, 2, 3]), Float64Array.from([2, 5]), 2);
      assert(near(x[0], -0.5) && near(x[1], 2), `x=${Array.from(x)}`);
    });

    // ---- Content-based cosine ----
    check("CB: cosine of two unit vectors sharing 1 of 2 features each = 0.5", () => {
      const a = new Map([["x", 1 / Math.SQRT2], ["y", 1 / Math.SQRT2]]), b = new Map([["y", 1 / Math.SQRT2], ["z", 1 / Math.SQRT2]]);
      let dot = 0; for (const [k, v] of a) if (b.has(k)) dot += v * b.get(k);
      assert(near(dot, 0.5), `dot=${dot}`);
    });

    // ---- Real-data checks (need window.EXAM) ----
    if (opts.withRealData && root.EXAM) {
      const ctx = C.buildContext(root.EXAM, "valid");
      check("Data: 4,030 users have a weeks 21-30 top-5; 1,150 of them are competition test users", () => {
        const n = Object.keys(root.EXAM.gtValid).length;
        assert(n === 4030 && ctx.cohort.length === 1150, `gt=${n} cohort=${ctx.cohort.length}`);
      });
      check("Data: every ground-truth programme airs in weeks 21-30 (is a candidate)", () => {
        let bad = 0; for (const u of ctx.cohort) for (const i of root.EXAM.gtValid[u]) if (!ctx.isCandidate[i]) bad++;
        assert(bad === 0, `${bad} ground-truth items outside the pool`);
      });
      check("Data: training context never contains weeks after 20", () => {
        let maxW = 0; for (const w of ctx.rowWeek) maxW = Math.max(maxW, w);
        assert(maxW <= 20, `max week ${maxW}`);
        return `max training week = ${maxW}`;
      });
      check("Core: topK always returns 5 distinct candidates (fallback padding)", () => {
        const s = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
        s[ctx.candidateList[0]] = 1; s[ctx.candidateList[1]] = 2;
        const { items, fromMethod } = C.topK(ctx, s);
        assert(items.length === 5 && new Set(items).size === 5 && fromMethod === 2 && items[0] === ctx.candidateList[1], JSON.stringify({ items, fromMethod }));
        assert(items.every((i) => ctx.isCandidate[i]), "non-candidate returned");
      });
      check("MF: ALS loss never increases (3 iterations, k=8)", () => {
        const m = root.ExamMF.trainALS(ctx, { factors: 8, lambda: 1, alpha: 5, iterations: 3, signal: "count", confidence: "log", seed: 1, trackLoss: true });
        const seq = m.history.flatMap((h) => [h.lossAfterUserStep, h.lossAfterItemStep]);
        for (let j = 1; j < seq.length; j++) assert(seq[j] <= seq[j - 1] * (1 + 1e-9), `loss rose at step ${j}: ${seq}`);
        assert(m.factorNorm > 1e-3, "factors collapsed to zero");
        return `loss: ${seq.map((v) => v.toFixed(0)).join(" -> ")}`;
      });
      check("PPR: probability mass sums to 1 for one user (isolated nodes are never reached)", () => {
        const p = root.ExamPPR.makePPR(ctx, { alpha: 0.5, weight: "binary" });
        const u = ctx.cohort[0];
        const r = p.run(u);
        const mass = r.p.reduce((s, v) => s + v, 0);
        // users/items without edges are isolated nodes: they are never reached, so they hold no mass
        assert(near(mass, 1, 1e-9) && r.converged, `mass=${mass} converged=${r.converged}`);
        return `nodes=${p.graph.n}, edges=${p.graph.edges}, isolated (zero-degree) nodes=${p.graph.dangling}, iterations=${r.iterations}`;
      });
    }
    const passed = results.filter((r) => r.ok).length;
    return { results, passed, failed: results.length - passed };
  }

  /**
   * TensorFlow.js checks (async; need root.tf and window.EXAM). Compare the TF.js methods with the
   * float64 plain-JS oracles and check memory hygiene.
   */
  async function runTF() {
    const results = [];
    const tf = root.tf;
    const check = async (name, fn) => {
      try { const detail = await fn(); results.push({ name, ok: true, detail: detail || "" }); }
      catch (e) { results.push({ name, ok: false, detail: e.message }); }
    };
    if (!tf || !root.EXAM) return { results: [{ name: "TF.js checks", ok: false, detail: "tf or EXAM missing" }], passed: 0, failed: 1 };
    const ctx = C.buildContext(root.EXAM, "valid");
    const users = ctx.cohort.slice(0, 20);
    await check(`PPR TF.js (${tf.getBackend()}) equals the float64 power iteration for 20 users (alpha 0.7, count)`, async () => {
      const before = tf.memory().numTensors;
      const a = await root.ExamPPRTF.makePPRTF(ctx, { alpha: 0.7, weight: "count" });
      await a.prepare(users);
      const b = root.ExamPPR.makePPR(ctx, { alpha: 0.7, weight: "count" });
      let maxDiff = 0, same = 0;
      for (const u of users) {
        const x = a.scoreUser(u), y = b.scoreUser(u);
        for (const i of ctx.candidateList) {
          const xv = Number.isFinite(x[i]) ? x[i] : 0, yv = Number.isFinite(y[i]) ? y[i] : 0;
          maxDiff = Math.max(maxDiff, Math.abs(xv - yv));
        }
        if (C.topK(ctx, x).items.join() === C.topK(ctx, y).items.join()) same++;
      }
      a.dispose();
      const leaked = tf.memory().numTensors - before;
      if (maxDiff > 1e-5 || leaked !== 0) throw new Error(`max |diff| ${maxDiff}, same Top-5 ${same}/20, leaked tensors ${leaked}`);
      return `max |diff| = ${maxDiff.toExponential(2)} (float32 vs float64), identical Top-5 for ${same}/20 users, leaked tensors 0, not converged ${a.conv.notConverged}`;
    });
    await check("MF TF.js: Adam lowers the full-matrix implicit loss and leaves no tensors behind", async () => {
      const before = tf.memory().numTensors;
      const m = await root.ExamMFTF.trainMFTF(ctx, { factors: 4, lambda: 50, alpha: 1, signal: "count", epochs: 25, batchSize: 8192, lr: 0.05, seed: 1 });
      const h = m.history;
      const leaked = tf.memory().numTensors - before;
      if (!(h[h.length - 1].loss < h[0].loss) || leaked !== 0 || !(m.factorNorm > 1e-3)) throw new Error(`loss ${h.map((x) => x.loss.toFixed(0))}, leaked ${leaked}, |V| ${m.factorNorm}`);
      return `loss epoch 1 = ${h[0].loss.toFixed(0)} -> epoch 25 = ${h[h.length - 1].loss.toFixed(0)}, |V| = ${m.factorNorm.toFixed(3)}, leaked tensors 0`;
    });
    const passed = results.filter((r) => r.ok).length;
    return { results, passed, failed: results.length - passed };
  }

  root.ExamSelfChecks = { run, runTF };
})(typeof window !== "undefined" ? window : globalThis);

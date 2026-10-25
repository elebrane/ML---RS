// Proxy experiment: how MAP@5 of the "own history" ranking depends on the training history length,
// the target-window length and the logging gap of weeks 21-27. Ground truth = organiser's rule on the given weeks.
const path = require("path");
// Usage: node tools/history_length_check.js > results/analysis/history_length_check.txt
require(path.join(__dirname, "..", "data", "exam_data.js"));
const E = globalThis.EXAM, L = E.log, nU = E.users.length;
const test = new Set(E.testUsers);
function agg(w0, w1) {
  const m = Array.from({ length: nU }, () => new Map());
  for (let r = 0; r < L.u.length; r++) {
    const w = L.w[r]; if (w < w0 || w > w1 || L.n[r] === 0) continue;
    const o = m[L.u[r]].get(L.i[r]) || [0, 0]; o[0] += L.n[r]; o[1] += L.m[r]; m[L.u[r]].set(L.i[r], o);
  }
  return m;
}
const rank = (mp) => [...mp.entries()].sort((a, b) => b[1][0] - a[1][0] || b[1][1] - a[1][1] || E.items.id[a[0]] - E.items.id[b[0]]).map((e) => e[0]);
function apk(actual, pred) { let h = 0, s = 0; pred.slice(0, 5).forEach((p, k) => { if (actual.includes(p)) { h++; s += h / (k + 1); } }); return s / Math.min(5, actual.length); }
function run(tr, ev, label) {
  const A = agg(...tr), G = agg(...ev);
  let sum = 0, n = 0, sumAll = 0, nAll = 0, views = 0;
  for (let u = 0; u < nU; u++) {
    if (G[u].size < 5) continue;
    const gt = rank(G[u]).slice(0, 5), pred = rank(A[u]).slice(0, 5);
    const ap = apk(gt, pred); sumAll += ap; nAll++;
    if (test.has(u)) { sum += ap; n++; }
    for (const v of G[u].values()) views += v[0];
  }
  console.log(`${label.padEnd(44)} test users ${String(n).padStart(4)}  MAP@5 ${(sum / n).toFixed(4)} | all users ${String(nAll).padStart(4)}  MAP@5 ${(sumAll / nAll).toFixed(4)} | full views per GT user ${(views / nAll).toFixed(1)}`);
}
console.log("Own-history ranking (proxy for PPR, validation 0.3552 vs PPR 0.3595); no candidate filter.");
run([11, 20], [21, 30], "train 11-20 -> target 21-30 (our validation)");
run([16, 20], [21, 30], "train 16-20 (5 wk)  -> target 21-30");
run([19, 20], [21, 30], "train 19-20 (2 wk)  -> target 21-30");
run([11, 20], [21, 25], "train 11-20 -> target 21-25 (5 wk)");
run([11, 20], [21, 23], "train 11-20 -> target 21-23 (3 wk, thin log)");
run([11, 20], [28, 30], "train 11-20 -> target 28-30 (3 wk, normal log)");
run([11, 27], [28, 30], "train 11-27 -> target 28-30 (3 wk, normal log)");
run([11, 17], [18, 20], "train 11-17 -> target 18-20 (3 wk, normal log)");

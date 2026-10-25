/**
 * Two small PPR subgraphs for the report (one per example user, 10 nodes each):
 * the seed user, the 4 other users with the highest PPR probability from that seed (where the walk goes),
 * and the seed's PPR Top-5 programmes. Edges = full views in the training weeks 11-20 between these nodes.
 * Selected setting: alpha 0.7 (link-following), edge weight = full-view count, float64 power iteration (js/ppr.js).
 * Usage: node tools/ppr_two_user_graph.js > results/analysis/ppr_two_user_graph.json
 */
const { loadEnv } = require("./load_env.js");
const g = loadEnv(["ppr.js"]);
const C = g.ExamCore, E = g.EXAM;

const ctx = C.buildContext(E, "valid");
const opts = { alpha: 0.7, weight: "count", tol: 1e-6, maxIter: 500 };
const model = g.ExamPPR.makePPR(ctx, opts);
const ex = C.exampleUsers(ctx);
const nU = ctx.nUsers;

function edgesOf(u) {
  const m = new Map();
  for (let k = ctx.aggPtr[u]; k < ctx.aggPtr[u + 1]; k++) if (ctx.aN[k] > 0) m.set(ctx.aItem[k], ctx.aN[k]);
  return m;
}

const panels = [];
for (const [role, { u }] of Object.entries(ex)) {
  const r = model.run(u);
  const scores = new Float64Array(ctx.nItems).fill(C.NO_SCORE);
  for (const i of ctx.candidateList) { const v = r.p[nU + i]; if (v > 0) scores[i] = v; }
  const top = C.topK(ctx, scores).items;
  const others = [];
  for (let v = 0; v < nU; v++) if (v !== u && r.p[v] > 0) others.push(v);
  others.sort((a, b) => r.p[b] - r.p[a]);
  const users = [u, ...others.slice(0, 4)];
  const actual = new Set(E.gtValid[u] || []);
  const seedEdges = edgesOf(u);
  const edges = [];
  users.forEach((v) => { const m = edgesOf(v); top.forEach((i) => { if (m.has(i)) edges.push({ user: E.users[v], item: E.items.id[i], w: m.get(i) }); }); });
  panels.push({
    role, seed: E.users[u], iterations: r.iterations, residual: r.residual,
    users: users.map((v, k) => ({ id: E.users[v], seed: k === 0, ppr: r.p[v], degree: edgesOf(v).size,
      fullViews: [...edgesOf(v).values()].reduce((s, x) => s + x, 0) })),
    items: top.map((i, k) => ({ id: E.items.id[i], rank: k + 1, ...C.itemLabel(E, i), ppr: scores[i],
      hit: actual.has(i), seedViews: seedEdges.get(i) || 0 })),
    edges,
  });
}
process.stdout.write(JSON.stringify({ settings: opts, weeks: "11-20", panels }, null, 1));

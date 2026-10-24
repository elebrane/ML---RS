/**
 * Material for a hand check of one user's AP@5: the user's full-view counts per programme in weeks 21-30 (the ground truth
 * is the top 5 of these by count, ties by minutes then id) and in weeks 11-20 (what the methods were fitted on).
 * Usage: node tools/handcheck_user.js <user_id> > results/analysis/handcheck_<user_id>.csv
 */
const { loadEnv } = require("./load_env.js");
const g = loadEnv([]);
const C = g.ExamCore, E = g.EXAM;
const uid = process.argv[2];
const u = E.users.indexOf(uid);
if (u < 0) throw new Error("unknown user " + uid);
const L = E.log, acc = new Map();
for (let r = 0; r < L.u.length; r++) {
  if (L.u[r] !== u || L.n[r] === 0) continue;
  const win = L.w[r] <= 20 ? "w11_20" : "w21_30";
  const k = L.i[r];
  if (!acc.has(k)) acc.set(k, { w11_20: [0, 0], w21_30: [0, 0] });
  acc.get(k)[win][0] += L.n[r]; acc.get(k)[win][1] += L.m[r];
}
const rows = [...acc.entries()].map(([i, o]) => ({ i, ...o }))
  .sort((a, b) => b.w21_30[0] - a.w21_30[0] || b.w21_30[1] - a.w21_30[1] || E.items.id[a.i] - E.items.id[b.i]);
const gt = E.gtValid[u] || [];
const out = rows.map((r, k) => { const lab = C.itemLabel(E, r.i); return [uid, E.items.id[r.i], lab.title, lab.category, r.w21_30[0], r.w21_30[1], gt.indexOf(r.i) >= 0 ? gt.indexOf(r.i) + 1 : "", r.w11_20[0], r.w11_20[1]]; });
process.stdout.write(C.toCsv(["user_id", "tv_show_id", "title", "category", "full_views_w21_30", "minutes_w21_30", "ground_truth_rank", "full_views_w11_20", "minutes_w11_20"], out));

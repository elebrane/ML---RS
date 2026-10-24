# Notes for the report (to include when the PDF is assembled)

- **PPR edge weights — a new input beyond the practical classes.** State explicitly that the PPR graph uses weighted
  edges (`binary`, `log(1+n)`, `count` = number of full views; `count` selected on validation), which were not part of
  the practical homework: week 6 built a binary graph (`R[u,i] = 1` if rating >= 4.0, see `week6/readme.md` §3.1), while
  weighted edges were only mentioned in the lectures. Evidence for the choice: `results/validation/ppr_sweep.csv`
  (e.g. alpha 0.7: binary 0.269, log 0.352, count 0.360 cohort MAP@5).
  Lecture reference (checked visually, the slides are images): lecture 06 "Graph-based RecSys", slide 3 "Graphs and Bipartite
  Graphs" — "Weight: strength of a connection" (edge drawn with weight 3); slide 5 — transition P_ij = A_ij / sum_k A_ik.

- **Figure candidate:** PPR tab → “Graph view for two users” (heavy user A on top, light user B at the bottom,
  colour by PPR score, edge width = edge weight). Shows that A's PPR Top-5 are his heaviest edges, i.e. the walk is
  dominated by habitual programmes when edges are weighted by full-view counts. Take the screenshot from the page.

- **Example users (§7):** say how the final pair was chosen — default P90 / P10 of training full views, or a draw from the
  heavy (top 25 %, >= 115 full views) / light (bottom 25 %, <= 34) lists on the page — and use the same pair for every method.
  If the pair is not the default, export `examples_all_methods.csv` from the Summary tab (saved to `results/browser/`) and cite that file.
- **Where submission_ppr.csv comes from:** built by `tools/run_experiments.js` (Node.js running the same `js/core.js` +
  `js/ppr_tf.js`, TF.js 4.22.0 WASM) on 2026-10-22 22:36 MSK. Verification: the page's own button (PPR tab → "Build Top-5 for
  the 1,260 test users" → Export, headless Chromium, WASM backend, 179 s) produced a byte-identical `submission_ppr.csv`
  (1,260/1,260 rows); the float64 plain-JS reference implementation also produced identical rows.

- **MF training controls aligned with HW5 (epochs, learning rate, L2, batch size).** Batch = a set of users (whole matrix rows);
  L2 on the batch's user rows only + the batch's share of ||V||^2 (HW5 lesson), so an epoch's batch losses add up to the full loss.
  batchSize >= #users is the full-matrix training of the earlier run — verified bit-identical (k = 16 setting: 0.312548 both ways).
  Stage 4 of the MF sweep (k 64, lambda 50, alpha 1, count), cohort MAP@5: full batch x 100 epochs 0.31597; 1,024 x 20 epochs (100
  updates) 0.29185; 256 x 20 epochs (380 updates) 0.30124; 1,024 x 100 epochs (500 updates) 0.31648 -> selected. Honest caveat: the
  gain over the full batch (+0.0005) is within noise, and on all 4,030 users the full batch is slightly better (0.29324 vs 0.29050).

- **Kaggle scores (screenshot `results/kaggle/kaggle_submissions_2026-10-25.png`, also in `Kaggle output/`), late submissions, 2026-10-23:**
  `submission_ppr.csv` public 0.41930 / private 0.40867; `submission_mf.csv` public 0.37819 / private 0.36688. Table: `results/kaggle/kaggle_scores.csv`.
  The validation ranking holds on Kaggle (PPR > MF by 0.044 on validation, 0.041 public). Kaggle scores are higher than validation MAP@5
  because the hidden test labels are another period/user set — compare methods, not absolute levels.
- **Which MF file was uploaded:** the browser export of 2026-10-23 03:56 MSK (`results/browser/submission_mf.csv` = `Kaggle output/submission_mf.csv`),
  made before the HW5 mini-batch controls existed, i.e. the **full-batch** setting k 64, λ 50, α 1, count, lr 0.05, 100 epochs (validation 0.315965,
  sweep config 9), WebGL backend. Check: `node tools/refit_submission.js mf '<that setting, batchSize 8192>' results/kaggle/repro_submission_mf_fullbatch.csv`
  (Node, WASM) gives 1,213/1,260 identical rows, 1,242 identical Top-5 sets, 99.7 % item overlap — the rest is float32 rounding differences
  between WebGL and WASM over 100 Adam steps. The currently selected MF (mini-batch 1,024, `results/submissions/submission_mf.csv`) was NOT uploaded.

- **References checked on 2026-10-25** (pages opened): Agrawal & Srikant VLDB'94 PDF (vldb.org), Jeh & Widom WWW'03 (pp. 271-279,
  DOI 10.1145/775152.775191, webis anthology record), Kingma & Ba arXiv 1412.6980 (ICLR 2015), Pazzani & Billsus chapter PDF,
  Hu-Koren-Volinsky ICDM'08 pp. 263-272 (citation in arXiv 1901.04704), N. Melentev TDS article 2021-03-20 (2nd place of this
  competition, private 0.42905), TF.js release tfjs-v4.22.0 (GitHub, 2024-10-21).
- **List analysis for §8:** `node tools/analyze_lists.js` → `results/analysis/list_profile.csv`, `per_user_ap_valid.csv` (all six MAP@5
  values reproduced exactly from a fresh refit). Hand-check material: `results/analysis/handcheck_15942687135322962396.csv`.

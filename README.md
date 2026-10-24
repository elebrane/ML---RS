# sweet.tv — TV programme recommender (LLM4Rec final exam, AVITO cohort)

Individual exam solution by Vladislav Kutranov. Kaggle community competition “sweet.tv — TV program recommender”, metric MAP@5.

- **Application (interface with the solution):** `Solution/index.html` — open it on GitHub Pages (this repository's Pages site opens it
  directly) or locally with `python3 Solution/server.py`. Tabs: data & protocol, popularity baseline, content-based, association rules,
  matrix factorization (TensorFlow.js), Personalized PageRank (TensorFlow.js), summary. Every method tab ends with a step-by-step
  rebuild of its Kaggle submission file. On a static host the page's exports are offered as downloads.
- **Final submission file:** [`submission_ppr.csv`](submission_ppr.csv) — Personalized PageRank (α 0.7, edge weight = number of full views),
  Kaggle public 0.41930 / private 0.40867 (late submission). Same file: `Solution/results/submissions/submission_ppr.csv`.
- **Code, protocol, results, reproduction:** [`Solution/readme.md`](Solution/readme.md).
- **All numbers:** `Solution/results/summary.csv` (validation), `Solution/results/kaggle/kaggle_scores.csv` (Kaggle),
  `Solution/results/analysis/` (list profile, per-user AP@5).
- **Data:** the raw competition files are not included — put them into `Kaggle input/sweettv-tv-program-recommender/` to rerun the
  pipeline (`Solution/readme.md` §2). `Solution/data/exam_data.js` is the packed derivative the page loads.

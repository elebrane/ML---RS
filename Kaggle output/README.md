# Files uploaded to Kaggle (output)

Exactly the files that were submitted (late submissions, 2026-10-23), and the returned scores
(`kaggle_submissions_2026-10-25.png`, unmodified screenshot of the competition's Submissions page).

| File | Method | Public | Private | MD5 |
|---|---|---:|---:|---|
| `submission_ppr.csv` — **final, designated submission** | Personalized PageRank, α 0.7, edge weight = number of full views | 0.41930 | 0.40867 | `7cf9fc16cd684f66e213817373f9afa8` |
| `submission_mf.csv` | Implicit MF (TF.js), k 64, λ 50, α 1, count, 100 epochs, full batch, built in the browser (WebGL) | 0.37819 | 0.36688 | `0fd81f793ad710ff25d01e1a5ba59eb6` |

`submission_ppr.csv` is byte-identical to `Solution/results/submissions/submission_ppr.csv` and to `submission_ppr.csv` in the
repository root. `submission_mf.csv` was exported from the browser before the HW5 mini-batch controls were added, so it differs
from `Solution/results/submissions/submission_mf.csv` (mini-batch setting); see `Solution/results/kaggle/kaggle_scores.csv` and
`Solution/readme.md` §4.

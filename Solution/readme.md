# sweet.tv — TV Programme Recommender (LLM4Rec final exam, AVITO cohort)

**Course**: LLM4Rec, HSE University · **Instructor**: Seungmin Jin · **Task**: individual exam quiz
**Competition**: Kaggle community competition “sweet.tv — TV program recommender” (slug `sweettv-tv-program-recommender`), metric MAP@5.

The application follows the format of the earlier homeworks: one `index.html` with a visual, tabbed interface
(one tab per method, a final Summary tab), plain JavaScript in `js/`, TensorFlow.js 4.22.0 for the two
learning methods (as in weeks 5 and 6), a packed dataset in `data/exam_data.js`, and a small `server.py` that
saves the page's CSV exports. Heavy data preparation (the 5.8 M-row join) is done once in Python.

## 1. Folder structure

| Path | Role |
|---|---|
| `index.html`, `style.css`, `js/app.js` | the application: tabs 1 Data & protocol · 2 Popularity · 3 Content-based · 4 Association rules · 5 MF · 6 PPR · 7 Summary |
| `js/core.js` | shared protocol: contexts, candidate pool, fallback, Top-5, MAP@5, CSV |
| `js/popularity.js` | required popularity baseline (+ “own history” reference) |
| `js/content.js` | method 1 — content-based filtering (week 2: item vectors, cosine, profile) |
| `js/rules.js` | method 2 — association rules (week 4: Apriori, support / confidence / lift) |
| `js/mf_tf.js` | method 3 — implicit matrix factorization in TensorFlow.js (week 5: embeddings, Adam; HW5 controls epochs, learning rate, L2, batch size) |
| `js/ppr_tf.js` | method 4 — Personalized PageRank in TensorFlow.js (week 6: two-block propagation) |
| `js/graphview.js`, `vendor/d3.min.js` | PPR tab, “Graph view for two users”: week-6 display (d3 7.9.0, two-column bipartite layout, colour by kind/degree or score, tooltip, node cap) for two chosen users, edge width = edge weight |
| `js/mf.js`, `js/ppr.js` | independent float64 plain-JS reference implementations (ALS; sparse power iteration) used as oracles |
| `js/methods.js` | one registry: every method's settings grid (shared by the page and the batch runner) |
| `js/selfchecks.js` | hand-checkable fixtures + data checks + TF.js-vs-oracle checks (“Run tests” button) |
| `tools/*.py` | data preparation and exploration (Python) |
| `tools/run_experiments.js` | batch runner (Node + TF.js WASM): validation sweeps, selection, examples, test refit, submissions |
| `tools/analyze_lists.js` | list profile of every selected method on the validation cohort (familiar vs new slots, hits, distinct programmes) and per-user AP@5 → `results/analysis/` |
| `tools/handcheck_user.js` | one user's full-view counts in weeks 11–20 and 21–30, for checking a ground-truth top-5 and an AP@5 by hand |
| `tools/ppr_two_user_graph.js`, `tools/draw_ppr_two_user_graph.py` | report Fig. 3: 10-node PPR subgraph per example user (seed, 4 users with the highest PPR probability, PPR Top-5), mirrored → `results/analysis/ppr_two_user_graph.{json,svg,png}` |
| `tools/refit_submission.js` | refit one method with explicit settings on weeks 11–30 and write its submission (used to check which settings produced an uploaded file) |
| `results/` | everything the code produced (CSV/JSON); `results/browser/` receives exports made from the page |
| `tests/` | `test_lib.py` (join + MAP@5), `test_methods.js` (18 checks), `test_checkpoint.js` (MF resume) |
| `notes/engineering_log.md` | defects found while building the solution, with root cause, fix and confirmation |

## 2. How to reproduce

Inputs: put the five competition files in `exam_quiz/Kaggle input/sweettv-tv-program-recommender/`
(`dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, `submission.csv`).

```bash
# from the exam_quiz folder — Python 3.10+, pandas, numpy
python3 Solution/tools/eda_views.py              # view-log profile, cache/views.npz            (~30 s)
python3 Solution/tools/build_interactions.py     # join + exploration tables in results/eda     (~40 s)
python3 Solution/tools/check_clock_alignment.py  # trap check: same clock in log and schedule
python3 Solution/tools/build_app_data.py         # data/exam_data.js for the page and the runner (~10 s)
python3 Solution/tests/test_lib.py               # join fixture + MAP@5 slide example

cd Solution && npm install                       # @tensorflow/tfjs 4.22.0 (+ WASM backend) for Node
node tests/test_methods.js --real                # 18 checks
node tests/test_checkpoint.js                    # MF checkpoint / resume check
node tools/run_experiments.js                    # all sweeps and exports (add --budget=SECONDS --shard=i/n to split)

python3 server.py                                # then open http://127.0.0.1:8000/  (add ?backend=wasm if WebGL is unavailable)
```

On the page, the last step of every method tab, **“Kaggle submission — step by step”**, rebuilds `submission_<method>.csv` in the browser
and shows each stage with its numbers (template users → training weeks 11–30 → candidates on air in weeks 31–42 → fit → Top-5 + fallback →
format checks → save to `results/browser/` and compare with `results/submissions/`). For PPR the browser-built file is byte-identical to the
batch-run file (checked in headless Chromium with the WASM backend).

The batch runner caches every evaluated setting in `results/validation/cache_<method>*.json` and resumes where it
stopped; `--shard=i/n` lets n processes split the pending settings (the numbers in `results/` were produced this way
with 3–4 parallel processes because a single command on the author's machine was limited to 3 minutes).

## 3. Shared protocol (identical for every method)

- **Join**: a view `(user_id, vsetv_id, start_time, stop_time)` is matched to every labelled airing of the same channel
  (`vsetv_id = channel_id`) whose interval overlaps it. Per (user, airing) all overlapping seconds are summed (capped at the
  airing duration) and divided by `duration` → screen time. **Watched = screen time ≥ 0.8**, the organiser's own label rule
  (Data Description). Schedule timestamps are day-first (`09.03.2020 06:00:00`) and are parsed with an explicit format.
- **`tv_show_id = 0`**: unlabelled airtime — dropped as items (never recommended or scored).
- **Validation**: fit on weeks 11–20, evaluate on weeks 21–30. **Final**: refit on weeks 11–30, predict weeks 31–42.
- **Ground truth (validation)**: organiser's rule on weeks 21–30 — per user, count watched airings per programme; users with ≥ 5
  programmes keep their 5 most frequent (ties: watched seconds, then id). 4,030 users.
- **Cohort**: the 1,150 competition test users that have such a top-5 (robustness column: all 4,030).
- **Candidates**: labelled programmes airing in the target window (6,376 in weeks 21–30; weeks 31–42 for the test). The test
  schedule is used only for what airs — never for who watched.
- **Repeats allowed** for every method (51 % of validation top-5 programmes were already watched by the same user).
- **Fallback** (no history / unfilled slots): popularity = distinct viewers in the last 4 training weeks (the baseline's selected setting).
- **MAP@5** as in the course slides: AP@5 = Σ precision at relevant ranks / min(5, |R_u|).
- **Example users** (exam §7, the same pair for every method): the batch runner uses the deterministic default — the cohort users at the
  90th (heavy) and 10th (light) percentile of weeks 11–20 full views. On the page (step 4 of any method tab) the pair can be changed:
  heavy = top 25 % of the cohort (≥ 115 full views), light = bottom 25 % (≤ 34, includes users with no history), chosen from the list or with
  “Random heavy / light / pair”; the choice is shared by all tabs and the page's exports record the pair.

## 4. Results (results/summary.csv)

| Method | Role | MAP@5, 1,150 cohort | MAP@5, all 4,030 | Selected setting |
|---|---|---|---|---|
| Popularity | required baseline | 0.265238 | 0.239822 | viewers, last 4 weeks |
| Content-based | method | 0.255261 | 0.228557 | meta 1, people 0.5, channel 1, time 1, linear profile |
| Association rules | method | 0.317197 | 0.285429 | next-week, min support 0.001, min confidence 0.1, max, lookback 4 |
| MF (TF.js) | method | 0.316484 | 0.290499 | k 64, λ 50, α 1, count, Adam lr 0.05, 100 epochs, batch 1,024 users (5 mini-batches per epoch), seed 42 |
| PPR (TF.js) | method | 0.359528 | 0.323687 | α 0.7, edge weight = full-view count, tol 1e-6 |
| MF by ALS (plain JS) | reference implementation | 0.316168 | 0.289042 | k 64, λ 50, α 1, hours, linear, 20 iterations |
| PPR (plain JS, float64) | reference implementation | 0.359528 | 0.323687 | same as PPR (TF.js) |
| Own viewing history | reference | 0.355249 | 0.317390 | — |

Selection used the same validation data, so the selected values are optimistic; the Kaggle public score is the independent check.

**Kaggle (late submissions, 2026-10-23; screenshot `results/kaggle/kaggle_submissions_2026-10-25.png`, table `results/kaggle/kaggle_scores.csv`):**

| File | Setting | Validation MAP@5 (cohort) | Public | Private |
|---|---|---|---|---|
| `submission_ppr.csv` (final) | PPR, α 0.7, count edge weights | 0.359528 | **0.41930** | 0.40867 |
| `submission_mf.csv` | MF, k 64, λ 50, α 1, count, lr 0.05, 100 epochs, **full batch** (built in the browser, WebGL) | 0.315965 | 0.37819 | 0.36688 |

The uploaded MF file was made before the HW5 mini-batch controls were added, so it is the full-batch setting (sweep config 9), not the
mini-batch setting in the table above; `tools/refit_submission.js` with that setting reproduces it (1,213/1,260 identical rows on the
WASM backend; the rest differ by float32 rounding between WebGL and WASM).
Every sweep is in `results/validation/<method>_sweep.csv` (all evaluated settings, selected one flagged).

## 5. Seeds, environment, runtime

- Seeds: MF (TF.js) `tf.randomNormal` seeds 42 (users) / 43 (programmes), user permutation Mulberry32(seed + 7), batch order Mulberry32(seed + 13);
  ALS Mulberry32 seed 42 (seeds 7 and 2026 also reported).
  Popularity, content-based, rules and PPR are deterministic. A refit of every selected setting reproduced its MAP@5 exactly
  (`refit_map_abs_diff` = 0 in `results/summary.csv`).
- Environment of the saved run (`results/environment.json`): Linux arm64 VM on the author's Mac, 4 vCPU, 3.8 GB RAM; Python 3.10.12,
  pandas 2.3.3, numpy 2.2.6; Node 22.23; TensorFlow.js 4.22.0, WASM backend (no GPU). The page uses WebGL by default.
- Runtime of one validation fit + scoring (selected setting): popularity 0.6 s, content 2.2 s, rules 0.2 s, MF 133 s, PPR 55 s.
  The MF test refit (weeks 11–30, k = 64) was split over two processes with a checkpoint (≈ 158 s + 59 s).

## 6. Data provenance

Kaggle community competition “sweet.tv — TV program recommender” (host `ogurbych`, 2020): view log `dataset11-30.csv` and broadcast
schedules `export_arh_*.csv`. The raw competition files are not redistributed here. The page needs `data/exam_data.js` (19.9 MB), a
packed, aggregated derivative (per user, programme and week counts; programme metadata), which is included so that the application
opens directly (also on GitHub Pages); `tools/build_app_data.py` rebuilds it from the Kaggle files (§2).

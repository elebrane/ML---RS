"""Step 2 + EDA part 2: join views with the labelled schedule and profile the result.

Inputs  (exam_quiz/Kaggle input/): export_arh_*.csv, submission.csv, plus Solution/cache/views.npz
        produced by eda_views.py.
Outputs: Solution/cache/pairs.csv.gz       one row per (user, airing) with overlap and screen_frac
         Solution/cache/airings.csv.gz     labelled airings (channel, start, duration, show, week)
         Solution/data/items.csv           one row per tv_show_id with metadata and airing counts
         Solution/results/eda/*.csv        summary tables for the report

Run from the exam_quiz folder:  python3 Solution/tools/build_interactions.py
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweettv_lib import (WATCHED_FRAC, apk, dedupe_airings, iso_week, join_views_airings,  # noqa: E402
                         read_schedule, top5_ground_truth)

ROOT = Path(__file__).resolve().parents[2]
INP = ROOT / "Kaggle input" / "sweettv-tv-program-recommender"
SOL = ROOT / "Solution"
OUT, CACHE, DATA = SOL / "results" / "eda", SOL / "cache", SOL / "data"
WINDOWS = {"train_11_20": (11, 20), "valid_21_30": (21, 30), "test_31_42": (31, 42)}


def window_of(week):
    for name, (a, b) in WINDOWS.items():
        if a <= week <= b:
            return name
    return "outside"


def main():
    t0 = time.time()
    for d in (OUT, CACHE, DATA):
        d.mkdir(parents=True, exist_ok=True)
    files = sorted(INP.glob("export_arh_*.csv"))
    if len(files) < 3 or not (INP / "submission.csv").exists():
        sys.exit(f"Missing inputs in {INP}: need 3 export_arh_*.csv files and submission.csv")

    # 1) Schedule: keep labelled rows only, record what was dropped.
    stats, parts = [], []
    for f in files:
        a, s = read_schedule(f)
        s["file"] = f.name
        s["distinct_shows"] = a["tv_show_id"].nunique()
        s["distinct_channels_labelled"] = a["channel_id"].nunique()
        s["categories"] = a["tv_show_category"].nunique() if "tv_show_category" in a else None
        stats.append(s)
        parts.append(a)
    pd.DataFrame(stats).set_index("file").to_csv(OUT / "schedule_summary.csv")
    air = dedupe_airings(pd.concat(parts, ignore_index=True))
    air["week"] = iso_week(air["start"].to_numpy())
    air["window"] = air["week"].map(window_of)
    air["duration"].describe(percentiles=[.05, .25, .5, .75, .95]).round(1).to_csv(OUT / "labelled_programme_duration_s.csv")

    # 2) Items table (metadata from the most recent airing; airing counts per window).
    meta_cols = [c for c in ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
                             "tv_show_genre_3", "year_of_production", "director", "actors"] if c in air]
    items = air.sort_values("start").groupby("tv_show_id")[meta_cols].last()
    counts = air.pivot_table(index="tv_show_id", columns="window", values="airing_id", aggfunc="size", fill_value=0)
    items = items.join(counts.add_prefix("airings_")).join(air.groupby("tv_show_id")["channel_id"].nunique().rename("n_channels"))
    items.to_csv(DATA / "items.csv")

    # 3) Join views (weeks 11-30 only exist in the log) with labelled airings of weeks <= 30.
    v = np.load(CACHE / "views.npz")
    hist = air[air["week"] <= 30]
    pairs = join_views_airings(v["user_idx"], v["channel"], v["start"], v["stop"], hist)
    pairs = pairs.merge(hist[["airing_id", "tv_show_id", "channel_id", "week", "window", "duration"]], on="airing_id")
    pairs.to_csv(CACHE / "pairs.csv.gz", index=False)
    air.drop(columns=meta_cols + ["channel_title"], errors="ignore").to_csv(CACHE / "airings.csv.gz", index=False)
    user_ids = v["user_ids"]

    # 4) How much viewing time lands on labelled programmes at all?
    total_view_s = int(v["dur"].sum())
    js = {"view_hours_total": round(total_view_s / 3600, 1),
          "view_hours_on_labelled_airings": round(pairs["overlap_s"].sum() / 3600, 1),
          "share_view_time_on_labelled": round(pairs["overlap_s"].sum() / total_view_s, 4),
          "user_airing_pairs": len(pairs),
          "pairs_watched_ge_0_8": int((pairs["screen_frac"] >= WATCHED_FRAC).sum())}
    pd.Series(js).to_csv(OUT / "join_summary.csv", header=["value"], index_label="metric")
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0000001]
    pd.cut(pairs["screen_frac"], bins, right=False).value_counts(sort=False).to_csv(OUT / "screen_frac_hist.csv")

    # 5) Ground truth built with the host's rule, per window.
    gt = {w: top5_ground_truth(pairs[pairs["window"] == w]) for w in ("train_11_20", "valid_21_30")}
    gt_all = top5_ground_truth(pairs)
    pd.Series({"users_with_5_watched_shows_train_11_20": len(gt["train_11_20"]),
               "users_with_5_watched_shows_valid_21_30": len(gt["valid_21_30"]),
               "users_with_5_watched_shows_11_30": len(gt_all)}).to_csv(OUT / "ground_truth_users.csv", header=["users"])

    # 6) Repeat-viewing diagnostic: is next-window top-5 already in the user's history?
    tr = pairs[(pairs["window"] == "train_11_20") & (pairs["screen_frac"] >= WATCHED_FRAC)]
    tr_freq = tr.groupby(["user_idx", "tv_show_id"]).size().rename("freq").reset_index()
    tr_hist = tr_freq.groupby("user_idx")["tv_show_id"].apply(set).to_dict()
    aired_train = set(air.loc[air["window"] == "train_11_20", "tv_show_id"])
    aired_valid = set(air.loc[air["window"] == "valid_21_30", "tv_show_id"])
    rows = []
    for u, shows in gt["valid_21_30"].items():
        h = tr_hist.get(u, set())
        rows.append({"user_idx": u, "gt_in_user_train_history": sum(s in h for s in shows),
                     "gt_aired_in_train_window": sum(s in aired_train for s in shows)})
    rep = pd.DataFrame(rows)
    pd.Series({"valid_users": len(rep),
               "share_gt_shows_already_watched_by_user": round(rep["gt_in_user_train_history"].sum() / (5 * len(rep)), 4),
               "share_gt_shows_aired_in_train_window": round(rep["gt_aired_in_train_window"].sum() / (5 * len(rep)), 4),
               "share_users_with_0_repeats": round((rep["gt_in_user_train_history"] == 0).mean(), 4),
               "share_users_with_5_repeats": round((rep["gt_in_user_train_history"] == 5).mean(), 4)}
              ).to_csv(OUT / "repeat_diagnostic.csv", header=["value"])

    # 7) Two quick reference points on the validation window (train 11-20 -> valid 21-30).
    #    a) popularity: shows watched (>=0.8) by most distinct users in 11-20, restricted to shows airing in 21-30;
    #    b) user's own most frequent shows in 11-20 airing in 21-30, padded with (a).
    pop = tr.groupby("tv_show_id")["user_idx"].nunique().sort_values(ascending=False)
    pop_list = [s for s in pop.index if s in aired_valid]
    users_v = list(gt["valid_21_30"])
    ap_pop, ap_own = [], []
    own_sorted = tr_freq.sort_values(["user_idx", "freq"], ascending=[True, False])
    own = own_sorted[own_sorted["tv_show_id"].isin(aired_valid)].groupby("user_idx")["tv_show_id"].apply(list).to_dict()
    for u in users_v:
        ap_pop.append(apk(gt["valid_21_30"][u], pop_list[:5]))
        lst = own.get(u, [])[:5]
        lst += [s for s in pop_list if s not in lst][: 5 - len(lst)]
        ap_own.append(apk(gt["valid_21_30"][u], lst))
    pd.Series({"valid_users": len(users_v), "MAP@5_popularity_airing": round(float(np.mean(ap_pop)), 5),
               "MAP@5_own_history_airing": round(float(np.mean(ap_own)), 5)}).to_csv(OUT / "reference_points_valid.csv", header=["value"])

    # 8) Test users and test candidate pool.
    sub = pd.read_csv(INP / "submission.csv", dtype={"user_id": "uint64"})
    pos = {int(x): i for i, x in enumerate(user_ids)}
    test_idx = [pos.get(int(x)) for x in sub["user_id"]]
    in_log = [i for i in test_idx if i is not None]
    watched_any = set(pairs.loc[pairs["screen_frac"] >= WATCHED_FRAC, "user_idx"])
    aired_test = set(air.loc[air["window"] == "test_31_42", "tv_show_id"])
    seen_hist = set(pairs.loc[pairs["screen_frac"] >= WATCHED_FRAC, "tv_show_id"])
    pd.Series({"test_users": len(sub), "test_users_in_view_log": len(in_log),
               "test_users_with_any_watched_airing_11_30": sum(i in watched_any for i in in_log),
               "test_users_with_5_watched_shows_11_30": sum(i in gt_all for i in in_log),
               "test_users_with_5_watched_shows_valid_21_30": sum(i in gt["valid_21_30"] for i in in_log),
               "shows_airing_in_test_31_42": len(aired_test),
               "of_which_watched_by_someone_11_30": len(aired_test & seen_hist)}).to_csv(OUT / "test_coverage.csv", header=["value"])
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

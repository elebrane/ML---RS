"""EDA step 1: profile the sweet.tv view log (dataset11-30.csv).

Reads the raw log in chunks (bounded memory), converts it to compact numeric
arrays, caches them in Solution/cache/views.npz for later steps, and writes
summary tables to Solution/results/eda/.

Run from the exam_quiz folder:
    python3 Solution/tools/eda_views.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # .../exam_quiz
RAW = ROOT / "Kaggle input" / "sweettv-tv-program-recommender" / "dataset11-30.csv"
OUT = ROOT / "Solution" / "results" / "eda"
CACHE = ROOT / "Solution" / "cache"
FMT = "%Y-%m-%d %H:%M:%S"


def load_views(path: Path, chunksize: int = 1_000_000) -> dict:
    """Stream the CSV and return numpy arrays (times as Unix seconds, naive local clock)."""
    parts = {k: [] for k in ("user", "channel", "start", "stop", "dur")}
    reader = pd.read_csv(path, chunksize=chunksize,
                         dtype={"user_id": "uint64", "vsetv_id": "int32", "duraton": "int64"})
    for chunk in reader:
        parts["user"].append(chunk["user_id"].to_numpy())
        parts["channel"].append(chunk["vsetv_id"].to_numpy())
        parts["start"].append(pd.to_datetime(chunk["start_time"], format=FMT).astype("int64").to_numpy() // 10**9)
        parts["stop"].append(pd.to_datetime(chunk["stop_time"], format=FMT).astype("int64").to_numpy() // 10**9)
        parts["dur"].append(chunk["duraton"].to_numpy())
    return {k: np.concatenate(v) for k, v in parts.items()}


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    v = load_views(RAW)
    n = len(v["user"])
    load_s = time.time() - t0

    # ISO week of the view start (2020-03-09 is Monday of ISO week 11).
    start_dt = pd.to_datetime(v["start"], unit="s")
    week = start_dt.isocalendar().week.to_numpy().astype(np.int16)
    users, user_idx = np.unique(v["user"], return_inverse=True)
    np.savez_compressed(CACHE / "views.npz", user_ids=users, user_idx=user_idx.astype(np.int32),
                        channel=v["channel"].astype(np.int32), start=v["start"], stop=v["stop"],
                        dur=v["dur"].astype(np.int64), week=week)

    dur = v["dur"]
    diff = v["stop"] - v["start"]
    summary = {
        "rows": n,
        "distinct_users": len(users),
        "distinct_channels": len(np.unique(v["channel"])),
        "first_start": str(start_dt.min()),
        "last_start": str(start_dt.max()),
        "last_stop": str(pd.to_datetime(v["stop"].max(), unit="s")),
        "min_week": int(week.min()),
        "max_week": int(week.max()),
        "rows_duration_equals_stop_minus_start": int((diff == dur).sum()),
        "rows_duration_nonpositive": int((dur <= 0).sum()),
        "rows_stop_before_start": int((diff < 0).sum()),
        "total_view_hours": round(float(dur.sum()) / 3600, 1),
        "load_seconds": round(load_s, 1),
    }
    for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        summary[f"duration_p{int(q*100)}_s"] = float(np.quantile(dur, q))
    summary["duration_max_s"] = int(dur.max())
    for cut in (60, 300, 600, 1800, 3600, 4 * 3600, 24 * 3600):
        summary[f"share_views_lt_{cut}s"] = round(float((dur < cut).mean()), 4)
        summary[f"share_time_in_views_ge_{cut}s"] = round(float(dur[dur >= cut].sum() / dur.sum()), 4)
    pd.Series(summary).to_csv(OUT / "views_summary.csv", header=["value"], index_label="metric")

    # Per-week activity.
    df = pd.DataFrame({"week": week, "user": user_idx, "dur": dur})
    per_week = df.groupby("week").agg(views=("dur", "size"), users=("user", "nunique"),
                                      view_hours=("dur", lambda s: round(s.sum() / 3600, 1)))
    per_week.to_csv(OUT / "views_per_week.csv")

    # Per-user activity distribution.
    pu = df.groupby("user").agg(views=("dur", "size"), hours=("dur", lambda s: s.sum() / 3600),
                                active_weeks=("week", "nunique"),
                                first_week=("week", "min"), last_week=("week", "max"))
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    pu.quantile(qs).round(2).to_csv(OUT / "per_user_quantiles.csv", index_label="quantile")
    split = pd.DataFrame({
        "users_active_in_weeks_11_20": [int((pu["first_week"] <= 20).sum())],
        "users_active_in_weeks_21_30": [int((pu["last_week"] >= 21).sum())],
        "users_active_in_both": [int(((pu["first_week"] <= 20) & (pu["last_week"] >= 21)).sum())],
        "users_only_11_20": [int((pu["last_week"] <= 20).sum())],
        "users_only_21_30": [int((pu["first_week"] >= 21).sum())],
    })
    split.T.to_csv(OUT / "users_by_window.csv", header=["users"], index_label="segment")

    # Channel concentration: share of a user's time on his top channel.
    uc = pd.DataFrame({"user": user_idx, "ch": v["channel"], "dur": dur}).groupby(["user", "ch"])["dur"].sum()
    top_share = (uc.groupby(level=0).max() / uc.groupby(level=0).sum())
    n_ch = uc.groupby(level=0).size()
    pd.DataFrame({"top_channel_time_share": top_share.quantile(qs).round(3),
                  "channels_per_user": n_ch.quantile(qs)}).to_csv(OUT / "channel_concentration.csv", index_label="quantile")
    ch_time = pd.Series(dur).groupby(v["channel"]).sum().sort_values(ascending=False)
    pd.DataFrame({"view_hours": (ch_time / 3600).round(1),
                  "cum_share": (ch_time.cumsum() / ch_time.sum()).round(4)}).head(30).to_csv(OUT / "top_channels.csv", index_label="channel_id")

    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

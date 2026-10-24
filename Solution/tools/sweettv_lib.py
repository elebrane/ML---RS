"""Shared helpers for the sweet.tv exam: schedule loading, view-to-programme join, MAP@K.

Join rule (fixed by the competition's own label definition, see Data Description):
  * rows with tv_show_id == 0 are unlabelled airtime and are dropped as items;
  * a view is matched to an airing on the SAME channel (vsetv_id == channel_id)
    when the two time intervals overlap;
  * screen_frac(user, airing) = sum of the user's overlapping seconds with that airing
    (capped at the airing duration) / airing duration;
  * an airing counts as "watched" when screen_frac >= 0.8.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FMT = "%Y-%m-%d %H:%M:%S"          # view log: 2020-03-09 00:00:02
SCHEDULE_FMT = "%d.%m.%Y %H:%M:%S"  # schedule:  09.03.2020 06:00:00 (day first!)
WATCHED_FRAC = 0.8
SCHEDULE_ITEM_COLS = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2",
                      "tv_show_genre_3", "year_of_production", "director", "actors", "channel_title"]


def to_unix(series: pd.Series) -> np.ndarray:
    """Parse timestamps to int64 Unix seconds (naive local clock).

    Only the two explicit formats used by the files are accepted; there is deliberately no
    format inference, because '09.03.2020' would otherwise risk a day/month swap.
    """
    for fmt in (FMT, SCHEDULE_FMT):
        try:
            return pd.to_datetime(series, format=fmt).astype("int64").to_numpy() // 10**9
        except (ValueError, TypeError):
            continue
    raise ValueError(f"Unrecognised timestamp format, e.g. {series.iloc[0]!r}")


def iso_week(unix_seconds: np.ndarray) -> np.ndarray:
    return pd.to_datetime(unix_seconds, unit="s").isocalendar().week.to_numpy().astype(np.int16)


def read_schedule(path, chunksize: int = 500_000):
    """Stream one export_arh_*.csv file.

    Returns (airings, stats): airings is a DataFrame of LABELLED rows only
    (tv_show_id != 0) with numeric channel/start/duration plus item metadata;
    stats counts all rows, labelled rows and unlabelled airtime.
    """
    keep, total, labelled, unl_seconds, lab_seconds = [], 0, 0, 0, 0
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        total += len(chunk)
        sid = pd.to_numeric(chunk["tv_show_id"], errors="coerce").fillna(0).astype("int64")
        dur = pd.to_numeric(chunk["duration"], errors="coerce").fillna(0).astype("int64")
        mask = sid != 0
        labelled += int(mask.sum())
        lab_seconds += int(dur[mask].sum())
        unl_seconds += int(dur[~mask].sum())
        part = chunk.loc[mask, ["channel_id", "start_time"] + [c for c in SCHEDULE_ITEM_COLS if c in chunk]].copy()
        part["tv_show_id"] = sid[mask].to_numpy()
        part["duration"] = dur[mask].to_numpy()
        part["start"] = to_unix(part.pop("start_time"))
        part["channel_id"] = pd.to_numeric(part["channel_id"], errors="coerce").astype("int64")
        keep.append(part)
    airings = pd.concat(keep, ignore_index=True)
    stats = {"rows": total, "labelled_rows": labelled, "unlabelled_rows": total - labelled,
             "labelled_share": round(labelled / total, 4) if total else 0.0,
             "labelled_hours": round(lab_seconds / 3600, 1), "unlabelled_hours": round(unl_seconds / 3600, 1)}
    return airings, stats


def dedupe_airings(airings: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (channel, start, show); drop non-positive durations; add end + airing_id."""
    a = airings[airings["duration"] > 0].drop_duplicates(["channel_id", "start", "tv_show_id"])
    a = a.sort_values(["channel_id", "start"]).reset_index(drop=True)
    a["end"] = a["start"] + a["duration"]
    a["airing_id"] = np.arange(len(a), dtype=np.int64)
    return a


def join_views_airings(v_user, v_channel, v_start, v_stop, airings: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (user, airing) with overlap seconds and screen_frac.

    For every channel, airings are sorted by start; for each view [s, e) the overlapping
    airings are those with start < e and end > s. Because airings on one channel can overlap
    slightly in the raw grid, the lower bound uses the running max of airing ends.
    """
    out = []
    v_channel = np.asarray(v_channel)
    order = np.argsort(v_channel, kind="stable")
    ch_sorted = v_channel[order]
    channels, first = np.unique(ch_sorted, return_index=True)
    bounds = list(first[1:]) + [len(order)]
    a_by_ch = {c: g for c, g in airings.groupby("channel_id", sort=False)}
    for c, lo_i, hi_i in zip(channels, first, bounds):
        g = a_by_ch.get(int(c))
        if g is None:
            continue
        idx = order[lo_i:hi_i]
        s, e, u = v_start[idx], v_stop[idx], v_user[idx]
        a_start = g["start"].to_numpy()
        a_end = g["end"].to_numpy()
        a_end_cummax = np.maximum.accumulate(a_end)
        lo = np.searchsorted(a_end_cummax, s, side="right")   # first airing whose (running) end > s
        hi = np.searchsorted(a_start, e, side="left")         # first airing starting at/after e
        cnt = np.clip(hi - lo, 0, None)
        if cnt.sum() == 0:
            continue
        rep = np.repeat(np.arange(len(idx)), cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        aj = np.repeat(lo, cnt) + offs
        ov = np.minimum(e[rep], a_end[aj]) - np.maximum(s[rep], a_start[aj])
        ok = ov > 0
        out.append(pd.DataFrame({"user_idx": u[rep][ok], "airing_id": g["airing_id"].to_numpy()[aj][ok],
                                 "overlap_s": ov[ok]}))
    if not out:
        return pd.DataFrame(columns=["user_idx", "airing_id", "overlap_s", "screen_frac"])
    pairs = pd.concat(out, ignore_index=True).groupby(["user_idx", "airing_id"], as_index=False)["overlap_s"].sum()
    dur = airings.set_index("airing_id")["duration"].reindex(pairs["airing_id"]).to_numpy()
    pairs["overlap_s"] = np.minimum(pairs["overlap_s"].to_numpy(), dur)
    pairs["screen_frac"] = pairs["overlap_s"] / dur
    return pairs


def apk(actual, predicted, k: int = 5) -> float:
    """AP@K exactly as in the course slides: sum of precision at relevant ranks / min(K, |R_u|)."""
    if not actual:
        return 0.0
    actual = set(actual)
    hits, score, seen = 0, 0.0, set()
    for r, p in enumerate(predicted[:k], start=1):
        if p in actual and p not in seen:
            hits += 1
            score += hits / r
        seen.add(p)
    return score / min(k, len(actual))


def mapk(actual_lists, predicted_lists, k: int = 5) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actual_lists, predicted_lists)]))


def top5_ground_truth(pairs_with_show: pd.DataFrame, k: int = 5) -> dict:
    """Host's label rule: per user, count watched (frac >= 0.8) airings per show, keep users with
    at least k such shows, take the k most frequent (ties broken by total watched seconds, then id)."""
    w = pairs_with_show[pairs_with_show["screen_frac"] >= WATCHED_FRAC]
    f = w.groupby(["user_idx", "tv_show_id"]).agg(freq=("airing_id", "size"), secs=("overlap_s", "sum")).reset_index()
    f = f.sort_values(["user_idx", "freq", "secs", "tv_show_id"], ascending=[True, False, False, True])
    top = f.groupby("user_idx").head(k)
    sizes = top.groupby("user_idx").size()
    keep = sizes[sizes == k].index
    return {u: g["tv_show_id"].tolist() for u, g in top[top["user_idx"].isin(keep)].groupby("user_idx")}

"""Step 3: pack the joined data into one browser-loadable file, Solution/data/exam_data.js.

The page (index.html) and the Node runner both read `window.EXAM` from this file.
Everything is derived from Solution/cache/{pairs,airings}.csv.gz (built by build_interactions.py),
Solution/data/items.csv and the competition's submission.csv.

Contents of window.EXAM
  users[u]            user_id as a STRING (64-bit ids do not fit a JS number)
  testUsers           user indices in submission.csv order
  items.*[i]          tv_show_id, title, category, genres, year, directors, actors
  air.{train,valid,test}[i]   number of labelled airings of item i in weeks 11-20 / 21-30 / 31-42
  hours.{...}[i*8+b]  airings of item i starting in 3-hour bin b (0 = 00-03h) per window
  chans.{...}[i]      channel ids item i airs on, per window
  log.{u,i,w,n,p,m}   one row per (user, item, week): n = airings watched >= 80 % (the host's rule),
                      p = airings watched 25-80 % (weak signal), m = minutes inside the n full views
  gtValid             {userIdx: [5 item indices]} = host's top-5 rule applied to weeks 21-30

Run from the exam_quiz folder:  python3 Solution/tools/build_app_data.py
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOL = ROOT / "Solution"
INP = ROOT / "Kaggle input" / "sweettv-tv-program-recommender"
WATCHED, PARTIAL = 0.8, 0.25
WIN = {"train": (11, 20), "valid": (21, 30), "test": (31, 42)}


def split_names(value, limit):
    if not isinstance(value, str) or not value.strip():
        return []
    return [s.strip() for s in value.split(",") if s.strip()][:limit]


def first_year(value):
    m = re.search(r"(19|20)\d{2}", str(value))
    return int(m.group(0)) if m else None


def main():
    t0 = time.time()
    pairs = pd.read_csv(SOL / "cache" / "pairs.csv.gz")
    air = pd.read_csv(SOL / "cache" / "airings.csv.gz")
    items = pd.read_csv(SOL / "data" / "items.csv")
    views = np.load(SOL / "cache" / "views.npz")
    users = [str(int(x)) for x in views["user_ids"]]
    pos = {u: k for k, u in enumerate(users)}

    item_ids = np.sort(items["tv_show_id"].unique())
    iidx = {int(s): k for k, s in enumerate(item_ids)}
    items = items.set_index("tv_show_id").loc[item_ids]

    # Per-window airing counts, channels and 3-hour bins (schedule only: what airs where and when).
    air["item"] = air["tv_show_id"].map(iidx)
    air["bin"] = (pd.to_datetime(air["start"], unit="s").dt.hour // 3).astype(int)
    n_items = len(item_ids)
    out_air, out_hours, out_chans = {}, {}, {}
    for name, (a, b) in WIN.items():
        w = air[(air["week"] >= a) & (air["week"] <= b)]
        out_air[name] = np.bincount(w["item"], minlength=n_items).tolist()
        hb = np.zeros(n_items * 8, dtype=np.int64)
        np.add.at(hb, w["item"].to_numpy() * 8 + w["bin"].to_numpy(), 1)
        out_hours[name] = hb.tolist()
        ch = w.groupby("item")["channel_id"].apply(lambda s: sorted(set(int(c) for c in s)))
        out_chans[name] = [ch.get(k, []) for k in range(n_items)]

    # Interaction log aggregated per (user, item, week).
    pairs["item"] = pairs["tv_show_id"].map(iidx)
    full = pairs["screen_frac"] >= WATCHED
    part = (pairs["screen_frac"] >= PARTIAL) & ~full
    pairs["n"] = full.astype(int)
    pairs["p"] = part.astype(int)
    pairs["m_s"] = np.where(full, pairs["overlap_s"], 0)
    agg = pairs[full | part].groupby(["user_idx", "item", "week"]).agg(n=("n", "sum"), p=("p", "sum"), m_s=("m_s", "sum")).reset_index()
    agg["m"] = (agg["m_s"] / 60).round().astype(int)
    agg = agg.sort_values(["user_idx", "week", "item"])

    # Validation ground truth with the host's rule on weeks 21-30 (ties: watched seconds, then id).
    v = pairs[full & (pairs["week"] >= 21) & (pairs["week"] <= 30)]
    f = v.groupby(["user_idx", "tv_show_id"]).agg(freq=("airing_id", "size"), secs=("overlap_s", "sum")).reset_index()
    f = f.sort_values(["user_idx", "freq", "secs", "tv_show_id"], ascending=[True, False, False, True])
    top = f.groupby("user_idx").head(5)
    gt = {int(u): [iidx[int(s)] for s in g["tv_show_id"]] for u, g in top.groupby("user_idx") if len(g) == 5}

    sub = pd.read_csv(INP / "submission.csv", dtype={"user_id": str})
    test_users = [pos[u] for u in sub["user_id"]]

    channel_titles = {}
    for f_ in sorted(INP.glob("export_arh_*.csv")):
        for chunk in pd.read_csv(f_, usecols=["channel_id", "channel_title"], chunksize=500_000):
            for c, t in chunk.drop_duplicates("channel_id").itertuples(index=False):
                channel_titles.setdefault(int(c), str(t))

    def col(name):
        return [None if pd.isna(x) else str(x) for x in items[name]] if name in items else [None] * n_items

    exam = {
        "meta": {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "watchedFrac": WATCHED, "partialFrac": PARTIAL,
                 "windows": WIN, "nUsers": len(users), "nItems": n_items, "nLogRows": int(len(agg)),
                 "source": "Kaggle sweettv-tv-program-recommender: dataset11-30.csv + export_arh_*.csv"},
        "users": users,
        "testUsers": test_users,
        "items": {
            "id": [int(x) for x in item_ids],
            "title": col("tv_show_title"),
            "category": col("tv_show_category"),
            "genres": [[g for g in (r.tv_show_genre_1, r.tv_show_genre_2, r.tv_show_genre_3) if isinstance(g, str) and g.strip()]
                       for r in items.itertuples()],
            "year": [first_year(x) for x in items["year_of_production"]],
            "directors": [split_names(x, 3) for x in items["director"]],
            "actors": [split_names(x, 5) for x in items["actors"]],
        },
        "channelTitles": channel_titles,
        "air": out_air, "hours": out_hours, "chans": out_chans,
        "log": {k: agg[c].astype(int).tolist() for k, c in [("u", "user_idx"), ("i", "item"), ("w", "week"), ("n", "n"), ("p", "p"), ("m", "m")]},
        "gtValid": gt,
    }
    out = SOL / "data" / "exam_data.js"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("// Auto-generated by tools/build_app_data.py — do not edit by hand.\n")
        fh.write("// Plain script (not a module): assigns the packed dataset to window.EXAM / globalThis.EXAM.\n")
        fh.write("(typeof window !== 'undefined' ? window : globalThis).EXAM = ")
        json.dump(exam, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    print(f"items={n_items} users={len(users)} logRows={len(agg)} gtValid={len(gt)} test={len(test_users)} "
          f"size={out.stat().st_size/1e6:.1f}MB in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""Trap check: are the view log and the schedule on the same clock?

If one file were in UTC and the other in local time, the join would silently misalign.
Viewers tend to switch channel at programme boundaries, so for the correct clock offset the
distance between a view's start and the nearest airing start on the same channel should pile
up near zero. We test offsets of -4..+4 hours applied to the view timestamps and report
(a) the share of view starts within 2 minutes of an airing start and (b) the number of
user-airing pairs watched >= 80 % on a 300k-view sample.

Run from the exam_quiz folder:  python3 Solution/tools/check_clock_alignment.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweettv_lib import join_views_airings  # noqa: E402

SOL = Path(__file__).resolve().parents[1]
v = np.load(SOL / "cache" / "views.npz")
air = pd.read_csv(SOL / "cache" / "airings.csv.gz")
air = air[air["week"] <= 30]
rng = np.random.default_rng(42)
idx = np.sort(rng.choice(len(v["user_idx"]), 300_000, replace=False))
u, ch, s, e = v["user_idx"][idx], v["channel"][idx], v["start"][idx], v["stop"][idx]
starts = {c: np.sort(g["start"].to_numpy()) for c, g in air.groupby("channel_id")}
rows = []
for h in range(-4, 5):
    off = h * 3600
    near, tot = 0, 0
    for c in np.unique(ch):
        if c not in starts:
            continue
        m = ch == c
        st = s[m] + off
        a = starts[c]
        j = np.clip(np.searchsorted(a, st), 1, len(a) - 1)
        d = np.minimum(np.abs(st - a[j - 1]), np.abs(a[j] - st))
        near += int((d <= 120).sum())
        tot += int(m.sum())
    pairs = join_views_airings(u, ch, s + off, e + off, air)
    rows.append({"offset_hours": h, "share_view_starts_within_2min_of_airing_start": round(near / tot, 4),
                 "watched_pairs_ge_0_8_in_sample": int((pairs["screen_frac"] >= 0.8).sum())})
out = pd.DataFrame(rows)
out.to_csv(SOL / "results" / "eda" / "clock_alignment.csv", index=False)
print(out.to_string(index=False))

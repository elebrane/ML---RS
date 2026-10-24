"""Hand-checkable fixtures for the join rule and MAP@5 (run: python3 Solution/tests/test_lib.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sweettv_lib import apk, dedupe_airings, join_views_airings, mapk  # noqa: E402

T = 1_600_000_000  # arbitrary base time (seconds)

# --- MAP@5: the two lists from the course slide, relevant = {A, B, C} ---
x = apk(["A", "B", "C"], ["A", "B", "D", "C", "E"])
y = apk(["A", "B", "C"], ["D", "A", "E", "B", "C"])
assert abs(x - (1 + 1 + 0.75) / 3) < 1e-12, x      # 0.9167
assert abs(y - (0.5 + 0.5 + 0.6) / 3) < 1e-12, y   # 0.5333
assert abs(mapk([["A", "B", "C"]] * 2, [["A", "B", "D", "C", "E"], ["D", "A", "E", "B", "C"]]) - (x + y) / 2) < 1e-12
assert apk(list("ABCDE"), list("VWXYZ")) == 0.0
assert apk(list("ABCDE"), list("ABCDE")) == 1.0
print(f"MAP slide example OK: X={x:.4f}  Y={y:.4f}")

# --- Join: channel 1 has show 101 [0,1800), show 102 [1800,3600); channel 2 has show 201 [0,3600) ---
air = dedupe_airings(pd.DataFrame({
    "channel_id": [1, 1, 2, 1], "start": [T, T + 1800, T, T],
    "duration": [1800, 1800, 3600, 1800], "tv_show_id": [101, 102, 201, 101]}))  # last row is a duplicate
assert len(air) == 3
users = np.array([0, 0, 1, 2])
chan = np.array([1, 1, 1, 2])
start = np.array([T, T + 1320, T + 600, T + 3000])
stop = np.array([T + 1200, T + 1800, T + 3000, T + 3600])
pairs = join_views_airings(users, chan, start, stop, air).merge(air[["airing_id", "tv_show_id"]], on="airing_id")
got = {(int(r.user_idx), int(r.tv_show_id)): (int(r.overlap_s), round(r.screen_frac, 4)) for r in pairs.itertuples()}
expected = {
    (0, 101): (1200 + 480, round(1680 / 1800, 4)),   # two pieces, 28 min of 30 -> 0.9333 -> watched
    (1, 101): (1200, round(1200 / 1800, 4)),         # 20 min of 30 -> 0.6667 -> not watched
    (1, 102): (1200, round(1200 / 1800, 4)),
    (2, 201): (600, round(600 / 3600, 4)),           # wrong channel for 101/102, 10 min of 60 on 201
}
assert got == expected, got
print("Join fixture OK:", got)

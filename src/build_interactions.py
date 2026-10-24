# -*- coding: utf-8 -*-
"""Step 1: load the sweet.tv data, report its shape, and turn channel views into
user-show watch time by joining each view with the programme schedule.

A view is (user, channel, start, stop). The schedule gives, per channel, the shows
with start time and duration. Each view is split over every show it overlaps, and
the overlap in seconds is credited to that (user, show). Shows with tv_show_id 0
are "irrelevant" per the competition and are dropped AFTER the overlap is computed,
so their time is not given to a neighbouring show.

Outputs (data/processed/):
  schedule.parquet      all schedule rows, weeks 11-42, with a parsed start/end
  shows.parquet         one row per tv_show_id with its metadata
  interactions.parquet  user_id, tv_show_id, week, seconds (weeks 11-30 only)
Usage: python build_interactions.py <data dir>
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

DATA = Path(sys.argv[1])
OUT = DATA / "processed"
OUT.mkdir(exist_ok=True)
t0 = time.time()

# ---------------------------------------------------------------- schedule
parts = []
for name in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"):
    s = pd.read_csv(DATA / name, dtype={"tv_show_id": "int64", "channel_id": "int64"}, engine="pyarrow")
    s["source"] = name
    parts.append(s)
sched = pd.concat(parts, ignore_index=True)
sched["start"] = pd.to_datetime(sched["start_time"], format="%d.%m.%Y %H:%M:%S")
sched["end"] = sched["start"] + pd.to_timedelta(sched["duration"], unit="s")
print("schedule rows", len(sched), "channels", sched.channel_id.nunique(),
      "range", sched.start.min(), sched.end.max())
for name, g in sched.groupby("source"):
    print("  %-28s rows %8d  %s .. %s  shows(non-zero) %d" % (name, len(g), g.start.min(), g.start.max(),
                                                           g.loc[g.tv_show_id != 0, "tv_show_id"].nunique()))
print("  share of rows with tv_show_id 0: %.3f" % (sched.tv_show_id == 0).mean())

# overlaps inside a channel would double-count time; measure them
sched = sched.sort_values(["channel_id", "start"]).reset_index(drop=True)
nxt = sched.groupby("channel_id")["start"].shift(-1)
overlap = (sched["end"] > nxt) & nxt.notna()
print("  rows overlapping the next show on the same channel:", int(overlap.sum()),
      "| duplicate (channel,start) rows:", int(sched.duplicated(["channel_id", "start"]).sum()))
# The week of a timestamp, counted like the file names (week 11 starts 2020-03-09).
WEEK0 = pd.Timestamp("2020-03-09")
week_of = lambda ts: ((ts - WEEK0).dt.days // 7 + 11).astype("int16")
# Epoch seconds whatever unit pandas stores (pandas 3 parses to microseconds).
secs = lambda ts: ts.astype("datetime64[s]").astype("int64").to_numpy()
sched["week"] = week_of(sched["start"])
sched.to_parquet(OUT / "schedule.parquet", index=False)

meta_cols = ["tv_show_title", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
             "year_of_production", "director", "actors"]
shows = (sched[sched.tv_show_id != 0].sort_values("start")
         .groupby("tv_show_id").agg(**{c: (c, "last") for c in meta_cols},
                                     channel_id=("channel_id", lambda x: x.mode().iat[0]),
                                     n_airings=("start", "size"),
                                     first_week=("week", "min"), last_week=("week", "max")))
shows.to_parquet(OUT / "shows.parquet")
print("shows (non-zero ids):", len(shows), "| airing in weeks 31-42:", int((shows.last_week >= 31).sum()),
      "| only in weeks 31-42:", int((shows.first_week >= 31).sum()))

# ---------------------------------------------------------------- views
# user ids reach 1.8e19 (above int64). pandas' pyarrow engine parses them as float
# before any dtype is applied (1.000835299983273e+19), so read the column as text
# directly with pyarrow: no id is ever rounded.
v = pacsv.read_csv(DATA / "dataset11-30.csv", convert_options=pacsv.ConvertOptions(
    column_types={"user_id": pa.string(), "vsetv_id": pa.int64(), "duraton": pa.int64(),
                  "start_time": pa.string(), "stop_time": pa.string()})).to_pandas()
v["start"] = pd.to_datetime(v["start_time"], format="%Y-%m-%d %H:%M:%S")
v["stop"] = pd.to_datetime(v["stop_time"], format="%Y-%m-%d %H:%M:%S")
print("\nviews", len(v), "users", v.user_id.nunique(), "channels", v.vsetv_id.nunique(),
      "range", v.start.min(), v.stop.max())
dur = (v.stop - v.start).dt.total_seconds()
print("  duraton column equals stop-start: %.4f of rows; median %.0f s, mean %.0f s, max %.0f s" %
      ((dur == v.duraton).mean(), v.duraton.median(), v.duraton.mean(), v.duraton.max()))
print("  views with stop < start:", int((dur < 0).sum()), "| channels missing from schedule:",
      sorted(set(v.vsetv_id) - set(sched.channel_id))[:20])

sub = pd.read_csv(DATA / "submission.csv", dtype=str)
print("submission users", len(sub), "unique", sub.user_id.nunique(),
      "| with views in weeks 11-30:", int(sub.user_id.isin(set(v.user_id)).sum()))

# ---------------------------------------------------------------- interval join
# Global sort key = channel * 1e10 + seconds, so one searchsorted works for all channels.
BIG = np.int64(10_000_000_000)
s_key = sched.channel_id.to_numpy(np.int64) * BIG + secs(sched.start)
s_end = sched.channel_id.to_numpy(np.int64) * BIG + secs(sched.end)
assert np.all(np.diff(s_key) >= 0)
v_ch = v.vsetv_id.to_numpy(np.int64) * BIG
v_a = v_ch + secs(v.start)
v_b = v_ch + secs(v.stop)
i0 = np.searchsorted(s_key, v_a, side="right") - 1  # show running at the view start
i1 = np.searchsorted(s_key, v_b, side="left") - 1   # last show starting before the view stops
i0 = np.maximum(i0, 0)
n = np.maximum(i1 - i0 + 1, 0)
print("\nshows touched per view: mean %.2f, max %d, views touching none %d" % (n.mean(), n.max(), int((n == 0).sum())))
rows = np.repeat(np.arange(len(v)), n)
offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
si = np.repeat(i0, n) + offs
same_channel = (s_key[si] // BIG) == (v_a[rows] // BIG)
ov = np.minimum(s_end[si], v_b[rows]) - np.maximum(s_key[si], v_a[rows])
keep = same_channel & (ov > 0)
# Share of the airing seen in this one view (one view = one channel without switching).
# The organisers count a show as watched when over 80% of an airing is seen this way
# (N. Melentev, Towards Data Science, 2021); 30% and 50% are kept as weaker signals.
slot_len = sched.duration.to_numpy(np.float64)[si[keep]]
frac = np.where(slot_len > 0, ov[keep] / np.maximum(slot_len, 1), 0.0)
pairs = pd.DataFrame({"user_id": v.user_id.to_numpy()[rows[keep]],
                      "tv_show_id": sched.tv_show_id.to_numpy()[si[keep]],
                      "week": sched.week.to_numpy()[si[keep]],
                      "seconds": ov[keep],
                      "n30": (frac > 0.3).astype(np.int32),
                      "n50": (frac > 0.5).astype(np.int32),
                      "n80": (frac > 0.8).astype(np.int32)})
total_view = float(v.duraton.sum())
matched = float(pairs.seconds.sum())
irrelevant = float(pairs.loc[pairs.tv_show_id == 0, "seconds"].sum())
print("viewing seconds: total %.3e, matched to a schedule slot %.1f%%, of which tv_show_id 0 %.1f%%" %
      (total_view, 100 * matched / total_view, 100 * irrelevant / max(matched, 1)))
pairs = pairs[pairs.tv_show_id != 0]
inter = pairs.groupby(["user_id", "tv_show_id", "week"], as_index=False)[["seconds", "n30", "n50", "n80"]].sum()
print("airings seen in one view: over 30%% %d, over 50%% %d, over 80%% %d" %
      (inter.n30.sum(), inter.n50.sum(), inter.n80.sum()))
inter.to_parquet(OUT / "interactions.parquet", index=False)
assert inter.user_id.str.fullmatch(r"\d+").all(), "a user id is not a plain integer string"
assert set(sub.user_id) <= set(inter.user_id) | set(v.user_id)
print("interactions (user, show, week):", len(inter), "| users", inter.user_id.nunique(),
      "| shows", inter.tv_show_id.nunique(), "| weeks", inter.week.min(), "-", inter.week.max())
per_user = inter.groupby("user_id").tv_show_id.nunique()
print("distinct shows per user: median %.0f, 10%% %.0f, 90%% %.0f" % tuple(per_user.quantile([.5, .1, .9])))
print("done in %.0f s" % (time.time() - t0))

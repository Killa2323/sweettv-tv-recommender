# -*- coding: utf-8 -*-
"""Pick the two users for the report by a stated rule and collect their evidence.

Rule: among submission users whose total watch time (weeks 11-30) lies between the
25th and 75th percentile (typical activity), user A has the LARGEST share of watch
time on their five most watched shows (a loyal viewer) and user B the SMALLEST
(an explorer). For each: history Top-5, every method's test Top-5 with show
attributes, and every method's AP@5 for that user on validation fold A.
Writes results/two_users.json. Usage: python two_users.py
"""
import json
import numpy as np
import pandas as pd
from common import load, weeks, on_air, true_top, apk, VAL, TEST, ROOT, PROC
import models as M

inter, shows, sched, sub = load()
users = list(sub.user_id)
titles = pd.read_parquet(PROC / "schedule.parquet", columns=["channel_id", "channel_title"]).drop_duplicates("channel_id")
chan_title = dict(zip(titles.channel_id, titles.channel_title))

h = weeks(inter, TEST["hist"])
h = h[h.user_id.isin(users)].groupby(["user_id", "tv_show_id"], as_index=False)["seconds"].sum()
tot = h.groupby("user_id").seconds.sum()
top5 = h.sort_values("seconds", ascending=False).groupby("user_id").head(5).groupby("user_id").seconds.sum()
stats = pd.DataFrame({"hours": tot / 3600, "top5_share": top5 / tot,
                      "shows": h.groupby("user_id").tv_show_id.nunique()})
lo, hi = stats.hours.quantile([0.25, 0.75])
band = stats[(stats.hours >= lo) & (stats.hours <= hi)]
A = band.top5_share.idxmax()
B = band.top5_share.idxmin()
print("activity band %.0f-%.0f h, %d users; median top-5 share %.2f" % (lo, hi, len(band), stats.top5_share.median()))


def describe(s):
    m = shows.loc[s]
    genres = "/".join(g for g in (m.tv_show_genre_1, m.tv_show_genre_2, m.tv_show_genre_3) if isinstance(g, str) and g)
    return {"id": int(s), "title": str(m.tv_show_title), "category": str(m.tv_show_category),
            "genres": genres, "channel": str(chan_title.get(int(m.channel_id), m.channel_id))}


test_lists = json.load(open(ROOT / "results" / "test_lists.json"))["lists"]

# validation fold A: every method's AP@5 for the two users
truth = true_top(inter, VAL["target"])
ctx = M.Context(inter, shows, VAL["hist"], on_air(sched, VAL["target"]), users)
s_ = weeks(sched, VAL["hist"]); A_h = s_[s_.tv_show_id != 0].groupby("tv_show_id").duration.sum().reindex(ctx.shows).fillna(0).to_numpy()
s_ = weeks(sched, VAL["target"]); A_t = s_[s_.tv_show_id != 0].groupby("tv_show_id").duration.sum().reindex(ctx.shows).fillna(0).to_numpy()
H = M.history(ctx)
R = H / (A_h + 1e5)[None, :] * A_t[None, :]
MF = M.mf(ctx, k=32)
val_scores = {"history": H, "popularity": M.popularity(ctx), "cbf": M.cbf(ctx), "mf": MF,
              "ppr": M.ppr(ctx, alpha=0.3)[0], "rate": R, "rate_mf": M.blend(R, MF, 0.1)}
val_lists = {k: ctx.topk(v) for k, v in val_scores.items()}

out = {"rule": "typical activity (25th-75th percentile of hours), max / min top-5 share",
       "band_hours": [float(lo), float(hi)], "band_users": int(len(band)),
       "median_top5_share": float(stats.top5_share.median()), "users": {}}
for label, u in (("A", A), ("B", B)):
    hu = h[h.user_id == u].sort_values("seconds", ascending=False)
    hist_ids = set(hu.tv_show_id)
    rec = {"user_id": u, "hours": float(stats.loc[u, "hours"]), "top5_share": float(stats.loc[u, "top5_share"]),
           "distinct_shows": int(stats.loc[u, "shows"]),
           "category_share": {str(k): float(v) for k, v in (hu.groupby(shows.reindex(hu.tv_show_id).tv_show_category.values)
                              .seconds.sum() / hu.seconds.sum()).sort_values(ascending=False).head(5).items()},
           "film_shows": int(shows.reindex(hu.tv_show_id).tv_show_title.astype(str).str.startswith(("Х/ф", "М/ф")).sum()),
           "history_top5": [dict(describe(s), hours=round(float(sec) / 3600, 1)) for s, sec in hu.head(5)[["tv_show_id", "seconds"]].to_numpy()],
           "methods": {}, "validation_ap": {}, "validation_truth": [describe(s) for s in truth.get(u, [])]}
    for m, lists in test_lists.items():
        rec["methods"][m] = [dict(describe(s), seen=bool(s in hist_ids)) for s in lists[u]]
        rec["validation_ap"][m] = apk(truth.get(u, []), val_lists[m].get(u, []))
    out["users"][label] = rec
    print("\nuser %s %s: %.0f h, top-5 share %.2f, %d shows" % (label, u, rec["hours"], rec["top5_share"], rec["distinct_shows"]))
    print("  history:", "; ".join("%s (%s, %.1f h)" % (x["title"][:30], x["channel"], x["hours"]) for x in rec["history_top5"]))
    for m, lst in rec["methods"].items():
        print("  %-10s AP(val) %.2f | %s" % (m, rec["validation_ap"][m], "; ".join(
            "%s%s" % (x["title"][:28], "" if x["seen"] else " [new]") for x in lst)))
json.dump(out, open(ROOT / "results" / "two_users.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

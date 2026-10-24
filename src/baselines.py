# -*- coding: utf-8 -*-
"""Reference points on the validation split (history weeks 11-18, future 19-30):
global popularity, the user's own most watched shows, and the same with a
recency weight. Every list is restricted to shows on air in the future window and
filled up to 5 with popular on-air shows.
Usage: python baselines.py
"""
import numpy as np
import pandas as pd
from common import load, weeks, on_air, true_top, mapk, VAL, K

inter, shows, sched, sub = load()
users = list(sub.user_id)
truth = true_top(inter, VAL["target"])
air = on_air(sched, VAL["target"])
hist = weeks(inter, VAL["hist"])
print("submission users with a future Top-5: %d of %d; median true list length %d" %
      (sum(u in truth for u in users), len(users), int(np.median([len(truth.get(u, [])) for u in users]))))
print("shows on air in weeks 19-30: %d; true Top-5 items on air: %.3f; true items seen in history: %.3f" % (
    len(air), np.mean([s in air for u in users for s in truth.get(u, [])]),
    np.mean([s in set(hist.loc[hist.user_id == u, "tv_show_id"]) for u in users[:300] for s in truth.get(u, [])])))

pop = (hist[hist.tv_show_id.isin(air)].groupby("tv_show_id")["seconds"].sum()
       .sort_values(ascending=False).index.tolist())


def fill(lst, k=K):
    out = [s for s in lst if s in air][:k]
    for s in pop:
        if len(out) >= k:
            break
        if s not in out:
            out.append(s)
    return out


def user_rank(df, weight=None):
    d = df[df.tv_show_id.isin(air)]
    w = d.seconds if weight is None else d.seconds * weight(d.week)
    d = d.assign(w=w.to_numpy()).groupby(["user_id", "tv_show_id"], as_index=False)["w"].sum()
    d = d.sort_values(["user_id", "w", "tv_show_id"], ascending=[True, False, True])
    return d.groupby("user_id")["tv_show_id"].apply(list).to_dict()


res = {"popularity": {u: fill([]) for u in users}}
r = user_rank(hist)
res["history (seconds)"] = {u: fill(r.get(u, [])) for u in users}
last = VAL["hist"][1]
for h in (1, 2, 4):
    r = user_rank(hist, lambda w, h=h: 0.5 ** ((last - w) / h))
    res["history, half-life %d wk" % h] = {u: fill(r.get(u, [])) for u in users}
for name, preds in res.items():
    print("%-28s MAP@5 = %.4f" % (name, mapk(truth, preds, users)))

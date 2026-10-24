# -*- coding: utf-8 -*-
"""Familiar vs new, per method. On validation fold A: MAP@5, hits (true Top-5 shows
placed in the list), split into hits the user had already watched and hits that
were new to the user. On the test lists: the share of the 6,300 slots (1,260 users
x 5) given to shows the user never watched in weeks 11-30.
Writes results/discovery.json. Usage: python discovery.py
"""
import json
import numpy as np
from common import load, weeks, on_air, true_top, mapk, VAL, TEST, ROOT
import models as M

inter, shows, sched, sub = load()
users = list(sub.user_id)
truth = true_top(inter, VAL["target"])
ctx = M.Context(inter, shows, VAL["hist"], on_air(sched, VAL["target"]), users)


def airtime(span):
    s = weeks(sched, span)
    return s[s.tv_show_id != 0].groupby("tv_show_id").duration.sum().reindex(ctx.shows).fillna(0).to_numpy()


A_h, A_t = airtime(VAL["hist"]), airtime(VAL["target"])
H = M.history(ctx)
R = H / (A_h + 1e5)[None, :] * A_t[None, :]
MF = M.mf(ctx, k=32)
scores = {"history": H, "popularity": M.popularity(ctx), "cbf": M.cbf(ctx), "mf": MF,
          "ppr": M.ppr(ctx, alpha=0.3)[0], "rate": R, "rate_mf": M.blend(R, MF, 0.1)}
seen = {u: set(ctx.shows[ctx.X[ctx.uix[u]].indices]) for u in users}
out = {"validation": {}, "test_unseen_share": {}}
for name, S in scores.items():
    lists = ctx.topk(S)
    hits_seen = sum(1 for u in users for s in lists[u] if s in truth.get(u, []) and s in seen[u])
    hits_new = sum(1 for u in users for s in lists[u] if s in truth.get(u, []) and s not in seen[u])
    slots_new = sum(1 for u in users for s in lists[u] if s not in seen[u])
    out["validation"][name] = {"map": mapk(truth, lists, users), "hits": hits_seen + hits_new,
                               "hits_seen": hits_seen, "hits_new": hits_new, "slots_new": slots_new,
                               "users_with_hit": sum(1 for u in users if set(lists[u]) & set(truth.get(u, [])))}
    print("%-10s MAP %.4f hits %4d (seen %4d, new %3d) | new slots %4d | users with a hit %d" % (
        name, out["validation"][name]["map"], hits_seen + hits_new, hits_seen, hits_new, slots_new,
        out["validation"][name]["users_with_hit"]))

test_lists = json.load(open(ROOT / "results" / "test_lists.json"))["lists"]
th = weeks(inter, TEST["hist"])
th = th[th.user_id.isin(users)]
seen_t = th.groupby("user_id").tv_show_id.agg(set).to_dict()
for name, lists in test_lists.items():
    out["test_unseen_share"][name] = float(np.mean([s not in seen_t.get(u, set()) for u in users for s in lists[u]]))
print("test: share of slots with shows new to the user:",
      {k: round(v, 3) for k, v in out["test_unseen_share"].items()})
json.dump(out, open(ROOT / "results" / "discovery.json", "w"), indent=1)

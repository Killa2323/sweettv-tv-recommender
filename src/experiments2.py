# -*- coding: utf-8 -*-
"""Second validation round: where the true Top-5 items come from, PPR with a
smaller alpha, and a schedule-aware history that uses the provided programme
schedule of the predicted window (allowed: it is metadata, not future viewing).

  airtime(s, span)  total seconds show s is broadcast in the weeks of span
  rate score        user's watch seconds of s / airtime(s, hist) x airtime(s, target)^b
Usage: python experiments2.py
"""
import json, time
import numpy as np
from common import load, weeks, on_air, true_top, mapk, VAL, ROOT
import models as M

t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
truth = true_top(inter, VAL["target"])
air = on_air(sched, VAL["target"])
ctx = M.Context(inter, shows, VAL["hist"], air, users)
res = {}

# 1. Where do the true Top-5 items come from?
seen_by_user = {u: set(ctx.shows[ctx.X[ctx.uix[u]].indices]) for u in users}
seen_any = set(ctx.shows[np.unique(ctx.X.indices)])
items = [(u, s) for u in users for s in truth.get(u, [])]
cats = {"seen by the user": sum(s in seen_by_user[u] for u, s in items),
        "seen only by others": sum(s not in seen_by_user[u] and s in seen_any for u, s in items),
        "new (no history views)": sum(s not in seen_any for u, s in items)}
res["truth_sources"] = {k: v / len(items) for k, v in cats.items()}
print("true Top-5 items:", len(items), {k: "%.3f" % v for k, v in res["truth_sources"].items()})


def airtime(span):
    s = weeks(sched, span)
    s = s[s.tv_show_id != 0].groupby("tv_show_id")["duration"].sum()
    return s.reindex(ctx.shows).fillna(0).to_numpy(np.float64)


A_h, A_t = airtime(VAL["hist"]), airtime(VAL["target"])
H = M.history(ctx)
print("history                 MAP@5 = %.4f" % mapk(truth, ctx.topk(H), users))
res["rate"] = {}
for b in (0.5, 1.0):
    for smooth in (3600.0, 36000.0):
        R = H / (A_h + smooth)[None, :] * (A_t ** b)[None, :]
        v = mapk(truth, ctx.topk(R), users)
        res["rate"]["b=%.1f smooth=%d" % (b, smooth)] = v
        print("rate b=%.1f smooth=%-6d MAP@5 = %.4f" % (b, smooth, v))
best_rate = max(res["rate"], key=res["rate"].get)
b, smooth = (float(x.split("=")[1]) for x in best_rate.split())
R = H / (A_h + smooth)[None, :] * (A_t ** b)[None, :]

res["ppr"] = {}
pprs = {}
for a in (0.1, 0.2):
    pprs[a], c = M.ppr(ctx, alpha=a)
    res["ppr"][str(a)] = {"map": mapk(truth, ctx.topk(pprs[a]), users), "conv": c}
    print("ppr alpha %.1f: MAP@5 = %.4f (%d it, res %.1e)  %.0f s" % (a, res["ppr"][str(a)]["map"], *c[:2], time.time() - t0))

MF = M.mf(ctx, k=32)
P = M.popularity(ctx) * (A_t > 0)[None, :]
res["rate_blend"] = {}
for name, S in (("mf", MF), ("popularity", P), ("ppr0.2", pprs[0.2])):
    for lam in (0.01, 0.03, 0.1, 0.3):
        v = mapk(truth, ctx.topk(M.blend(R, S, lam)), users)
        res["rate_blend"]["%s %.2f" % (name, lam)] = v
    best = max((k for k in res["rate_blend"] if k.startswith(name)), key=res["rate_blend"].get)
    print("rate + %-10s best %s MAP@5 = %.4f" % (name, best, res["rate_blend"][best]))
json.dump(res, open(ROOT / "results" / "validation2.json", "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))

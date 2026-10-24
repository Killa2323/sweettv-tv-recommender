# -*- coding: utf-8 -*-
"""Third validation round: content-based scores that can reach shows nobody has
watched yet (12% of the true Top-5 in validation).

  title family   new seasons often get a new tv_show_id but keep the title, so the
                 user's watch rate is pooled over shows with the same normalised title
  channel        user's watch rate on the show's channel x the show's future airtime
Both use only the provided schedule metadata of the predicted window.
Usage: python experiments3.py
"""
import json, re, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from common import load, weeks, on_air, true_top, mapk, VAL, ROOT
import models as M

t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
truth = true_top(inter, VAL["target"])
air = on_air(sched, VAL["target"])
ctx = M.Context(inter, shows, VAL["hist"], air, users)
res = {}


def airtime(span):
    s = weeks(sched, span)
    s = s[s.tv_show_id != 0].groupby("tv_show_id")["duration"].sum()
    return s.reindex(ctx.shows).fillna(0).to_numpy(np.float64)


def title_key(t):
    t = str(t).lower().strip()
    m = re.match(r"^['\"«](.+?)['\"»]", t)
    t = m.group(1) if m else re.split(r"[.,(:]", t)[0]
    return re.sub(r"[^a-zа-яёіїєґ ]+", " ", t).strip() or "?"


A_h, A_t = airtime(VAL["hist"]), airtime(VAL["target"])
H = M.history(ctx)


def pooled_rate(groups, smooth):
    """Rate of each target user on each group, mapped back to shows: seconds watched on
    the group's shows / (group airtime in history + smooth)."""
    codes, uniq = pd.factorize(groups)
    G = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(uniq)))
    watched = sp.csr_matrix(H) @ G                      # [users, groups]
    aired = G.T @ A_h                                   # [groups]
    rate = watched.toarray() / (aired + smooth)[None, :]
    return rate[:, codes]                               # back to [users, shows]


res["rate"] = {}
for smooth in (36000.0, 1e5, 3e5):
    R = H / (A_h + smooth)[None, :] * A_t[None, :]
    res["rate"][str(int(smooth))] = mapk(truth, ctx.topk(R), users)
print("show rate:", {k: "%.4f" % v for k, v in res["rate"].items()})
s_best = float(max(res["rate"], key=res["rate"].get))
R = H / (A_h + s_best)[None, :] * A_t[None, :]

keys = ctx.meta.tv_show_title.map(title_key).to_numpy()
res["title_family"] = {}
for smooth in (36000.0, 1e5, 3e5):
    T = pooled_rate(keys, smooth) * A_t[None, :]
    res["title_family"][str(int(smooth))] = mapk(truth, ctx.topk(T), users)
print("title family:", {k: "%.4f" % v for k, v in res["title_family"].items()})
t_best = float(max(res["title_family"], key=res["title_family"].get))
T = pooled_rate(keys, t_best) * A_t[None, :]

chan = ctx.meta.channel_id.fillna(-1).astype(int).to_numpy()
Cn = pooled_rate(chan, 36000.0) * A_t[None, :]
res["channel"] = mapk(truth, ctx.topk(Cn), users)
print("channel rate x airtime: %.4f" % res["channel"])

MF = M.mf(ctx, k=32)
res["combo"] = {}
for base_name, base in (("show rate", R), ("title family", T)):
    for lam_mf in (0.0, 0.05, 0.1, 0.2):
        for lam_c in (0.0, 0.01, 0.03):
            S = M.blend(base, MF, lam_mf) + lam_c * M.row_norm(Cn)
            v = mapk(truth, ctx.topk(S), users)
            res["combo"]["%s | mf %.2f | channel %.2f" % (base_name, lam_mf, lam_c)] = v
best = sorted(res["combo"].items(), key=lambda kv: -kv[1])[:6]
for k, v in best:
    print("%-45s MAP@5 = %.4f" % (k, v))
json.dump(res, open(ROOT / "results" / "validation3.json", "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))

# -*- coding: utf-8 -*-
"""Compare the methods on the validation split (history weeks 11-18, future 19-30)
for the 1,260 submission users, tune each method's setting and the blend weight,
and write every number to results/validation.json.
Usage: python validate.py
"""
import json, time
from pathlib import Path
import numpy as np
from common import load, on_air, true_top, mapk, VAL, ROOT
import models as M

t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
truth = true_top(inter, VAL["target"])
air = on_air(sched, VAL["target"])
ctx = M.Context(inter, shows, VAL["hist"], air, users)
print("matrix %d users x %d shows, %d non-zero; candidates on air %d" %
      (ctx.X.shape[0], ctx.X.shape[1], ctx.X.nnz, int(ctx.air_mask.sum())))

out = {"split": VAL, "users": len(users), "users_with_truth": sum(u in truth for u in users), "single": {}, "blend": {}}


def score(name, scores):
    v = mapk(truth, ctx.topk(scores), users)
    out["single"][name] = v
    print("%-22s MAP@5 = %.4f   (%.0f s)" % (name, v, time.time() - t0))
    return v


H = M.history(ctx)
score("history", H)
P = M.popularity(ctx)
score("popularity", P)
C = M.cbf(ctx)
score("cbf", C)
mfs = {}
for k in (16, 32, 64):
    mfs[k] = M.mf(ctx, k=k)
    score("mf k=%d" % k, mfs[k])
pprs, conv = {}, {}
for a in (0.3, 0.5, 0.7, 0.85):
    pprs[a], conv[a] = M.ppr(ctx, alpha=a)
    print("   ppr alpha %.2f: %d iterations, residual %.1e, converged %s" % (a, *conv[a]))
    score("ppr alpha=%.2f" % a, pprs[a])
out["ppr_convergence"] = {str(a): c for a, c in conv.items()}

best_mf = max(mfs, key=lambda k: out["single"]["mf k=%d" % k])
best_a = max(pprs, key=lambda a: out["single"]["ppr alpha=%.2f" % a])
out["best_mf_k"], out["best_ppr_alpha"] = int(best_mf), float(best_a)
cands = {"popularity": P, "cbf": C, "mf": mfs[best_mf], "ppr": pprs[best_a]}
for name, S in cands.items():
    row = {}
    for lam in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
        row[str(lam)] = mapk(truth, ctx.topk(M.blend(H, S, lam)), users)
    out["blend"][name] = row
    best = max(row, key=row.get)
    print("history + %-10s best lambda %-5s MAP@5 = %.4f   %s" % (name, best, row[best],
          " ".join("%s:%.4f" % kv for kv in row.items())))

Path(ROOT / "results").mkdir(exist_ok=True)
json.dump(out, open(ROOT / "results" / "validation.json", "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))

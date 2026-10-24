# -*- coding: utf-8 -*-
"""Build Kaggle submission files for the test window (history weeks 11-30, predicted
weeks 31-42) with the settings chosen on validation, check their format, and save
every method's Top-5 per user for the report.

Outputs: submissions/sub_<name>.csv, results/test_lists.json
Usage: python predict.py
"""
import json, time
from pathlib import Path
import numpy as np
from common import load, weeks, on_air, TEST, ROOT, K
import models as M

SMOOTH = 1e5        # chosen on two validation folds (results/tuning.json)
LAM_MF, MF_K = 0.1, 32
PPR_ALPHA = 0.3

t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
air = on_air(sched, TEST["target"])
ctx = M.Context(inter, shows, TEST["hist"], air, users)
print("test matrix %d x %d, nnz %d; shows on air in weeks 31-42: %d" %
      (ctx.X.shape[0], ctx.X.shape[1], ctx.X.nnz, len(air)))


def airtime(span):
    s = weeks(sched, span)
    s = s[s.tv_show_id != 0].groupby("tv_show_id")["duration"].sum()
    return s.reindex(ctx.shows).fillna(0).to_numpy(np.float64)


A_h, A_t = airtime(TEST["hist"]), airtime(TEST["target"])
H = M.history(ctx)
R = H / (A_h + SMOOTH)[None, :] * A_t[None, :]
MF = M.mf(ctx, k=MF_K)
P, conv = M.ppr(ctx, alpha=PPR_ALPHA)
print("ppr alpha %.2f: %d iterations, residual %.1e, converged %s" % (PPR_ALPHA, *conv))
# A user with fewer than five scored on-air shows would get arbitrary fillers (ties
# broken by id). A popularity term too small to reorder any scored show fills them.
POP = M.popularity(ctx)
POP_TIE = 1e-9 * M.row_norm(POP)
short = int(((H * ctx.air_mask[None, :]) > 0).sum(axis=1).__lt__(K).sum())
print("users with fewer than %d watched on-air shows: %d" % (K, short))
methods = {
    "history": H + POP_TIE,
    "popularity": POP,
    "cbf": M.cbf(ctx),
    "mf": MF,
    "ppr": P,
    "rate": R + POP_TIE,
    "rate_mf": M.blend(R, MF, LAM_MF),
}
lists = {name: ctx.topk(S) for name, S in methods.items()}

out_dir = ROOT / "submissions"
out_dir.mkdir(exist_ok=True)
for name in ("history", "rate", "rate_mf"):
    rows = []
    for u in users:
        top = lists[name][u]
        assert len(top) == K and len(set(top)) == K and all(s in air for s in top), (name, u, top)
        rows.append("%s,%s" % (u, " ".join(str(s) for s in top)))
    path = out_dir / ("sub_%s.csv" % name)
    path.write_text("user_id,tv_show_id\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print("wrote", path.name, len(rows), "rows")

json.dump({"settings": {"smooth": SMOOTH, "lam_mf": LAM_MF, "mf_k": MF_K, "ppr_alpha": PPR_ALPHA,
                        "ppr_convergence": conv},
           "lists": {name: {u: l for u, l in d.items()} for name, d in lists.items()}},
          open(ROOT / "results" / "test_lists.json", "w"))
print("done in %.0f s" % (time.time() - t0))

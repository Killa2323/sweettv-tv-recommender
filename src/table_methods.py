# -*- coding: utf-8 -*-
"""One table for the report: every method and the final blend on both validation
folds, with the settings fixed by tune.py. Writes results/methods_table.json.
Usage: python table_methods.py
"""
import json
import numpy as np
from common import load, weeks, on_air, true_top, mapk, ROOT
import models as M

FOLDS = {"A": {"hist": (11, 18), "target": (19, 30)}, "B": {"hist": (11, 22), "target": (23, 30)}}
SMOOTH, MF_K, LAM_MF, ALPHA = 1e5, 32, 0.1, 0.3

inter, shows, sched, sub = load()
users = list(sub.user_id)
out = {"settings": {"smooth": SMOOTH, "mf_k": MF_K, "lam_mf": LAM_MF, "ppr_alpha": ALPHA}, "folds": FOLDS, "map": {}}
for fname, split in FOLDS.items():
    truth = true_top(inter, split["target"])
    ctx = M.Context(inter, shows, split["hist"], on_air(sched, split["target"]), users)

    def airtime(span):
        s = weeks(sched, span)
        return s[s.tv_show_id != 0].groupby("tv_show_id").duration.sum().reindex(ctx.shows).fillna(0).to_numpy()

    H = M.history(ctx)
    R = H / (airtime(split["hist"]) + SMOOTH)[None, :] * airtime(split["target"])[None, :]
    MF = M.mf(ctx, k=MF_K)
    P, conv = M.ppr(ctx, alpha=ALPHA)
    scores = {"popularity": M.popularity(ctx), "history": H, "cbf": M.cbf(ctx), "mf": MF, "ppr": P,
              "rate": R, "rate_mf": M.blend(R, MF, LAM_MF)}
    out["map"][fname] = {k: mapk(truth, ctx.topk(v), users) for k, v in scores.items()}
    out["map"][fname]["ppr_convergence"] = conv
    out["map"][fname]["users_with_truth"] = sum(u in truth for u in users)
    print(fname, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["map"][fname].items()})
json.dump(out, open(ROOT / "results" / "methods_table.json", "w"), indent=1)

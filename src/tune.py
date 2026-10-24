# -*- coding: utf-8 -*-
"""Final tuning on two validation folds, so a setting is kept only if it helps on both:
  fold A  history weeks 11-18, future 19-30 (12 weeks, as in the test)
  fold B  history weeks 11-22, future 23-30 (longer history, closer to the 20 test weeks)
Grid: rate smoothing, MF rank and weight, PPR weight, popularity weight.
Score = rate share + lam_mf * MF + lam_ppr * PPR + lam_pop * popularity (each scaled to [0, 1]).
Writes results/tuning.json with every combination's MAP@5 on both folds.
Usage: python tune.py
"""
import itertools, json, time
import numpy as np
from common import load, weeks, on_air, true_top, mapk, ROOT
import models as M

FOLDS = {"A": {"hist": (11, 18), "target": (19, 30)}, "B": {"hist": (11, 22), "target": (23, 30)}}
SMOOTHS = (3e4, 1e5, 3e5)
MF_KS = (16, 32, 64)
LAM_MF = (0.0, 0.05, 0.1, 0.2)
LAM_PPR = (0.0, 0.05, 0.1, 0.2)
LAM_POP = (0.0, 0.01)


def scaled(a):
    return M.row_norm(a - a.min(axis=1, keepdims=True))


t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
grid = {}
singles = {}
for fname, split in FOLDS.items():
    truth = true_top(inter, split["target"])
    air = on_air(sched, split["target"])
    ctx = M.Context(inter, shows, split["hist"], air, users)

    def airtime(span):
        s = weeks(sched, span)
        s = s[s.tv_show_id != 0].groupby("tv_show_id")["duration"].sum()
        return s.reindex(ctx.shows).fillna(0).to_numpy(np.float64)

    A_h, A_t = airtime(split["hist"]), airtime(split["target"])
    H = M.history(ctx)
    MF = {k: scaled(M.mf(ctx, k=k)) for k in MF_KS}
    P, conv = M.ppr(ctx, alpha=0.3)
    P = scaled(P)
    POP = scaled(M.popularity(ctx) * (A_t > 0)[None, :])
    singles[fname] = {"history": mapk(truth, ctx.topk(H), users), "ppr a0.3": mapk(truth, ctx.topk(P), users),
                      "ppr_convergence": conv}
    for k in MF_KS:
        singles[fname]["mf k=%d" % k] = mapk(truth, ctx.topk(MF[k]), users)
    for sm in SMOOTHS:
        R = H / (A_h + sm)[None, :] * A_t[None, :]
        tot = R.sum(axis=1, keepdims=True)
        Rs = np.where(tot > 0, R / np.where(tot > 0, tot, 1), 0)  # share, as in models.blend
        singles[fname]["rate %g" % sm] = mapk(truth, ctx.topk(Rs), users)
        for k, lm, lp, lpop in itertools.product(MF_KS, LAM_MF, LAM_PPR, LAM_POP):
            if lm == 0 and k != MF_KS[0]:
                continue  # MF rank is irrelevant when MF is off
            S = Rs + lm * MF[k] + lp * P + lpop * POP
            key = "smooth=%g mf_k=%d lam_mf=%.2f lam_ppr=%.2f lam_pop=%.2f" % (sm, k, lm, lp, lpop)
            grid.setdefault(key, {})[fname] = mapk(truth, ctx.topk(S), users)
    print("fold %s done: %s  (%.0f s)" % (fname, {k: (round(v, 4) if isinstance(v, float) else v)
                                                for k, v in singles[fname].items()}, time.time() - t0))

ranked = sorted(grid.items(), key=lambda kv: -(kv[1]["A"] + kv[1]["B"]))
for key, v in ranked[:10]:
    print("%-62s A %.4f  B %.4f  mean %.4f" % (key, v["A"], v["B"], (v["A"] + v["B"]) / 2))
json.dump({"folds": FOLDS, "singles": singles, "grid": grid, "best": ranked[0][0]},
          open(ROOT / "results" / "tuning.json", "w"), indent=1)
print("best:", ranked[0][0], "| done in %.0f s" % (time.time() - t0))

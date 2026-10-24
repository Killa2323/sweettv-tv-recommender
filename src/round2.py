# -*- coding: utf-8 -*-
"""Round 2: validation with the organisers' definition of a watched show.

A show is watched when over 80% of an airing is seen in one view without switching
channel (the rule reported by N. Melentev, 2nd place, Towards Data Science 2021). A
user's true Top-5 = shows with the most such airings in the future window (ties:
more watch seconds, then smaller id). First check: the submitted model (seconds-based
schedule-aware history + 0.1 MF, Kaggle public 0.31182) should score near that value
under this truth if the definition is right.

Count-based scores, all restricted to shows on air in the predicted window:
  hist_nXX      airings seen over XX% in the history (the user keeps watching)
  crate_nXX     the same per airing in the history x airings in the predicted window
  + MF          PureSVD on log(1 + n50) counts, blended with weight lam
Writes results/round2.json. Usage: python round2.py
"""
import itertools, json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import load, weeks, on_air, mapk, ROOT, K
import models as M

FOLDS = {"A": {"hist": (11, 18), "target": (19, 30)}, "B": {"hist": (11, 22), "target": (23, 30)}}
t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)


def true_top80(span, k=K):
    t = weeks(inter, span).groupby(["user_id", "tv_show_id"], as_index=False)[["n80", "seconds"]].sum()
    t = t[t.n80 > 0].sort_values(["user_id", "n80", "seconds", "tv_show_id"], ascending=[True, False, False, True])
    return t.groupby("user_id")["tv_show_id"].apply(lambda s: list(s.head(k))).to_dict()


def scaled(a):
    return M.row_norm(a - a.min(axis=1, keepdims=True))


out = {"rule": "true Top-5 by airings seen over 80% in one view", "folds": FOLDS, "map": {}, "grid": {}}
for fname, split in FOLDS.items():
    truth = true_top80(split["target"])
    ctx = M.Context(inter, shows, split["hist"], on_air(sched, split["target"]), users)
    h = weeks(inter, split["hist"]).groupby(["user_id", "tv_show_id"], as_index=False)[["n30", "n50", "n80"]].sum()
    r = h.user_id.map(ctx.uix).to_numpy(); c = h.tv_show_id.map(ctx.six).to_numpy()
    C = {col: sp.csr_matrix((h[col].to_numpy(np.float64), (r, c)), shape=ctx.X.shape) for col in ("n30", "n50", "n80")}

    def airings(span):
        s = weeks(sched, span)
        return s[s.tv_show_id != 0].groupby("tv_show_id").size().reindex(ctx.shows).fillna(0).to_numpy(np.float64)

    def seconds_air(span):
        s = weeks(sched, span)
        return s[s.tv_show_id != 0].groupby("tv_show_id").duration.sum().reindex(ctx.shows).fillna(0).to_numpy(np.float64)

    N_h, N_t = airings(split["hist"]), airings(split["target"])
    pop80 = np.asarray(C["n80"].sum(axis=0)).ravel() * (N_t > 0)
    TIE = 1e-9 * (pop80 / max(pop80.max(), 1))[None, :]
    res = {"users_with_truth": sum(u in truth for u in users)}

    # 0. the submitted model under this truth (sanity check against Kaggle 0.31182)
    H = M.history(ctx)
    R = H / (seconds_air(split["hist"]) + 1e5)[None, :] * seconds_air(split["target"])[None, :]
    MFs = M.mf(ctx, k=32)
    res["submitted (seconds rate + 0.1 MF)"] = mapk(truth, ctx.topk(M.blend(R, MFs, 0.1)), users)
    res["popularity n80"] = mapk(truth, ctx.topk(np.tile(pop80, (len(users), 1))), users)
    hist = {}
    for col in ("n30", "n50", "n80"):
        Hc = C[col][ctx.trows].toarray()
        hist[col] = Hc
        res["hist_" + col] = mapk(truth, ctx.topk(Hc + TIE), users)
    crates = {}
    for col, sm in itertools.product(("n30", "n50", "n80"), (1.0, 3.0, 10.0, 30.0, 100.0)):
        Rc = hist[col] / (N_h + sm)[None, :] * N_t[None, :]
        crates[(col, sm)] = Rc
        res["crate_%s s=%g" % (col, sm)] = mapk(truth, ctx.topk(Rc + TIE), users)
    L = C["n50"].copy(); L.data = np.log1p(L.data)
    U, S_, Vt = svds(L, k=32, v0=np.random.default_rng(0).standard_normal(min(L.shape)))
    MFc = scaled((U[ctx.trows] * S_) @ Vt)
    res["mf counts k32"] = mapk(truth, ctx.topk(MFc), users)
    out["map"][fname] = res
    print("fold %s (%.0f s):" % (fname, time.time() - t0))
    for k_, v in res.items():
        print("   %-40s %s" % (k_, ("%.4f" % v) if isinstance(v, float) else v))

    # blends of the best count scores with count MF and popularity
    POPN = M.row_norm(np.tile(pop80, (len(users), 1)))
    for base_key in ("hist_n50", "crate_n50 s=10", "crate_n50 s=30", "crate_n50 s=100", "crate_n30 s=30", "crate_n30 s=100"):
        if base_key.startswith("hist_"):
            base = hist[base_key[5:]]
        else:
            col, sm = base_key[6:].split(" s=")
            base = crates[(col, float(sm))]
        tot = base.sum(axis=1, keepdims=True)
        share = np.where(tot > 0, base / np.where(tot > 0, tot, 1), 0)
        for lam, lpop in itertools.product((0.0, 0.05, 0.1, 0.2), (0.0, 0.01, 0.03, 0.05, 0.1)):
            S = share + lam * MFc + lpop * POPN + TIE
            out["grid"].setdefault("%s | mf %.2f | pop %.2f" % (base_key, lam, lpop), {})[fname] = mapk(truth, ctx.topk(S), users)

ranked = sorted(out["grid"].items(), key=lambda kv: -(kv[1]["A"] + kv[1]["B"]))
print("\nbest blends (mean of folds):")
for k_, v in ranked[:8]:
    print("   %-45s A %.4f  B %.4f  mean %.4f" % (k_, v["A"], v["B"], (v["A"] + v["B"]) / 2))
out["best"] = ranked[0][0]
json.dump(out, open(ROOT / "results" / "round2.json", "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))

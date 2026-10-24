# -*- coding: utf-8 -*-
"""Round 2 submission: the count-based blend chosen in round2.py on both folds.

score = share of [airings seen over 50% in one view in weeks 11-30
                  / (airings broadcast in weeks 11-30 + 30) x airings in weeks 31-42]
        + 0.05 x PureSVD(rank 32) on log(1 + n50), scaled to [0, 1]
        + 0.05 x popularity (airings seen over 80%, all users), scaled to [0, 1]
Only shows on air in weeks 31-42 are candidates.
Outputs submissions/sub_counts.csv and results/test_lists_round2.json.
Usage: python predict2.py
"""
import json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import load, weeks, on_air, TEST, ROOT, K
import models as M

COL, SMOOTH, LAM_MF, LAM_POP, MF_K = "n50", 30.0, 0.05, 0.05, 32   # results/round2.json

t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
air = on_air(sched, TEST["target"])
ctx = M.Context(inter, shows, TEST["hist"], air, users)
h = weeks(inter, TEST["hist"]).groupby(["user_id", "tv_show_id"], as_index=False)[["n50", "n80"]].sum()
r = h.user_id.map(ctx.uix).to_numpy(); c = h.tv_show_id.map(ctx.six).to_numpy()
C50 = sp.csr_matrix((h.n50.to_numpy(np.float64), (r, c)), shape=ctx.X.shape)
C80 = sp.csr_matrix((h.n80.to_numpy(np.float64), (r, c)), shape=ctx.X.shape)


def airings(span):
    s = weeks(sched, span)
    return s[s.tv_show_id != 0].groupby("tv_show_id").size().reindex(ctx.shows).fillna(0).to_numpy(np.float64)


N_h, N_t = airings(TEST["hist"]), airings(TEST["target"])
base = C50[ctx.trows].toarray() / (N_h + SMOOTH)[None, :] * N_t[None, :]
tot = base.sum(axis=1, keepdims=True)
share = np.where(tot > 0, base / np.where(tot > 0, tot, 1), 0)
L = C50.copy(); L.data = np.log1p(L.data)
U, S_, Vt = svds(L, k=MF_K, v0=np.random.default_rng(0).standard_normal(min(L.shape)))
mf = (U[ctx.trows] * S_) @ Vt
mf = M.row_norm(mf - mf.min(axis=1, keepdims=True))
pop80 = np.asarray(C80.sum(axis=0)).ravel() * (N_t > 0)
popn = M.row_norm(np.tile(pop80, (len(users), 1)))
tie = 1e-9 * (pop80 / max(pop80.max(), 1))[None, :]
score = share + LAM_MF * mf + LAM_POP * popn + tie
lists = ctx.topk(score)
print("users with no show seen over 50%% in weeks 11-30 that is on air later: %d" % int((tot.ravel() == 0).sum()))

rows = []
for u in users:
    top = lists[u]
    assert len(top) == K and len(set(top)) == K and all(s in air for s in top), (u, top)
    rows.append("%s,%s" % (u, " ".join(str(s) for s in top)))
path = ROOT / "submissions" / "sub_counts.csv"
path.write_text("user_id,tv_show_id\n" + "\n".join(rows) + "\n", encoding="utf-8")
print("wrote", path.name, len(rows), "rows")
old = json.load(open(ROOT / "results" / "test_lists.json"))["lists"]["rate_mf"]
same = sum(len(set(lists[u]) & set(old[u])) for u in users) / (K * len(users))
print("share of slots shared with the first submission: %.3f" % same)
json.dump({"settings": {"col": COL, "smooth": SMOOTH, "lam_mf": LAM_MF, "lam_pop": LAM_POP, "mf_k": MF_K},
           "lists": {"counts": lists}}, open(ROOT / "results" / "test_lists_round2.json", "w"))
print("done in %.0f s" % (time.time() - t0))

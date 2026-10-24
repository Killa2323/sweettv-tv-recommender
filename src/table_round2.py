# -*- coding: utf-8 -*-
"""Report numbers for the final (round 2) method, all under the organisers' rule:
a user's true Top-5 = shows with the most airings seen over 80% in one view.

For both validation folds: MAP@5 of every method, all built on the same count signal
(airings seen over 50% in one view, log(1 + n) as edge or matrix weight):
  popularity, CBF, MF (PureSVD), PPR, own history, schedule-aware counts, final blend,
  and the first submission (seconds-based) for reference.
Fold A also gives the familiar / new split of correct items. For the test window it
gives every method's Top-5 for the two report users and checks that the final lists
equal the submitted file's lists.
Writes results/round2_report.json. Usage: python table_round2.py
"""
import copy, json, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import load, weeks, on_air, apk, mapk, TEST, ROOT, K
import models as M

FOLDS = {"A": {"hist": (11, 18), "target": (19, 30)}, "B": {"hist": (11, 22), "target": (23, 30)}}
SMOOTH, LAM_MF, LAM_POP, MF_K, ALPHA = 30.0, 0.05, 0.05, 32, 0.3   # as predict2.py

t0 = time.time()
inter, shows, sched, sub = load()
users = list(sub.user_id)
two = json.load(open(ROOT / "results" / "two_users.json", encoding="utf-8"))
report_users = {lab: two["users"][lab]["user_id"] for lab in ("A", "B")}


def true_top80(span, k=K):
    t = weeks(inter, span).groupby(["user_id", "tv_show_id"], as_index=False)[["n80", "seconds"]].sum()
    t = t[t.n80 > 0].sort_values(["user_id", "n80", "seconds", "tv_show_id"], ascending=[True, False, False, True])
    return t.groupby("user_id")["tv_show_id"].apply(lambda s: list(s.head(k))).to_dict()


def scaled(a):
    return M.row_norm(a - a.min(axis=1, keepdims=True))


def all_scores(hist_span, target_span):
    """Every method's score matrix for the submission users."""
    ctx = M.Context(inter, shows, hist_span, on_air(sched, target_span), users)
    h = weeks(inter, hist_span).groupby(["user_id", "tv_show_id"], as_index=False)[["n50", "n80"]].sum()
    r = h.user_id.map(ctx.uix).to_numpy(); c = h.tv_show_id.map(ctx.six).to_numpy()
    C50 = sp.csr_matrix((h.n50.to_numpy(np.float64), (r, c)), shape=ctx.X.shape)
    C80 = sp.csr_matrix((h.n80.to_numpy(np.float64), (r, c)), shape=ctx.X.shape)

    def count(span, what):
        s = weeks(sched, span)
        s = s[s.tv_show_id != 0].groupby("tv_show_id")
        s = s.size() if what == "airings" else s.duration.sum()
        return s.reindex(ctx.shows).fillna(0).to_numpy(np.float64)

    N_h, N_t = count(hist_span, "airings"), count(target_span, "airings")
    pop80 = np.asarray(C80.sum(axis=0)).ravel() * (N_t > 0)
    tie = 1e-9 * (pop80 / max(pop80.max(), 1))[None, :]
    cctx = copy.copy(ctx)                      # same index, count signal as weights
    cctx.L = C50.copy(); cctx.L.data = np.log1p(cctx.L.data)
    U, S_, Vt = svds(cctx.L, k=MF_K, v0=np.random.default_rng(0).standard_normal(min(cctx.L.shape)))
    mf = scaled((U[ctx.trows] * S_) @ Vt)
    hist = C50[ctx.trows].toarray()
    crate = hist / (N_h + SMOOTH)[None, :] * N_t[None, :]
    tot = crate.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, crate / np.where(tot > 0, tot, 1), 0)
    popn = M.row_norm(np.tile(pop80, (len(users), 1)))
    ppr, conv = M.ppr(cctx, alpha=ALPHA)
    H = M.history(ctx)                          # seconds, for the first submission
    R = H / (count(hist_span, "seconds") + 1e5)[None, :] * count(target_span, "seconds")[None, :]
    scores = {"popularity": np.tile(pop80, (len(users), 1)), "cbf": M.cbf(cctx), "mf": mf, "ppr": ppr,
              "history": hist + tie, "schedule": crate + tie,
              "final": share + LAM_MF * mf + LAM_POP * popn + tie,
              "first submission": M.blend(R, M.mf(ctx, k=32), 0.1)}
    seen = {u: set(ctx.shows[ctx.X[ctx.uix[u]].indices]) for u in users}
    return ctx, scores, seen, conv


out = {"settings": {"smooth": SMOOTH, "lam_mf": LAM_MF, "lam_pop": LAM_POP, "mf_k": MF_K, "ppr_alpha": ALPHA},
       "map": {}, "discovery": {}, "users": {lab: {"user_id": u, "ap": {}, "test": {}} for lab, u in report_users.items()}}
for fname, split in FOLDS.items():
    truth = true_top80(split["target"])
    ctx, scores, seen, conv = all_scores(split["hist"], split["target"])
    lists = {k: ctx.topk(v) for k, v in scores.items()}
    out["map"][fname] = {k: mapk(truth, l, users) for k, l in lists.items()}
    out["map"][fname]["ppr_convergence"] = conv
    print("fold %s:" % fname, {k: round(v, 4) for k, v in out["map"][fname].items() if isinstance(v, float)},
          "(%.0f s)" % (time.time() - t0))
    if fname == "A":
        for k, l in lists.items():
            hs = sum(1 for u in users for s in l[u] if s in truth.get(u, []) and s in seen[u])
            hn = sum(1 for u in users for s in l[u] if s in truth.get(u, []) and s not in seen[u])
            sn = sum(1 for u in users for s in l[u] if s not in seen[u])
            out["discovery"][k] = {"hits_seen": hs, "hits_new": hn, "slots_new": sn}
        for lab, u in report_users.items():
            out["users"][lab]["ap"] = {k: apk(truth.get(u, []), l[u]) for k, l in lists.items()}
            out["users"][lab]["truth_A"] = [int(s) for s in truth.get(u, [])]

ctx, scores, seen, conv = all_scores(TEST["hist"], TEST["target"])
lists = {k: ctx.topk(v) for k, v in scores.items()}
submitted = json.load(open(ROOT / "results" / "test_lists_round2.json"))["lists"]["counts"]
assert all(lists["final"][u] == submitted[u] for u in users), "final lists differ from sub_counts.csv"
out["test_unseen_share"] = {k: float(np.mean([s not in seen[u] for u in users for s in l[u]])) for k, l in lists.items()}
for lab, u in report_users.items():
    out["users"][lab]["test"] = {k: [int(s) for s in l[u]] for k, l in lists.items()}
json.dump(out, open(ROOT / "results" / "round2_report.json", "w"), indent=1)
print("discovery fold A:", out["discovery"])
print("final test lists equal sub_counts.csv: yes | done in %.0f s" % (time.time() - t0))

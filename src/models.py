# -*- coding: utf-8 -*-
"""The recommendation methods, all scored on one user x show matrix built from a
history window. Each method returns a dense score array [target users, shows];
ranking keeps only shows on air in the predicted window.

  history     the user's own watch seconds (reference point, not one of the four)
  popularity  total watch seconds over all users
  cbf         content-based: user profile over show category, genres and channel
  mf          matrix factorisation: truncated SVD of log watch minutes (PureSVD)
  ppr         personalised PageRank on the weighted user-show graph
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import weeks, K


class Context:
    """Index maps and the history matrix for one split."""

    def __init__(self, inter, shows, hist_span, air, target_users):
        h = weeks(inter, hist_span)
        h = h.groupby(["user_id", "tv_show_id"], as_index=False)["seconds"].sum()
        self.users = sorted(set(h.user_id) | set(target_users))
        self.shows = np.array(sorted(set(h.tv_show_id) | set(air)), dtype=np.int64)
        self.uix = {u: i for i, u in enumerate(self.users)}
        self.six = {s: j for j, s in enumerate(self.shows)}
        rows = h.user_id.map(self.uix).to_numpy()
        cols = h.tv_show_id.map(self.six).to_numpy()
        self.X = sp.csr_matrix((h.seconds.to_numpy(np.float64), (rows, cols)),
                               shape=(len(self.users), len(self.shows)))
        self.L = self.X.copy()
        self.L.data = np.log1p(self.L.data / 60.0)  # log watch minutes
        self.target = list(target_users)
        self.trows = np.array([self.uix[u] for u in self.target])
        self.air_mask = np.isin(self.shows, list(air))
        self.meta = shows.reindex(self.shows)

    def topk(self, scores, k=K):
        """user_id -> k on-air shows with the highest score (ties: smaller id)."""
        s = np.where(self.air_mask[None, :], scores, -np.inf)
        out = {}
        for r, u in enumerate(self.target):
            row = s[r]
            idx = np.lexsort((self.shows, -row))[:k]
            out[u] = [int(self.shows[j]) for j in idx if np.isfinite(row[j])]
        return out


def history(ctx):
    return ctx.X[ctx.trows].toarray()


def popularity(ctx):
    pop = np.asarray(ctx.X.sum(axis=0)).ravel()
    return np.tile(pop, (len(ctx.trows), 1))


def cbf(ctx):
    """Profile = watch-weighted sum of show attribute vectors; score = cosine."""
    m = ctx.meta
    feats = []
    for col, prefix in (("tv_show_category", "cat"), ("tv_show_genre_1", "g"), ("tv_show_genre_2", "g"),
                        ("tv_show_genre_3", "g"), ("channel_id", "ch")):
        vals = m[col].astype("string").fillna("")
        feats.append(pd.DataFrame({"j": np.arange(len(m)), "f": prefix + ":" + vals}).query("f != @prefix + ':'"))
    f = pd.concat(feats).drop_duplicates()
    fid = {x: i for i, x in enumerate(sorted(f.f.unique()))}
    F = sp.csr_matrix((np.ones(len(f)), (f.j.to_numpy(), f.f.map(fid).to_numpy())), shape=(len(m), len(fid)))
    df = np.asarray((F > 0).sum(axis=0)).ravel()
    F = F @ sp.diags(np.log((1 + len(m)) / (1 + df)) + 1)  # idf weighting
    norms = np.sqrt(np.asarray(F.multiply(F).sum(axis=1)).ravel())
    F = sp.diags(1 / np.maximum(norms, 1e-12)) @ F
    prof = ctx.L[ctx.trows] @ F
    pn = np.sqrt(np.asarray(prof.multiply(prof).sum(axis=1)).ravel())
    prof = sp.diags(1 / np.maximum(pn, 1e-12)) @ prof
    return (prof @ F.T).toarray()


def mf(ctx, k=32, seed=0):
    """PureSVD: rank-k truncated SVD of the log-minutes matrix; score = U S V^T."""
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(ctx.L.shape))
    U, S, Vt = svds(ctx.L, k=k, v0=v0)
    return (U[ctx.trows] * S) @ Vt


def ppr(ctx, alpha=0.5, tol=1e-8, max_iter=500):
    """Personalised PageRank on the bipartite graph with edge weight log minutes,
    restart at each target user, update p' = a (Q^T p + d v) + (1 - a) v.
    Returns show-node scores and (iterations, residual, converged)."""
    W = ctx.L
    du = np.asarray(W.sum(axis=1)).ravel()
    ds = np.asarray(W.sum(axis=0)).ravel()
    Pu = (sp.diags(np.where(du > 0, 1 / np.maximum(du, 1e-300), 0)) @ W).tocsr()        # user -> show
    Ps = (sp.diags(np.where(ds > 0, 1 / np.maximum(ds, 1e-300), 0)) @ W.T).tocsr()      # show -> user
    PuT, PsT = Pu.T.tocsr(), Ps.T.tocsr()
    nU, nS = W.shape
    B = len(ctx.trows)
    vu = np.zeros((nU, B))
    vu[ctx.trows, np.arange(B)] = 1.0
    pu, ps = vu.copy(), np.zeros((nS, B))
    zu, zs = (du == 0), (ds == 0)
    it, res = 0, np.inf
    while it < max_iter:
        d = pu[zu].sum(axis=0) + ps[zs].sum(axis=0)          # mass on zero-degree nodes
        nu = alpha * (PsT @ ps + vu * d) + (1 - alpha) * vu   # into users: from shows
        ns = alpha * (PuT @ pu)                                # into shows: from users
        res = (np.abs(nu - pu).sum(axis=0) + np.abs(ns - ps).sum(axis=0)).max()
        pu, ps, it = nu, ns, it + 1
        if res < tol:
            break
    return ps.T, (it, float(res), bool(res < tol))


def row_norm(a):
    m = a.max(axis=1, keepdims=True)
    return np.where(m > 0, a / np.where(m > 0, m, 1), 0)


def blend(hist_scores, method_scores, lam):
    """History share plus lam x the method score scaled to [0, 1] per user."""
    tot = hist_scores.sum(axis=1, keepdims=True)
    share = np.where(tot > 0, hist_scores / np.where(tot > 0, tot, 1), 0)
    m = method_scores - method_scores.min(axis=1, keepdims=True)
    return share + lam * row_norm(m)

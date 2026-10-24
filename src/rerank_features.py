# -*- coding: utf-8 -*-
"""Candidate tables for the second-layer re-ranker (rerank.py).

For a window (history weeks -> predicted weeks) every user gets candidates from the
first layer, all on air in the predicted weeks: top 30 of the final blend
(predict2.py), top 15 of PPR, top 10 of MF, top 30 by expected exposure, top 15 by
exposure with the profile of the last 4 history weeks, and the 10 most popular shows.

Expected exposure: from the raw views, the share of time the user is tuned to each
channel in each hour of the day (weekdays and weekends apart) during the history
weeks. A show's exposure is that share summed over the show's airings (channel and
hour of the airing's middle), so a night rerun on a channel the user never watches at
night adds almost nothing. It is computed for the history weeks and for the
predicted weeks (with the history profile, no future viewing).

Features are per-week rates, ranks and shares, so windows of different length are
comparable. gain = 5 - position of the show in the user's true Top-5 of the predicted
weeks (organisers' rule: airings seen over 80% in one view), 0 if absent.
Windows: A 11-18 -> 19-30, B 11-22 -> 23-30, E 11-16 -> 17-22 (all users) and
T 11-30 -> 31-42 (submission users, no labels).
Writes data/processed/view_profile.parquet (once) and data/processed/rerank/{A,B,E,T}.parquet.
Usage: python rerank_features.py
"""
import copy, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from common import load, weeks, on_air, mapk, DATA, PROC, ROOT, K
import models as M

SMOOTH, LAM_MF, LAM_POP, MF_K, ALPHA = 30.0, 0.05, 0.05, 32, 0.3   # as predict2.py
NCAND = {"final": 30, "ppr": 15, "mf": 10, "exp": 30, "exp4": 15, "pop": 10}
WINDOWS = {"A": ((11, 18), (19, 30)), "B": ((11, 22), (23, 30)), "E": ((11, 16), (17, 22)), "T": ((11, 30), (31, 42))}
CHUNK = 600
DAY0 = (pd.Timestamp("2020-03-09") - pd.Timestamp("1970-01-01")).days   # Monday of week 11
OUT = PROC / "rerank"
OUT.mkdir(exist_ok=True)
t0 = time.time()


def slot_of(epoch_s):
    """Hour of the day, + 24 on Saturday and Sunday (1970-01-01 was a Thursday)."""
    hr = epoch_s // 3600
    return (hr % 24 + 24 * ((hr // 24 + 3) % 7 >= 5)).astype(np.int64)


def view_profile():
    """Viewing seconds per (user, channel, week, slot) from the raw views, built once."""
    path = PROC / "view_profile.parquet"
    if path.exists():
        return pd.read_parquet(path)
    v = pacsv.read_csv(DATA / "dataset11-30.csv", convert_options=pacsv.ConvertOptions(
        include_columns=["user_id", "vsetv_id", "start_time", "stop_time"],
        column_types={"user_id": pa.string(), "vsetv_id": pa.int64(), "start_time": pa.string(),
                      "stop_time": pa.string()})).to_pandas()
    secs = lambda col: pd.to_datetime(v[col], format="%Y-%m-%d %H:%M:%S").astype("datetime64[s]").astype("int64").to_numpy()
    a, b = secs("start_time"), secs("stop_time")
    ok = b > a
    uid, uniq = pd.factorize(v.user_id)
    uid, ch, a, b = uid[ok], v.vsetv_id.to_numpy()[ok], a[ok], b[ok]
    h0 = a // 3600
    n = (b - 1) // 3600 - h0 + 1                                   # clock hours each view touches
    rows = np.repeat(np.arange(len(a)), n)
    hr = h0[rows] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    sec = np.minimum(b[rows], (hr + 1) * 3600) - np.maximum(a[rows], hr * 3600)
    df = pd.DataFrame({"u": uid[rows], "channel_id": ch[rows], "week": ((hr // 24 - DAY0) // 7 + 11).astype(np.int16),
                       "slot": slot_of(hr * 3600).astype(np.int8), "seconds": sec})
    prof = df.groupby(["u", "channel_id", "week", "slot"], as_index=False).seconds.sum()
    prof.insert(0, "user_id", np.asarray(uniq)[prof.u.to_numpy()])
    prof = prof.drop(columns="u")
    prof.to_parquet(path, index=False)
    print("view profile: %d views (%d with stop <= start dropped) -> %d hour pieces -> %d rows, %.3e s (views total %.3e s)"
          % (len(a), int((~ok).sum()), len(rows), len(prof), prof.seconds.sum(), float((b - a).sum())), flush=True)
    return prof


inter, shows, _, sub = load()
SUBU = list(sub.user_id)
SUBSET = set(SUBU)
ALLU = sorted(set(inter.user_id) | SUBSET)       # 2 submission users have no relevant show at all
sched = pd.read_parquet(PROC / "schedule.parquet", columns=["channel_id", "tv_show_id", "week", "duration", "start"])
mid = sched.start.astype("datetime64[s]").astype("int64").to_numpy() + sched.duration.to_numpy() // 2
prof = view_profile()
CH = np.union1d(prof.channel_id.unique(), sched.channel_id.unique())
NC = len(CH)
sched["cslot"] = np.searchsorted(CH, sched.channel_id.to_numpy()) * 48 + slot_of(mid)
prof["cslot"] = np.searchsorted(CH, prof.channel_id.to_numpy()) * 48 + prof.slot.to_numpy().astype(np.int64)
CAT = {c: i for i, c in enumerate(sorted(shows.tv_show_category.dropna().astype(str).unique()))}
print("loaded in %.0f s: %d channels, %d categories" % (time.time() - t0, NC, len(CAT)), flush=True)


def true_top80(span, k=K):
    t = weeks(inter, span).groupby(["user_id", "tv_show_id"], as_index=False)[["n80", "seconds"]].sum()
    t = t[t.n80 > 0].sort_values(["user_id", "n80", "seconds", "tv_show_id"], ascending=[True, False, False, True])
    return t.groupby("user_id")["tv_show_id"].apply(lambda s: list(s.head(k))).to_dict()


def scaled(a):
    return M.row_norm(a - a.min(axis=1, keepdims=True))


def window(name, hist, target, users, labels=True):
    ctx = M.Context(inter, shows, hist, on_air(sched, target), users)
    nS, air = len(ctx.shows), ctx.air_mask
    wh, wt = hist[1] - hist[0] + 1, target[1] - target[0] + 1
    h = weeks(inter, hist)
    agg = h.groupby(["user_id", "tv_show_id"], as_index=False)[["seconds", "n30", "n50", "n80"]].sum()
    act = h[h.n50 > 0].groupby(["user_id", "tv_show_id"]).agg(last=("week", "max"), active=("week", "nunique")).reset_index()
    rec = weeks(h, (hist[1] - 3, hist[1])).groupby(["user_id", "tv_show_id"], as_index=False).n50.sum()

    def mat(df, col):
        return sp.csr_matrix((df[col].to_numpy(np.float64), (df.user_id.map(ctx.uix).to_numpy(),
                              df.tv_show_id.map(ctx.six).to_numpy())), shape=ctx.X.shape)

    C30, C50, C80, SEC = (mat(agg, c) for c in ("n30", "n50", "n80", "seconds"))
    LAST, ACT, REC = mat(act, "last"), mat(act, "active"), mat(rec, "n50")

    def per_show(span, what):
        s = weeks(sched, span)
        s = s[s.tv_show_id != 0].groupby("tv_show_id")
        s = s.size() if what == "airings" else s.duration.sum()
        return s.reindex(ctx.shows).fillna(0).to_numpy(np.float64)

    def airing_slots(span):                                        # show x (channel, slot) airings
        s = weeks(sched, span)
        s = s[s.tv_show_id.isin(ctx.shows)]
        return sp.csr_matrix((np.ones(len(s)), (s.tv_show_id.map(ctx.six).to_numpy(), s.cslot.to_numpy())),
                             shape=(nS, NC * 48))

    def profile(span):                                             # share of each slot spent on each channel
        p = weeks(prof, span).groupby(["user_id", "cslot"], as_index=False).seconds.sum()
        p = p[p.user_id.isin(ctx.users)]
        nw = span[1] - span[0] + 1
        days = np.where(p.cslot.to_numpy() % 48 < 24, 5 * nw, 2 * nw)
        return sp.csr_matrix((p.seconds.to_numpy() / (days * 3600.0), (p.user_id.map(ctx.uix).to_numpy(),
                             p.cslot.to_numpy())), shape=(len(ctx.users), NC * 48))

    recent = (hist[1] - 3, hist[1])                                # last 4 history weeks
    N_h, N_t, A_t = per_show(hist, "airings"), per_show(target, "airings"), per_show(target, "seconds")
    Ch, Ct, Ch4 = airing_slots(hist).T.tocsr(), airing_slots(target).T.tocsr(), airing_slots(recent).T.tocsr()
    P, P4 = profile(hist), profile(recent)
    pop80 = np.asarray(C80.sum(axis=0)).ravel() * (N_t > 0)
    popn = pop80 / pop80.max()
    tie = 1e-9 * pop80 / max(pop80.max(), 1)
    reach = np.asarray((C50 > 0).sum(axis=0)).ravel() / max(1, int((np.asarray(C50.sum(axis=1)).ravel() > 0).sum()))
    pop_top = np.argsort(-np.where(air, pop80, -np.inf), kind="stable")[:NCAND["pop"]]
    cctx = copy.copy(ctx)                                          # count signal as weights, as predict2.py
    cctx.L = C50.copy(); cctx.L.data = np.log1p(cctx.L.data)
    U, S_, Vt = svds(cctx.L, k=MF_K, v0=np.random.default_rng(0).standard_normal(min(cctx.L.shape)))
    chan, _ = pd.factorize(ctx.meta.channel_id.fillna(-1).astype(np.int64))
    G = sp.csr_matrix((np.ones(nS), (np.arange(nS), chan)))
    cat = ctx.meta.tv_show_category.astype("string").map(CAT).fillna(-1).to_numpy(np.int64)
    Hc = sp.csr_matrix((np.ones(int((cat >= 0).sum())), (np.flatnonzero(cat >= 0), cat[cat >= 0])), shape=(nS, len(CAT)))
    truth = true_top80(target) if labels else {}
    parts = []
    for start in range(0, len(users), CHUNK):
        part = users[start:start + CHUNK]
        rr = np.array([ctx.uix[u] for u in part])
        B, bi = len(rr), np.arange(len(rr))[:, None]
        pc = copy.copy(cctx); pc.trows = rr
        ppr, _ = M.ppr(pc, alpha=ALPHA)
        mf = scaled((U[rr] * S_) @ Vt)
        c50 = C50[rr].toarray()
        crate = c50 / (N_h + SMOOTH)[None, :] * N_t[None, :]
        tot = crate.sum(axis=1, keepdims=True)
        share = np.where(tot > 0, crate / np.where(tot > 0, tot, 1), 0)
        final = share + LAM_MF * mf + LAM_POP * popn[None, :] + tie[None, :]
        Et, Eh = (P[rr] @ Ct).toarray(), (P[rr] @ Ch).toarray()  # expected airings caught
        Et4, Eh4 = (P4[rr] @ Ct).toarray(), (P4[rr] @ Ch4).toarray()
        rank = {}
        for key, v in (("final", final), ("ppr", ppr), ("mf", mf), ("exp", Et), ("exp4", Et4)):
            order = np.argsort(-np.where(air, v, -np.inf), axis=1, kind="stable")   # ties: smaller id
            r = np.empty_like(order); r[bi, order] = np.arange(nS)[None, :]
            rank[key] = r
        mask = np.zeros((B, nS), bool)
        for key in ("final", "ppr", "mf", "exp", "exp4"):
            mask |= rank[key] < NCAND[key]
        mask[:, pop_top] = True
        mask &= air[None, :]
        b, j = np.nonzero(mask)
        gain = np.zeros((B, nS), np.int8)
        for k_, u in enumerate(part):
            for pos, s in enumerate(truth.get(u, [])):
                gain[k_, ctx.six[s]] = K - pos
        sec = SEC[rr].toarray(); usec = np.maximum(sec.sum(axis=1), 1)
        uch = (SEC[rr] @ G).toarray(); uch /= np.maximum(uch.sum(axis=1, keepdims=True), 1)
        ucat = (SEC[rr] @ Hc).toarray(); ucat /= np.maximum(ucat.sum(axis=1, keepdims=True), 1)
        lastw = LAST[rr].toarray()[b, j]
        c50bj, c80bj, rec50 = c50[b, j], C80[rr].toarray()[b, j], REC[rr].toarray()[b, j]
        parts.append(pd.DataFrame({
            "user_id": np.asarray(part, dtype=object)[b], "tv_show_id": ctx.shows[j], "gain": gain[b, j],
            "final": final[b, j], "final_rank": rank["final"][b, j], "share": share[b, j],
            "crate_wk": crate[b, j] / wt, "ppr": ppr[b, j], "ppr_rank": rank["ppr"][b, j],
            "mf": mf[b, j], "mf_rank": rank["mf"][b, j],
            "n30_wk": C30[rr].toarray()[b, j] / wh, "n50_wk": c50bj / wh, "n80_wk": c80bj / wh,
            "n50_recent_wk": rec50 / 4, "active_share": ACT[rr].toarray()[b, j] / wh,
            "weeks_since_last": np.where(lastw > 0, hist[1] - lastw, 99),
            "sec_share": sec[b, j] / usec[b], "hours_wk": sec[b, j] / 3600 / wh,
            "chan_share": uch[b, chan[j]], "cat_share": np.where(cat[j] >= 0, ucat[b, np.maximum(cat[j], 0)], 0),
            "exp_hist_wk": Eh[b, j] / wh, "exp_target_wk": Et[b, j] / wt, "exp_rank": rank["exp"][b, j],
            "catch": c50bj / (Eh[b, j] + 1), "exp_pred_wk": c50bj / (Eh[b, j] + 1) * Et[b, j] / wt,
            "exp_change": (Et[b, j] / wt) / (Eh[b, j] / wh + 0.01),
            "exp_pred80_wk": c80bj / (Eh[b, j] + 1) * Et[b, j] / wt,
            "exp_target_recent_wk": Et4[b, j] / wt, "exp_recent_rank": rank["exp4"][b, j],
            "exp_pred_recent_wk": rec50 / (Eh4[b, j] + 1) * Et4[b, j] / wt,
            "pop80_wk": pop80[j] / wh, "reach50": reach[j], "air_hist_wk": N_h[j] / wh, "air_target_wk": N_t[j] / wt,
            "air_change": (N_t[j] / wt) / (N_h[j] / wh + 0.1), "airhours_target_wk": A_t[j] / 3600 / wt,
            "category": cat[j], "user_shows": (c50 > 0).sum(axis=1)[b], "user_hours_wk": usec[b] / 3600 / wh}))
        print("   %s users %d-%d (%.0f s)" % (name, start, start + B, time.time() - t0), flush=True)
    d = pd.concat(parts, ignore_index=True)
    d.insert(1, "group", np.where(d.user_id.isin(SUBSET), "sub", "other"))
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)
    d.to_parquet(OUT / ("%s.parquet" % name), index=False)
    # first-layer Top-5 = the five best final ranks; recall and ceiling of the candidate lists
    first = d[d.final_rank < K].sort_values(["user_id", "final_rank"]).groupby("user_id").tv_show_id.apply(list).to_dict()
    msg = "%s: %d rows, %.1f candidates per user" % (name, len(d), len(d) / len(users))
    if labels:
        su = [u for u in users if u in SUBSET]
        cand = d.groupby("user_id").tv_show_id.apply(set).to_dict()
        rec_ = [len(set(truth[u]) & cand[u]) / min(K, len(truth[u])) for u in su if u in truth]
        best = {u: [s for s in truth.get(u, []) if s in cand.get(u, set())] for u in su}
        msg += " | submission users: first layer MAP@5 %.4f, candidate recall %.3f, ceiling MAP@5 %.4f" % (
            mapk(truth, first, su), float(np.mean(rec_)), mapk(truth, best, su))
        ex = d[d.group == "sub"]
        msg += " | exposure per week, truth vs other candidates: %.2f vs %.2f" % (
            ex.loc[ex.gain > 0, "exp_target_wk"].mean(), ex.loc[ex.gain == 0, "exp_target_wk"].mean())
    print(msg, "(%.0f s)" % (time.time() - t0), flush=True)
    return first


for name, (hist, target) in WINDOWS.items():
    first = window(name, hist, target, SUBU if name == "T" else ALLU, labels=(name != "T"))
    if name == "T":
        submitted = pd.read_csv(ROOT / "submissions" / "sub_counts.csv", dtype=str)
        same = all(first[u] == [int(x) for x in s.split()] for u, s in zip(submitted.user_id, submitted.tv_show_id))
        print("T: first-layer lists equal sub_counts.csv:", same)
        assert same
print("done in %.0f s" % (time.time() - t0))

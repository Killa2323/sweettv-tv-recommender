# -*- coding: utf-8 -*-
"""Second layer: LightGBM re-ranks the candidates built by rerank_features.py (the
idea of a second-layer ranker is from N. Melentev's write-up; the code is ours).

Two checks, each against the first-layer final blend on the same users and weeks:
  E1 user split: train on users NOT in submission.csv, window A (11-18 -> 19-30);
     evaluate on the 1,260 submission users, window A.
  E2 time split: train on all users, window E (11-16 -> 17-22); evaluate on the
     submission users, window B (11-22 -> 23-30), weeks after the training weeks.
Objectives: binary (is the show in the true Top-5) and LambdaRank (graded by the
position in the true Top-5). Each model is the mean of three seeds, each with its own
10% of training users held out for early stopping. A model may write the test file only if it beats the first layer in BOTH
checks; it is then trained on windows A and B (all users) and applied to window T.
Writes results/rerank.json and, if a model qualifies, submissions/sub_rerank.csv.
Usage: python rerank.py
"""
import json, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import load, weeks, on_air, apk, mapk, ROOT, PROC, K

FEATS = ["final", "final_rank", "share", "crate_wk", "ppr", "ppr_rank", "mf", "mf_rank", "n30_wk", "n50_wk",
         "n80_wk", "n50_recent_wk", "active_share", "weeks_since_last", "sec_share", "hours_wk", "chan_share",
         "cat_share", "exp_hist_wk", "exp_target_wk", "exp_rank", "catch", "exp_pred_wk", "exp_change", "pop80_wk",
         "reach50", "air_hist_wk", "air_target_wk", "air_change", "airhours_target_wk", "category", "user_shows",
         "user_hours_wk", "exp_pred80_wk", "exp_target_recent_wk", "exp_recent_rank", "exp_pred_recent_wk"]
SEEDS = (0, 1, 2)                     # each seed: its own held-out users and row sampling; scores averaged
PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, n_estimators=3000, random_state=0, deterministic=True,
              force_row_wise=True, verbose=-1)
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")
t0 = time.time()
inter, shows, sched, sub = load()
SUBU = list(sub.user_id)
D = {n: pd.read_parquet(PROC / "rerank" / ("%s.parquet" % n)) for n in "ABET"}


def true_top80(span, k=K):
    t = weeks(inter, span).groupby(["user_id", "tv_show_id"], as_index=False)[["n80", "seconds"]].sum()
    t = t[t.n80 > 0].sort_values(["user_id", "n80", "seconds", "tv_show_id"], ascending=[True, False, False, True])
    return t.groupby("user_id")["tv_show_id"].apply(lambda s: list(s.head(k))).to_dict()


def top5(df, score):
    d = df[["user_id", "tv_show_id", "final_rank"]].assign(p=score)
    d = d.sort_values(["user_id", "p", "final_rank"], ascending=[True, False, True])
    return d.groupby("user_id").tv_show_id.apply(lambda s: [int(x) for x in s.head(K)]).to_dict()


def fit_one(df, objective, seed):
    users = np.sort(df.user_id.unique())
    held = set(np.random.default_rng(seed).choice(users, size=len(users) // 10, replace=False))
    df = df.sort_values("user_id", kind="stable")
    va = df.user_id.isin(held).to_numpy()
    tr, ev = df[~va], df[va]
    stop = [lgb.early_stopping(100, verbose=False)]
    params = dict(PARAMS, random_state=seed)
    if objective == "binary":
        m = lgb.LGBMClassifier(objective="binary", **params)
        m.fit(tr[FEATS], (tr.gain > 0).astype(int), eval_set=[(ev[FEATS], (ev.gain > 0).astype(int))],
              categorical_feature=["category"], callbacks=stop)
    else:
        size = lambda x: x.groupby("user_id", sort=False).size().to_numpy()
        m = lgb.LGBMRanker(objective="lambdarank", **params)
        m.fit(tr[FEATS], tr.gain, group=size(tr), eval_set=[(ev[FEATS], ev.gain)], eval_group=[size(ev)],
              eval_at=[K], categorical_feature=["category"], callbacks=stop)
    return m


def fit(df, objective):
    return [fit_one(df, objective, s) for s in SEEDS]


def score(models, df):
    one = lambda m: m.predict_proba(df[FEATS])[:, 1] if isinstance(m, lgb.LGBMClassifier) else m.predict(df[FEATS])
    return np.mean([one(m) for m in models], axis=0)


def first_layer(df):
    return df[df.final_rank < K].sort_values(["user_id", "final_rank"]).groupby("user_id").tv_show_id.apply(
        lambda s: [int(x) for x in s]).to_dict()


def compare(truth, base, new, users, n_boot=2000):
    a = np.array([apk(truth.get(u, []), base.get(u, [])) for u in users])
    b = np.array([apk(truth.get(u, []), new.get(u, [])) for u in users])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(users), size=(n_boot, len(users)))
    diff = (b - a)[idx].mean(axis=1)
    return {"first_layer": float(a.mean()), "rerank": float(b.mean()), "gain": float(b.mean() - a.mean()),
            "gain_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
            "users_better": int((b > a).sum()), "users_worse": int((b < a).sum())}


def discovery(test, lists, truth):
    """Correct items the user had watched in the history, correct new items, slots given to new items."""
    seen = set(zip(test.user_id[test.hours_wk > 0], test.tv_show_id[test.hours_wk > 0]))
    slots = [(u, s, s in truth.get(u, [])) for u, l in lists.items() for s in l]
    return {"hits_seen": sum(1 for u, s, hit in slots if hit and (u, s) in seen),
            "hits_new": sum(1 for u, s, hit in slots if hit and (u, s) not in seen),
            "slots_new": sum(1 for u, s, hit in slots if (u, s) not in seen)}


checks = {"E1 user split": (D["A"][D["A"].group == "other"], D["A"][D["A"].group == "sub"], true_top80((19, 30))),
          "E2 time split": (D["E"], D["B"][D["B"].group == "sub"], true_top80((23, 30)))}
out = {"features": FEATS, "params": {k: v for k, v in PARAMS.items() if k != "verbose"}, "checks": {}}
for cname, (train, test, truth) in checks.items():
    base = first_layer(test)
    assert len(base) == len(SUBU) and all(len(v) == K for v in base.values())
    scores, lists = {}, {"first_layer": base}
    for obj in ("binary", "lambdarank"):
        m = fit(train, obj)
        scores[obj] = score(m, test)
        lists[obj] = top5(test, scores[obj])
        res = compare(truth, base, lists[obj], SUBU)
        res["trees"] = int(np.mean([x.best_iteration_ or x.n_estimators for x in m]))
        out["checks"].setdefault(cname, {})[obj] = res
        print("%s | %-10s first layer %.4f -> rerank %.4f (gain %+.4f, 95%% CI %+.4f..%+.4f; users better %d, worse %d; %d trees) (%.0f s)"
              % (cname, obj, res["first_layer"], res["rerank"], res["gain"], res["gain_ci95"][0], res["gain_ci95"][1],
                 res["users_better"], res["users_worse"], res["trees"], time.time() - t0), flush=True)
    # both objectives together: mean of the two per-user ranks
    rk = lambda s: pd.Series(s, index=test.index).groupby(test.user_id).rank(ascending=False).to_numpy()
    lists["both"] = top5(test, -(rk(scores["binary"]) + rk(scores["lambdarank"])))
    res = compare(truth, base, lists["both"], SUBU)
    out["checks"][cname]["both"] = res
    out.setdefault("check_lists", {})[cname] = lists
    out.setdefault("discovery", {})[cname] = {k: discovery(test, l, truth) for k, l in lists.items()}
    print("%s | discovery (correct familiar, correct new, slots new): %s" % (
        cname, {k: tuple(v.values()) for k, v in out["discovery"][cname].items()}), flush=True)
    print("%s | %-10s first layer %.4f -> rerank %.4f (gain %+.4f, 95%% CI %+.4f..%+.4f)" % (
        cname, "both", res["first_layer"], res["rerank"], res["gain"], res["gain_ci95"][0], res["gain_ci95"][1]), flush=True)

ok = [o for o in ("binary", "lambdarank", "both") if all(out["checks"][c][o]["gain"] > 0 for c in checks)]
out["qualified"] = ok
if ok:
    pick = max(ok, key=lambda o: sum(out["checks"][c][o]["gain"] for c in checks))
    out["picked"] = pick
    train = pd.concat([D["A"], D["B"]], ignore_index=True)
    test = D["T"]
    models = {o: fit(train, o) for o in ("binary", "lambdarank") if pick in (o, "both")}
    if pick == "both":
        rk = lambda s: pd.Series(s, index=test.index).groupby(test.user_id).rank(ascending=False).to_numpy()
        s_test = -(rk(score(models["binary"], test)) + rk(score(models["lambdarank"], test)))
    else:
        s_test = score(models[pick], test)
    lists = top5(test, s_test)
    air = on_air(sched, (31, 42))
    lines = []
    for u in SUBU:
        top = lists[u]
        assert len(top) == K and len(set(top)) == K and all(s in air for s in top), (u, top)
        lines.append("%s,%s" % (u, " ".join(map(str, top))))
    path = ROOT / "submissions" / "sub_rerank.csv"
    path.write_text("user_id,tv_show_id\n" + "\n".join(lines) + "\n", encoding="utf-8")
    first = first_layer(test)
    out["test_shared_with_sub_counts"] = float(np.mean([len(set(lists[u]) & set(first[u])) / K for u in SUBU]))
    imp = {o: dict(zip(FEATS, np.mean([x.booster_.feature_importance("gain") for x in m], axis=0).round(1).tolist()))
           for o, m in models.items()}
    out["importance_gain"] = imp
    out["test_lists"] = lists
    print("picked %s; wrote %s (%d rows); slots shared with sub_counts.csv: %.3f" % (
        pick, path.name, len(lines), out["test_shared_with_sub_counts"]))
    for o, d in imp.items():
        top = sorted(d.items(), key=lambda kv: -kv[1])[:10]
        print("   %s top features by gain:" % o, ", ".join("%s %.0f" % kv for kv in top))
else:
    print("no model beats the first layer in both checks: keep sub_counts.csv")
json.dump(out, open(ROOT / "results" / "rerank.json", "w"), indent=1)
print("done in %.0f s" % (time.time() - t0))

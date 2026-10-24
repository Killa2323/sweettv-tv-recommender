# -*- coding: utf-8 -*-
"""Shared data access, the validation split and the MAP@5 metric.

Validation mirrors the competition: Kaggle scores the Top-5 shows of weeks 31-42
(12 weeks) after 20 weeks of history. Here the "future" is weeks 19-30 (12 weeks)
and the history is weeks 11-18. A user's true Top-5 is the five shows with the most
watch seconds in the future window (ties: smaller tv_show_id first). Candidates are
the shows that are on air in the window being predicted, because a show that is not
broadcast cannot be watched.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
K = 5

VAL = {"hist": (11, 18), "target": (19, 30)}
TEST = {"hist": (11, 30), "target": (31, 42)}


def load():
    inter = pd.read_parquet(PROC / "interactions.parquet")
    shows = pd.read_parquet(PROC / "shows.parquet")
    sched = pd.read_parquet(PROC / "schedule.parquet", columns=["channel_id", "tv_show_id", "week", "duration"])
    sub = pd.read_csv(DATA / "submission.csv", dtype=str)
    return inter, shows, sched, sub


def weeks(df, span):
    a, b = span
    return df[(df.week >= a) & (df.week <= b)]


def on_air(sched, span):
    """Non-zero shows broadcast at least once in the weeks of `span`."""
    s = weeks(sched, span)
    return set(s.loc[s.tv_show_id != 0, "tv_show_id"].unique())


def true_top(inter, span, k=K):
    """user_id -> list of the k most watched shows in `span` (by seconds, ties by id)."""
    t = weeks(inter, span).groupby(["user_id", "tv_show_id"], as_index=False)["seconds"].sum()
    t = t.sort_values(["user_id", "seconds", "tv_show_id"], ascending=[True, False, True])
    return t.groupby("user_id")["tv_show_id"].apply(lambda s: list(s.head(k))).to_dict()


def apk(actual, predicted, k=K):
    """Average precision at k, as in the Kaggle MAP@K metric."""
    if not actual:
        return 0.0
    predicted = list(predicted)[:k]
    hits, score = 0, 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1
            score += hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(truth, preds, users, k=K):
    """Mean AP@k over `users` (users without a prediction score 0)."""
    return float(np.mean([apk(truth.get(u, []), preds.get(u, []), k) for u in users]))

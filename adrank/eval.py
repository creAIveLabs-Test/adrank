"""Stage 1: the offline eval harness — the ruler every model is measured on.

Pipeline:
  1. Rebuild labeled examples from the event log: each impression becomes a row,
     label 1 if a matching click followed it. Clicks are matched to the most
     recent unmatched impression with the same (user, ad, position, page) —
     robust even when the same ad is shown to the same user repeatedly.
  2. TIME-BASED split: first 80% train, last 20% test. Never random — a random
     split leaks future popularity into the past.
  3. Score four baselines on the test set:
       random      p = global train CTR (AUC 0.5 by construction)
       popularity  p = per-ad smoothed train CTR
       logreg      logistic regression on honest features (train-only aggregates)
       oracle      the simulator's true click probability (upper bound;
                   allowed to read hidden ground truth — models never are)
  4. Report AUC, LogLoss, ECE (calibration), and in-pageview ranking accuracy
     (of page views with exactly one click, how often the model ranks the
     clicked ad above its slot-mates — a tiny NDCG stand-in with no
     off-policy correction needed because we compare within a served set).

Run:  uv run python -m adrank.eval
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sqlmodel import select

from .db import get_session
from .models import CATEGORIES, Ad, Event, SimUser
from .simulator import true_ctr

SMOOTH_CLICKS = 1.0   # Laplace smoothing for per-ad CTR
SMOOTH_IMPS = 50.0


# ── 1. Labeled examples from the log ─────────────────────────────────────────

def load_examples() -> pd.DataFrame:
    with get_session() as s:
        events = s.exec(select(Event).order_by(Event.ts, Event.id)).all()
        ads = {a.id: a for a in s.exec(select(Ad)).all()}
        users = {u.id: u for u in s.exec(select(SimUser)).all()}

    rows = []
    open_imps: dict[tuple, list[int]] = defaultdict(list)  # key -> row indexes awaiting a click
    for ev in events:
        key = (ev.user_id, ev.ad_id, ev.position, ev.page_category)
        if ev.event_type == "impression":
            ad = ads[ev.ad_id]
            rows.append({
                "ts": ev.ts, "user_id": ev.user_id, "ad_id": ev.ad_id,
                "position": ev.position, "page_category": ev.page_category,
                "ad_category": ad.category, "bid": ad.bid,
                "context_match": int(ev.page_category == ad.category),
                "clicked": 0,
            })
            open_imps[key].append(len(rows) - 1)
        else:  # click → most recent unmatched impression with the same key
            if open_imps[key]:
                rows[open_imps[key].pop()]["clicked"] = 1

    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    df["users_obj"] = df["user_id"].map(users)
    df["ads_obj"] = df["ad_id"].map(ads)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ── 2. Honest features (all aggregates computed on TRAIN only) ───────────────

def build_features(train: pd.DataFrame, test: pd.DataFrame):
    global_ctr = train["clicked"].mean()

    ad_stats = train.groupby("ad_id")["clicked"].agg(["sum", "count"])
    ad_ctr = ((ad_stats["sum"] + SMOOTH_CLICKS) /
              (ad_stats["count"] + SMOOTH_IMPS)).to_dict()

    uc = train.groupby(["user_id", "ad_category"])["clicked"].agg(["sum", "count"])
    user_cat_ctr = ((uc["sum"] + SMOOTH_CLICKS) /
                    (uc["count"] + SMOOTH_IMPS)).to_dict()

    def featurize(d: pd.DataFrame) -> np.ndarray:
        f = pd.DataFrame(index=d.index)
        f["ad_ctr"] = d["ad_id"].map(lambda a: ad_ctr.get(a, global_ctr))
        f["user_cat_ctr"] = [
            user_cat_ctr.get((u, c), global_ctr)
            for u, c in zip(d["user_id"], d["ad_category"])
        ]
        f["position"] = d["position"]
        f["context_match"] = d["context_match"]
        f["bid"] = d["bid"]
        for cat in CATEGORIES:
            f[f"cat_{cat}"] = (d["ad_category"] == cat).astype(int)
        return f.to_numpy(dtype=float)

    return featurize, global_ctr


# ── 3. Baselines ──────────────────────────────────────────────────────────────

def predict_baselines(train, test, featurize, global_ctr) -> dict[str, np.ndarray]:
    preds = {}
    preds["random"] = np.full(len(test), global_ctr)

    ad_stats = train.groupby("ad_id")["clicked"].agg(["sum", "count"])
    ad_ctr = ((ad_stats["sum"] + SMOOTH_CLICKS) /
              (ad_stats["count"] + SMOOTH_IMPS)).to_dict()
    preds["popularity"] = test["ad_id"].map(lambda a: ad_ctr.get(a, global_ctr)).to_numpy()

    lr = LogisticRegression(max_iter=2000)
    lr.fit(featurize(train), train["clicked"])
    preds["logreg"] = lr.predict_proba(featurize(test))[:, 1]

    preds["oracle"] = np.array([
        true_ctr(
            r.users_obj.interest_vector()[CATEGORIES.index(r.ad_category)],
            r.ads_obj.quality, r.position, bool(r.context_match),
        )
        for r in test.itertuples()
    ])
    return preds


# ── 4. Metrics ────────────────────────────────────────────────────────────────

def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: |mean predicted − actual| per bin, weighted."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


def pageview_ranking_accuracy(test: pd.DataFrame, p: np.ndarray) -> float:
    """Within page views that got exactly one click, how often does the model
    score the clicked ad above every slot-mate? (Within-served-set comparison,
    so no off-policy correction is needed.)"""
    d = test.copy()
    d["p"] = p
    d["pv"] = (d["ts"].astype("int64") // 10**9).astype(str) + "-" + d["user_id"].astype(str)
    wins = trials = 0
    for _, g in d.groupby("pv"):
        if len(g) < 2 or g["clicked"].sum() != 1:
            continue
        trials += 1
        clicked_p = g.loc[g["clicked"] == 1, "p"].iloc[0]
        if clicked_p > g.loc[g["clicked"] == 0, "p"].max():
            wins += 1
    return wins / trials if trials else float("nan")


def main() -> None:
    df = load_examples()
    train, test = time_split(df)
    print(f"{len(df)} impressions · {df['clicked'].sum()} clicks · "
          f"train {len(train)} / test {len(test)} (time split)\n")

    featurize, global_ctr = build_features(train, test)
    preds = predict_baselines(train, test, featurize, global_ctr)
    y = test["clicked"].to_numpy()

    print(f"{'model':<12}{'AUC':>8}{'LogLoss':>10}{'ECE':>8}{'RankAcc':>9}")
    for name, p in preds.items():
        p = np.clip(p, 1e-6, 1 - 1e-6)
        auc = roc_auc_score(y, p) if y.sum() else float("nan")
        print(f"{name:<12}{auc:>8.4f}{log_loss(y, p):>10.4f}"
              f"{ece(y, p):>8.4f}{pageview_ranking_accuracy(test, p):>9.4f}")
    print("\noracle = ceiling (reads hidden ground truth); models may only close the gap.")


if __name__ == "__main__":
    main()

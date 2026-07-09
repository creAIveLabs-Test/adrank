"""Stage 2: learned pCTR — champion selection + isotonic calibration.

Trains BOTH logistic regression and LightGBM on the logged events and
promotes whichever wins AUC on the calibration slice. Lab note from this
world: LR wins (test AUC ~0.65 vs ~0.55). The click function is smooth and
near-log-linear in the CTR-style features, so LR's inductive bias fits;
GBDT must approximate a smooth surface with step functions estimated from
only ~700 positives. At real ads scale (millions of rows, high-cardinality
raw IDs, strong feature interactions) the ordering flips — that flip is the
whole reason deep CTR models exist.

Split (chronological): 70% train / 10% calibration / 20% test.
Calibration matters because stage 4 prices with ad_rank = bid x pCTR:
a model that ranks well but runs hot corrupts the auction. Isotonic
regression on held-out data fixes the probability scale; see
reports/reliability_pctr.png for before/after.

Artifacts (models/): pctr_model.joblib (champion + calibrator + type),
pctr_aggs.json (train-time aggregates for request-time featurization).
Run:  uv run python -m adrank.pctr
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from .eval import (SMOOTH_CLICKS, SMOOTH_IMPS, build_features, ece,
                   load_examples, pageview_ranking_accuracy)
from .pctr_model import PctrModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",           # logloss early-stops after ~2 rounds at 3% CTR
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 30,
    "verbosity": -1,
    "seed": 42,
}


def reliability_diagram(y, p_raw, p_cal, path: Path, bins: int = 10) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for p, label, marker in ((p_raw, "before calibration", "o"),
                             (p_cal, "after isotonic", "s")):
        edges = np.quantile(p, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
        xs = [p[idx == b].mean() for b in range(bins) if (idx == b).sum() > 0]
        ys = [y[idx == b].mean() for b in range(bins) if (idx == b).sum() > 0]
        ax.plot(xs, ys, marker=marker, label=label)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color="grey", label="perfect")
    ax.set_xlabel("mean predicted pCTR (bin)")
    ax.set_ylabel("actual CTR (bin)")
    ax.set_title("pCTR reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_examples()
    n = len(df)
    train = df.iloc[: int(n * 0.7)].copy()
    calib = df.iloc[int(n * 0.7): int(n * 0.8)].copy()
    test = df.iloc[int(n * 0.8):].copy()
    print(f"{n} impressions -> train {len(train)} / calib {len(calib)} / test {len(test)}")

    featurize, global_ctr = build_features(train, test)
    X_tr, y_tr = featurize(train), train["clicked"].to_numpy()
    X_ca, y_ca = featurize(calib), calib["clicked"].to_numpy()
    X_te, y_te = featurize(test), test["clicked"].to_numpy()

    # Contender 1: logistic regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_tr, y_tr)
    # Contender 2: LightGBM
    booster = lgb.train(
        LGB_PARAMS, lgb.Dataset(X_tr, y_tr),
        valid_sets=[lgb.Dataset(X_ca, y_ca)],
        num_boost_round=800,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    contenders = {
        "logreg": lr.predict_proba(X_ca)[:, 1],
        "gbdt": booster.predict(X_ca),
    }
    champ_kind = max(contenders, key=lambda k: roc_auc_score(y_ca, contenders[k]))
    champ = lr if champ_kind == "logreg" else booster
    print(f"champion on calibration AUC: {champ_kind} "
          f"({ {k: round(roc_auc_score(y_ca, v), 4) for k, v in contenders.items()} })")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=0.5)
    iso.fit(contenders[champ_kind], y_ca)
    model = PctrModel(champ_kind, champ, iso)

    p_raw = model.predict_raw(X_te)
    p_cal = model.predict(X_te)
    print(f"\n{'model':<18}{'AUC':>8}{'LogLoss':>10}{'ECE':>8}{'RankAcc':>9}")
    for name, p in ((f"{champ_kind} raw", p_raw), (f"{champ_kind} calibrated", p_cal)):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        print(f"{name:<18}{roc_auc_score(y_te, p):>8.4f}{log_loss(y_te, p):>10.4f}"
              f"{ece(y_te, p):>8.4f}{pageview_ranking_accuracy(test, p):>9.4f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    reliability_diagram(y_te, p_raw, p_cal, REPORTS_DIR / "reliability_pctr.png")
    print(f"\nreliability diagram -> {REPORTS_DIR / 'reliability_pctr.png'}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "pctr_model.joblib")
    ad_stats = train.groupby("ad_id")["clicked"].agg(["sum", "count"])
    uc = train.groupby(["user_id", "ad_category"])["clicked"].agg(["sum", "count"])
    aggs = {
        "global_ctr": float(global_ctr),
        "smooth_clicks": SMOOTH_CLICKS,
        "smooth_imps": SMOOTH_IMPS,
        "ad_ctr": {str(k): [float(v["sum"]), float(v["count"])] for k, v in ad_stats.iterrows()},
        "user_cat_ctr": {f"{u}|{c}": [float(v["sum"]), float(v["count"])]
                          for (u, c), v in uc.iterrows()},
    }
    (MODELS_DIR / "pctr_aggs.json").write_text(json.dumps(aggs))
    print(f"serving artifacts -> {MODELS_DIR}/")


if __name__ == "__main__":
    main()

"""Serving-side pCTR wrapper — separate module so joblib pickles a stable
import path (adrank.pctr_model.PctrModel) no matter how training was invoked."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .models import CATEGORIES


class PctrModel:
    """Champion model + isotonic calibrator behind one predict interface."""

    def __init__(self, kind: str, model, iso: IsotonicRegression):
        self.kind = kind
        self.model = model
        self.iso = iso

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "logreg":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.iso.predict(self.predict_raw(X))


def featurize_serving(cands: list[dict], aggs: dict) -> np.ndarray:
    """Request-time featurization — must mirror eval.build_features exactly.
    Each candidate dict: user_id, ad_id, ad_category, bid, context_match.
    Position is 0 for all: we score as-if top slot and rank by that score."""
    g = aggs["global_ctr"]
    sc, si = aggs["smooth_clicks"], aggs["smooth_imps"]

    def smoothed(entry):
        if entry is None:
            return g
        s, c = entry
        return (s + sc) / (c + si)

    rows = []
    for c in cands:
        rows.append(
            [smoothed(aggs["ad_ctr"].get(str(c["ad_id"])))]
            + [smoothed(aggs["user_cat_ctr"].get(f"{c['user_id']}|{c['ad_category']}"))]
            + [0, c["context_match"], c["bid"]]
            + [int(c["ad_category"] == cat) for cat in CATEGORIES]
        )
    return np.array(rows, dtype=float)

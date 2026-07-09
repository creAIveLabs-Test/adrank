"""Serving policy: which ads fill the slots for this request.

Stage 0 policy is random-among-eligible — deliberately dumb, so every later
stage (pCTR model, two-tower retrieval, auction) has an honest baseline.
Each policy returns (ads, version_tag); the tag is logged with every
impression so offline eval can compare policies on their own traffic.
"""
import random

from sqlmodel import Session, select

from .models import N_SLOTS, Ad


def eligible_ads(session: Session, page_category: str) -> list[Ad]:
    """Targeting filter + budget filter. This is the top of the funnel."""
    ads = session.exec(select(Ad).where(Ad.active == True)).all()  # noqa: E712
    out = []
    for ad in ads:
        if ad.spent_today >= ad.daily_budget:
            continue
        if ad.target_categories and page_category not in ad.target_categories.split(","):
            continue
        out.append(ad)
    return out


def serve_random(session: Session, page_category: str) -> tuple[list[Ad], str]:
    pool = eligible_ads(session, page_category)
    k = min(N_SLOTS, len(pool))
    return random.sample(pool, k=k), "random-v0"


def serve_pctr(session: Session, page_category: str, user_id: int) -> tuple[list[Ad], str]:
    """Stage 2 policy: rank eligible ads by calibrated pCTR, serve top slots."""
    import json as _json
    from pathlib import Path

    import joblib
    import numpy as np

    from .pctr_model import featurize_serving  # lazy import

    global _PCTR_MODEL, _PCTR_AGGS
    if _PCTR_MODEL is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
        _PCTR_MODEL = joblib.load(models_dir / "pctr_model.joblib")
        _PCTR_AGGS = _json.loads((models_dir / "pctr_aggs.json").read_text())

    pool = eligible_ads(session, page_category)
    if not pool:
        return [], "pctr-v1"
    cands = [{"user_id": user_id, "ad_id": ad.id, "ad_category": ad.category,
              "bid": ad.bid, "context_match": int(page_category == ad.category)}
             for ad in pool]
    scores = _PCTR_MODEL.predict(featurize_serving(cands, _PCTR_AGGS))
    order = np.argsort(-scores)[:N_SLOTS]
    return [pool[i] for i in order], "pctr-v1"


_PCTR_MODEL = None
_PCTR_AGGS = None

# Single dispatch point — later stages register "twotower-v2", ...
POLICIES = {"random-v0": serve_random, "pctr-v1": serve_pctr}
ACTIVE_POLICY = "pctr-v1"


def serve(session: Session, page_category: str, user_id: int) -> tuple[list[Ad], str]:
    policy = POLICIES[ACTIVE_POLICY]
    if policy is serve_random:
        return serve_random(session, page_category)
    return policy(session, page_category, user_id)

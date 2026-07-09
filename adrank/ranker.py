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


# Single dispatch point — later stages register "pctr-v1", "twotower-v2", ...
POLICIES = {"random-v0": serve_random}
ACTIVE_POLICY = "random-v0"


def serve(session: Session, page_category: str) -> tuple[list[Ad], str]:
    return POLICIES[ACTIVE_POLICY](session, page_category)

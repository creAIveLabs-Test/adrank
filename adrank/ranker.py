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
_TT = None


def serve_twotower(session: Session, page_category: str, user_id: int) -> tuple[list[Ad], str]:
    """Stage 3 policy — the full funnel:
    eligible -> retrieve candidates (two-tower ANN UNION popularity) -> pCTR rank.
    Multi-source retrieval: embeddings personalize, popularity backstops; the
    union beat either source alone at equal K in offline recall."""
    import json as _json
    from pathlib import Path

    import joblib
    import numpy as np

    from .pctr_model import featurize_serving
    from .twotower import numpy_forward

    global _PCTR_MODEL, _PCTR_AGGS, _TT
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if _PCTR_MODEL is None:
        _PCTR_MODEL = joblib.load(models_dir / "pctr_model.joblib")
        _PCTR_AGGS = _json.loads((models_dir / "pctr_aggs.json").read_text())
    if _TT is None:
        s = np.load(models_dir / "twotower_serving.npz")
        w = np.load(models_dir / "twotower.npz")
        _TT = {
            "row_of_ad": {int(a): i for i, a in enumerate(s["ad_ids"])},
            "ad_embs": s["ad_embs"],
            "pop": s["pop_counts"],
            "user_row": {int(u): i for i, u in enumerate(s["user_ids"])},
            "user_feats": s["user_feats"],
            "w": {k: w[k] for k in w.files},
        }

    pool = eligible_ads(session, page_category)
    if not pool:
        return [], "twotower-v2"

    # Retrieval over the eligible subset (25 per source)
    rows = np.array([_TT["row_of_ad"][ad.id] for ad in pool])
    urow = _TT["user_row"].get(user_id)
    if urow is not None:
        u_emb = numpy_forward(_TT["user_feats"][urow][None, :], _TT["w"], "u")[0]
        tt_scores = _TT["ad_embs"][rows] @ u_emb
        tt_top = set(np.argsort(-tt_scores)[:25])
    else:
        tt_top = set()  # cold user: popularity only
    pop_top = set(np.argsort(-_TT["pop"][rows])[:25])
    cand_idx = sorted(tt_top | pop_top)
    cands_pool = [pool[i] for i in cand_idx]

    # Ranking: calibrated pCTR over the ~50 candidates
    cands = [{"user_id": user_id, "ad_id": ad.id, "ad_category": ad.category,
              "bid": ad.bid, "context_match": int(page_category == ad.category)}
             for ad in cands_pool]
    scores = _PCTR_MODEL.predict(featurize_serving(cands, _PCTR_AGGS))
    order = np.argsort(-scores)[:N_SLOTS]
    return [cands_pool[i] for i in order], "twotower-v2"


def _retrieve_and_score(session: Session, page_category: str, user_id: int,
                        calibrated: bool = True):
    """Shared funnel front: eligible -> multi-source retrieval -> pCTR scores.
    Returns list of (ad, pctr). Used by twotower-v2 (rank by pCTR) and
    auction-v3 (rank by bid x pCTR + GSP prices)."""
    import json as _json
    from pathlib import Path

    import joblib
    import numpy as np

    from .pctr_model import featurize_serving
    from .twotower import numpy_forward

    global _PCTR_MODEL, _PCTR_AGGS, _TT
    models_dir = Path(__file__).resolve().parent.parent / "models"
    if _PCTR_MODEL is None:
        _PCTR_MODEL = joblib.load(models_dir / "pctr_model.joblib")
        _PCTR_AGGS = _json.loads((models_dir / "pctr_aggs.json").read_text())
    if _TT is None:
        s = np.load(models_dir / "twotower_serving.npz")
        w = np.load(models_dir / "twotower.npz")
        _TT = {
            "row_of_ad": {int(a): i for i, a in enumerate(s["ad_ids"])},
            "ad_embs": s["ad_embs"],
            "pop": s["pop_counts"],
            "user_row": {int(u): i for i, u in enumerate(s["user_ids"])},
            "user_feats": s["user_feats"],
            "w": {k: w[k] for k in w.files},
        }

    pool = eligible_ads(session, page_category)
    if not pool:
        return []
    rows = np.array([_TT["row_of_ad"][ad.id] for ad in pool])
    urow = _TT["user_row"].get(user_id)
    if urow is not None:
        u_emb = numpy_forward(_TT["user_feats"][urow][None, :], _TT["w"], "u")[0]
        tt_top = set(np.argsort(-(_TT["ad_embs"][rows] @ u_emb))[:25])
    else:
        tt_top = set()
    pop_top = set(np.argsort(-_TT["pop"][rows])[:25])
    cands_pool = [pool[i] for i in sorted(tt_top | pop_top)]

    cands = [{"user_id": user_id, "ad_id": ad.id, "ad_category": ad.category,
              "bid": ad.bid, "context_match": int(page_category == ad.category)}
             for ad in cands_pool]
    X = featurize_serving(cands, _PCTR_AGGS)
    scores = _PCTR_MODEL.predict(X) if calibrated else _PCTR_MODEL.predict_raw(X)
    return list(zip(cands_pool, scores))


def serve_auction(session: Session, page_category: str, user_id: int):
    """Stage 4 policy: full funnel + GSP auction. Returns winners with CPC prices."""
    from .auction import run_auction

    scored = _retrieve_and_score(session, page_category, user_id)
    if not scored:
        return [], [], "auction-v3"
    board = run_auction(scored)
    winners = [a for a in board if a.slot is not None]
    winners.sort(key=lambda a: a.slot)
    return [w.ad for w in winners], [w.price for w in winners], "auction-v3"


# Single dispatch point
POLICIES = {"random-v0": serve_random, "pctr-v1": serve_pctr,
            "twotower-v2": serve_twotower, "auction-v3": serve_auction}
ACTIVE_POLICY = "auction-v3"


def serve(session: Session, page_category: str, user_id: int):
    """Returns (ads, prices, version). Pre-auction policies charge the full
    bid on click (first-price billing) — kept that way on purpose so the
    revenue comparison between naive billing and GSP stays visible."""
    policy = POLICIES[ACTIVE_POLICY]
    if policy is serve_auction:
        return policy(session, page_category, user_id)
    if policy is serve_random:
        ads, v = serve_random(session, page_category)
    else:
        ads, v = policy(session, page_category, user_id)
    return ads, [ad.bid for ad in ads], v

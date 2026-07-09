"""Traffic simulator: synthetic users browse the feed and click ads.

This is the ground-truth side of AdRank. It knows each user's hidden
interest vector and each ad's hidden quality; the serving side never does.

Click model:
  p(click | user, ad, slot) =
      POSITION_DECAY[slot]                     (does the user even see it)
    * clip(0.015 + 0.25 * interest * quality)  (do they want it)
    * 1.3 if page_category == ad.category      (contextual bump)

Run:  uv run python -m adrank.simulator --sessions 500
"""
import argparse
import random

import httpx
import numpy as np
from sqlmodel import select

from .db import get_session
from .models import CATEGORIES, POSITION_DECAY, Ad, SimUser

RNG = np.random.default_rng()


def true_ctr(interest: float, quality: float, slot: int, context_match: bool) -> float:
    base = 0.015 + 0.25 * interest * quality
    if context_match:
        base *= 1.3
    return float(np.clip(base, 0.002, 0.35) * POSITION_DECAY[slot])


def pick_page(interests: list[float]) -> str:
    """Users mostly read pages about what they like (correlated context)."""
    w = np.array(interests) + 0.05
    return CATEGORIES[int(RNG.choice(len(CATEGORIES), p=w / w.sum()))]


def run_direct(n_sessions: int, pages_per_session: int = 3,
               bot_users: int = 0, bot_click_rate: float = 0.25) -> None:
    """Bulk mode: drives the same serving funnel in-process (no HTTP) — for
    generating job-scale traffic volumes. bot_users > 0 adds users who click
    ~uniformly at bot_click_rate regardless of interest (chaos drills).
    NOTE: exercises ranker.serve + Event writes directly; the HTTP layer is
    covered by the default mode."""
    import random as _random

    from .db import get_session
    from .models import Event
    from .ranker import serve

    with get_session() as s:
        users = s.exec(select(SimUser)).all()
        ads = {a.id: a for a in s.exec(select(Ad)).all()}
    weights = np.array([u.activity for u in users])
    weights = weights / weights.sum()
    bot_ids = set(u.id for u in users[:bot_users])

    imps = clicks = 0
    with get_session() as s:
        for i in range(n_sessions):
            user = users[int(RNG.choice(len(users), p=weights))]
            interests = user.interest_vector()
            for _ in range(pages_per_session):
                page = pick_page(interests)
                served, prices, tags = serve(s, page, user.id)
                for pos, ad in enumerate(served):
                    s.add(Event(event_type="impression", user_id=user.id, ad_id=ad.id,
                                position=pos, page_category=page, served_by=tags[pos]))
                    imps += 1
                    if user.id in bot_ids:
                        p = bot_click_rate * POSITION_DECAY[pos]
                    else:
                        interest = interests[CATEGORIES.index(ad.category)]
                        p = true_ctr(interest, ad.quality, pos, page == ad.category)
                    if _random.random() < p:
                        clicks += 1
                        price = min(prices[pos], ad.bid)
                        s.add(Event(event_type="click", user_id=user.id, ad_id=ad.id,
                                    position=pos, page_category=page, served_by=tags[pos],
                                    price_paid=price))
                        ad.spent_today = round(ad.spent_today + price, 4)
                        s.add(ad)
            if (i + 1) % 500 == 0:
                s.commit()
                print(f"  {i+1}/{n_sessions} sessions · {imps} imps · {clicks} clicks · "
                      f"CTR {clicks/imps:.4f}")
        s.commit()
    print(f"Done (direct): {n_sessions} sessions, {imps} impressions, {clicks} clicks, "
          f"CTR {clicks/max(imps,1):.4f}")


def run(base_url: str, n_sessions: int, pages_per_session: int = 3) -> None:
    with get_session() as s:
        users = s.exec(select(SimUser)).all()
        ads = {a.id: a for a in s.exec(select(Ad)).all()}
    weights = np.array([u.activity for u in users])
    weights = weights / weights.sum()

    imps = clicks = 0
    with httpx.Client(base_url=base_url, timeout=10) as http:
        for i in range(n_sessions):
            user = users[int(RNG.choice(len(users), p=weights))]
            interests = user.interest_vector()
            for _ in range(pages_per_session):
                page = pick_page(interests)
                r = http.get("/serve-ads", params={"user_id": user.id, "page_category": page})
                r.raise_for_status()
                payload = r.json()
                for slot in payload["ads"]:
                    imps += 1
                    ad = ads[slot["ad_id"]]
                    interest = interests[CATEGORIES.index(ad.category)]
                    p = true_ctr(interest, ad.quality, slot["position"], page == ad.category)
                    if random.random() < p:
                        clicks += 1
                        http.post("/event", json={
                            "user_id": user.id, "ad_id": ad.id,
                            "position": slot["position"], "page_category": page,
                            "served_by": slot.get("served_by", payload["served_by"]),
                            "price": slot.get("price"),
                        }).raise_for_status()
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{n_sessions} sessions · {imps} impressions · "
                      f"{clicks} clicks · CTR {clicks/imps:.4f}")
    print(f"Done: {n_sessions} sessions, {imps} impressions, {clicks} clicks, "
          f"CTR {clicks/max(imps,1):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=500)
    ap.add_argument("--base-url", default="http://127.0.0.1:8100")
    ap.add_argument("--direct", action="store_true", help="in-process bulk mode (no HTTP)")
    ap.add_argument("--bot-users", type=int, default=0)
    args = ap.parse_args()
    if args.direct or args.bot_users:
        run_direct(args.sessions, bot_users=args.bot_users)
    else:
        run(args.base_url, args.sessions)

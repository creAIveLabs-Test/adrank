"""Generate the AdRank world: ~500 ads and ~800 synthetic users.

Users get sparse hidden interest vectors (2-4 strong categories each).
Ads get a bid (lognormal, like real auctions), a budget, and a hidden quality.
Run:  uv run python -m adrank.seed
"""
import random

import numpy as np
from sqlmodel import select

from .db import get_session, init_db
from .models import CATEGORIES, Ad, Event, SimUser

RNG = np.random.default_rng(42)
random.seed(42)

N_ADS = 500
N_USERS = 800

ADJECTIVES = ["Smart", "Fresh", "Ultra", "Daily", "Prime", "Pure", "Swift", "Cozy", "Bold", "True"]
OFFERS = ["20% off", "free trial", "2-for-1", "new arrival", "members deal", "flash sale",
          "starter kit", "premium plan", "weekend special", "bundle offer"]


def make_ads() -> list[Ad]:
    ads = []
    for i in range(N_ADS):
        cat = random.choice(CATEGORIES)
        adj = random.choice(ADJECTIVES)
        advertiser = f"{adj}{cat.capitalize()} Co"
        headline = f"{adj} {cat} — {random.choice(OFFERS)}"
        # Lognormal bids: many small, few large — like real auctions.
        bid = float(np.clip(RNG.lognormal(mean=-0.7, sigma=0.7), 0.10, 5.00))
        budget = float(RNG.uniform(20, 200))
        quality = float(RNG.uniform(0.3, 1.0))
        # 30% of ads target specific page categories; the rest run everywhere.
        if random.random() < 0.30:
            targets = ",".join(sorted(random.sample(CATEGORIES, k=random.randint(1, 3))))
        else:
            targets = ""
        ads.append(Ad(advertiser=advertiser, headline=headline, category=cat,
                      bid=round(bid, 2), daily_budget=round(budget, 2),
                      quality=round(quality, 3), target_categories=targets))
    return ads


def make_users() -> list[SimUser]:
    users = []
    for i in range(N_USERS):
        n_strong = random.randint(2, 4)
        strong = set(random.sample(range(len(CATEGORIES)), k=n_strong))
        vec = [
            round(float(RNG.uniform(0.6, 1.0)), 3) if c in strong
            else round(float(RNG.uniform(0.0, 0.15)), 3)
            for c in range(len(CATEGORIES))
        ]
        users.append(SimUser(
            name=f"user_{i:04d}",
            interests=",".join(str(v) for v in vec),
            activity=round(float(RNG.uniform(0.3, 2.0)), 2),
        ))
    return users


def main() -> None:
    init_db()
    with get_session() as s:
        if s.exec(select(Ad).limit(1)).first():
            print("World already seeded — delete adrank.db to reseed.")
            return
        for ad in make_ads():
            s.add(ad)
        for u in make_users():
            s.add(u)
        s.commit()
        n_ads = len(s.exec(select(Ad)).all())
        n_users = len(s.exec(select(SimUser)).all())
    print(f"Seeded {n_ads} ads and {n_users} users into adrank.db")


if __name__ == "__main__":
    main()

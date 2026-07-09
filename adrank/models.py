"""Data model for the AdRank world.

Three tables:
  Ad      — the inventory: creative, category, bid, budget, and a hidden
            `quality` used only by the ground-truth click model (the platform
            must never read it at serving time — that would be cheating).
  SimUser — synthetic users. `interests` is the hidden ground-truth interest
            vector over categories; same rule: simulator-only.
  Event   — append-only impression/click log. This is the ONLY data models
            are allowed to train on.
"""
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel

CATEGORIES = [
    "tech", "fitness", "travel", "food", "fashion",
    "gaming", "home", "auto", "finance", "beauty",
    "pets", "sports", "books", "music", "kids",
    "garden", "health", "education", "coffee", "outdoors",
]

N_SLOTS = 3
# Ground-truth probability a user even looks at slot k (position bias).
POSITION_DECAY = [1.0, 0.6, 0.35]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ad(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    advertiser: str
    headline: str
    category: str = Field(index=True)
    bid: float                      # CPC bid in dollars
    daily_budget: float
    spent_today: float = 0.0
    quality: float                  # HIDDEN: true creative quality in [0.3, 1.0]
    target_categories: str = ""     # comma-separated page categories; "" = run everywhere
    active: bool = True


class SimUser(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    interests: str                  # HIDDEN: comma-separated floats, one per category
    activity: float = 1.0           # relative browsing frequency

    def interest_vector(self) -> list[float]:
        return [float(x) for x in self.interests.split(",")]


class Event(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    event_type: str = Field(index=True)   # "impression" | "click"
    user_id: int = Field(index=True)
    ad_id: int = Field(index=True)
    position: int                          # 0-based slot index
    page_category: str                     # context of the page being browsed
    served_by: str = "random-v0"           # model/policy version that served this
    price_paid: float = 0.0                # auction price (stage 4+)

"""Stage 5: the experiments platform in miniature.

Assignment: deterministic hash bucketing — md5(experiment_name:user_id) mod 2.
Same user, same arm, every request, no storage; changing the experiment name
reshuffles everyone (a new experiment must not inherit the last one's split).

Readout: per-arm CTR and revenue with a two-proportion z-test. Exploration
impressions (served_by suffix "+explore") are excluded from arm readouts —
they're policy-independent by construction.
"""
import hashlib
import math

from sqlmodel import Session, func, select

from .models import Event

EXPERIMENT = {
    "name": "auction-vs-pctr-1",
    "enabled": True,
    "arms": {"A": "pctr-v1", "B": "auction-v3"},
    # Readout window starts when the experiment was turned on. Without this,
    # arm counts absorb historical traffic from when a single policy served
    # EVERYONE (found the hard way: the first readout mixed stage-2/4 traffic
    # into both arms).
    "start": "2026-07-09T18:01:03.635138+00:00",
}


def assign(user_id: int, name: str | None = None) -> str:
    name = name or EXPERIMENT["name"]
    h = hashlib.md5(f"{name}:{user_id}".encode()).hexdigest()
    return "A" if int(h, 16) % 2 == 0 else "B"


def _arm_counts(s: Session, policy: str) -> tuple[int, int, float]:
    from datetime import datetime
    start = datetime.fromisoformat(EXPERIMENT["start"])
    imps = s.exec(select(func.count()).where(
        Event.event_type == "impression", Event.served_by == policy,
        Event.ts >= start)).one()
    clicks = s.exec(select(func.count()).where(
        Event.event_type == "click", Event.served_by == policy,
        Event.ts >= start)).one()
    rev = s.exec(select(func.sum(Event.price_paid)).where(
        Event.event_type == "click", Event.served_by == policy,
        Event.ts >= start)).one() or 0.0
    return imps, clicks, rev


def readout(s: Session) -> dict:
    arms = {}
    for arm, policy in EXPERIMENT["arms"].items():
        n, c, rev = _arm_counts(s, policy)
        arms[arm] = {
            "policy": policy, "impressions": n, "clicks": c,
            "ctr": round(c / n, 4) if n else 0.0,
            "revenue": round(rev, 2),
            "rev_per_1k_imps": round(1000 * rev / n, 2) if n else 0.0,
        }

    a, b = arms["A"], arms["B"]
    result = {"experiment": EXPERIMENT["name"], "enabled": EXPERIMENT["enabled"],
              "arms": arms}
    if a["impressions"] and b["impressions"]:
        p1, n1 = a["ctr"], a["impressions"]
        p2, n2 = b["ctr"], b["impressions"]
        pooled = (a["clicks"] + b["clicks"]) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        z = (p2 - p1) / se if se else 0.0
        p_value = math.erfc(abs(z) / math.sqrt(2))   # two-sided
        result["ctr_test"] = {
            "delta": round(p2 - p1, 4), "z": round(z, 3),
            "p_value": round(p_value, 4),
            "significant_at_05": p_value < 0.05,
        }
    return result

"""Stage 4: the auction — where ranking becomes an economic decision.

Ranking:  ad_rank = bid x calibrated pCTR  (the classic search-ads rule:
          an ad wins by expected value to the platform, not bid alone —
          a $5 bid with pCTR 0.01 loses to a $1 bid with pCTR 0.08).

Pricing:  generalized second price (GSP) with quality normalization.
          The winner of slot k pays the minimum CPC that still beats the
          next competitor's ad_rank:
              price_k = ad_rank_{k+1} / pCTR_k + $0.01
          capped at the winner's own bid, floored at the reserve. Charged
          only on click (CPC). This is why calibration matters: pCTR sits
          in the DENOMINATOR of the price — a miscalibrated model doesn't
          just reorder ads, it misprices every click.
"""
from dataclasses import dataclass

from .models import N_SLOTS, Ad

RESERVE_PRICE = 0.05   # minimum CPC


@dataclass
class Allocation:
    ad: Ad
    pctr: float
    ad_rank: float
    slot: int | None      # None = did not win a slot
    price: float | None   # CPC the winner will pay on click


def run_auction(candidates: list[tuple[Ad, float]]) -> list[Allocation]:
    """candidates: (ad, calibrated_pctr) pairs. Returns ALL candidates with
    their auction outcome — winners get slot+price, losers get None — so the
    debug panel can show the whole board."""
    board = sorted(
        (Allocation(ad=ad, pctr=p, ad_rank=ad.bid * p, slot=None, price=None)
         for ad, p in candidates),
        key=lambda a: -a.ad_rank,
    )
    for k in range(min(N_SLOTS, len(board))):
        winner = board[k]
        if k + 1 < len(board):
            runner_up_rank = board[k + 1].ad_rank
            price = runner_up_rank / winner.pctr + 0.01
        else:
            price = RESERVE_PRICE
        winner.price = round(min(max(price, RESERVE_PRICE), winner.ad.bid), 4)
        winner.slot = k
    return board

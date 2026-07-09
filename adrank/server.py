"""AdRank ad server.

GET  /                → demo feed page (pick a persona, browse, see ads)
GET  /serve-ads       → run the funnel for one page view; logs impressions
POST /event           → log a click
GET  /stats           → live counters: impressions, clicks, CTR, spend
Run:  uv run uvicorn adrank.server:app --reload --port 8100
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from sqlmodel import func, select

from .db import get_session, init_db
from .models import CATEGORIES, Ad, Event, SimUser
from .ranker import serve

app = FastAPI(title="AdRank")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def feed(request: Request):
    with get_session() as s:
        users = s.exec(select(SimUser).limit(30)).all()
    return templates.TemplateResponse(
        request, "feed.html",
        {"users": users, "categories": CATEGORIES},
    )


@app.get("/serve-ads")
def serve_ads(user_id: int, page_category: str):
    if page_category not in CATEGORIES:
        raise HTTPException(400, f"unknown page_category {page_category!r}")
    with get_session() as s:
        if not s.get(SimUser, user_id):
            raise HTTPException(404, "unknown user")
        ads, prices, tags = serve(s, page_category, user_id)
        for pos, ad in enumerate(ads):
            s.add(Event(event_type="impression", user_id=user_id, ad_id=ad.id,
                        position=pos, page_category=page_category, served_by=tags[pos]))
        s.commit()
        return {
            "served_by": tags[0] if tags else "none",
            "ads": [
                {"ad_id": ad.id, "position": pos, "advertiser": ad.advertiser,
                 "headline": ad.headline, "category": ad.category,
                 "price": prices[pos], "served_by": tags[pos]}
                for pos, ad in enumerate(ads)
            ],
        }


class ClickIn(BaseModel):
    user_id: int
    ad_id: int
    position: int
    page_category: str
    served_by: str = "random-v0"
    price: float | None = None   # auction CPC quoted at serve time


@app.post("/event")
def click(ev: ClickIn):
    with get_session() as s:
        ad = s.get(Ad, ev.ad_id)
        if not ad:
            raise HTTPException(404, "unknown ad")
        # GSP billing: charge the quoted auction price, never more than the bid.
        # Pre-auction policies quote the full bid (first-price) — kept visible
        # so the revenue comparison across policies stays honest.
        price = min(ev.price, ad.bid) if ev.price is not None else ad.bid
        s.add(Event(event_type="click", user_id=ev.user_id, ad_id=ev.ad_id,
                    position=ev.position, page_category=ev.page_category,
                    served_by=ev.served_by, price_paid=price))
        ad.spent_today = round(ad.spent_today + price, 4)
        s.add(ad)
        s.commit()
    return {"ok": True}


@app.get("/debug/auction")
def debug_auction(user_id: int, page_category: str, calibrated: bool = True):
    """The 'why did this ad win' panel: the full auction board for one request.
    Logs nothing. calibrated=false shows what raw (uncalibrated) pCTR would do
    to prices — the calibration payoff, live."""
    from .auction import run_auction
    from .ranker import _retrieve_and_score

    with get_session() as s:
        scored = _retrieve_and_score(s, page_category, user_id, calibrated=calibrated)
    board = run_auction(scored)
    return {
        "user_id": user_id, "page_category": page_category, "calibrated": calibrated,
        "board": [
            {"slot": a.slot, "advertiser": a.ad.advertiser, "category": a.ad.category,
             "bid": a.ad.bid, "pctr": round(float(a.pctr), 5),
             "ad_rank": round(float(a.ad_rank), 5),
             "price": a.price,
             "note": ("wins: pays min CPC to hold slot" if a.slot is not None else "")}
            for a in board
        ],
    }


@app.get("/auction", response_class=HTMLResponse)
def auction_page(request: Request):
    with get_session() as s:
        users = s.exec(select(SimUser).limit(30)).all()
    return templates.TemplateResponse(
        request, "auction.html", {"users": users, "categories": CATEGORIES})


@app.get("/experiment")
def experiment_readout():
    from .experiment import readout
    with get_session() as s:
        return readout(s)


@app.post("/admin/reload-models")
def reload_models():
    """Hot-swap: clear model caches so the next request loads fresh artifacts.
    Called by the retrain loop after it rewrites models/."""
    from . import ranker
    ranker._PCTR_MODEL = None
    ranker._PCTR_AGGS = None
    ranker._TT = None
    ranker._TS_CACHE["stats"] = None
    return {"reloaded": True}


@app.get("/stats")
def stats():
    with get_session() as s:
        imps = s.exec(select(func.count()).where(Event.event_type == "impression")).one()
        clicks = s.exec(select(func.count()).where(Event.event_type == "click")).one()
        spend = s.exec(select(func.sum(Event.price_paid)).where(Event.event_type == "click")).one() or 0.0
        by_pos = {
            pos: {
                "impressions": s.exec(select(func.count()).where(
                    Event.event_type == "impression", Event.position == pos)).one(),
                "clicks": s.exec(select(func.count()).where(
                    Event.event_type == "click", Event.position == pos)).one(),
            }
            for pos in range(3)
        }
        for v in by_pos.values():
            v["ctr"] = round(v["clicks"] / v["impressions"], 4) if v["impressions"] else 0.0
        versions = s.exec(select(Event.served_by).distinct()).all()
        by_policy = {}
        for v in versions:
            vi = s.exec(select(func.count()).where(
                Event.event_type == "impression", Event.served_by == v)).one()
            vc = s.exec(select(func.count()).where(
                Event.event_type == "click", Event.served_by == v)).one()
            vrev = s.exec(select(func.sum(Event.price_paid)).where(
                Event.event_type == "click", Event.served_by == v)).one() or 0.0
            by_policy[v] = {"impressions": vi, "clicks": vc,
                            "ctr": round(vc / vi, 4) if vi else 0.0,
                            "revenue": round(vrev, 2),
                            "rev_per_1k_imps": round(1000 * vrev / vi, 2) if vi else 0.0}
    return {
        "impressions": imps,
        "clicks": clicks,
        "ctr": round(clicks / imps, 4) if imps else 0.0,
        "revenue": round(spend, 2),
        "by_position": by_pos,
        "by_policy": by_policy,
    }

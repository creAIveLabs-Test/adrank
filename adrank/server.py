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
        ads, version = serve(s, page_category, user_id)
        for pos, ad in enumerate(ads):
            s.add(Event(event_type="impression", user_id=user_id, ad_id=ad.id,
                        position=pos, page_category=page_category, served_by=version))
        s.commit()
        return {
            "served_by": version,
            "ads": [
                {"ad_id": ad.id, "position": pos, "advertiser": ad.advertiser,
                 "headline": ad.headline, "category": ad.category}
                for pos, ad in enumerate(ads)
            ],
        }


class ClickIn(BaseModel):
    user_id: int
    ad_id: int
    position: int
    page_category: str
    served_by: str = "random-v0"


@app.post("/event")
def click(ev: ClickIn):
    with get_session() as s:
        ad = s.get(Ad, ev.ad_id)
        if not ad:
            raise HTTPException(404, "unknown ad")
        s.add(Event(event_type="click", user_id=ev.user_id, ad_id=ev.ad_id,
                    position=ev.position, page_category=ev.page_category,
                    served_by=ev.served_by, price_paid=ad.bid))
        # Stage 0 billing: charge the bid on click (simple CPC). Real GSP arrives in stage 4.
        ad.spent_today = round(ad.spent_today + ad.bid, 4)
        s.add(ad)
        s.commit()
    return {"ok": True}


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
            by_policy[v] = {"impressions": vi, "clicks": vc,
                            "ctr": round(vc / vi, 4) if vi else 0.0}
    return {
        "impressions": imps,
        "clicks": clicks,
        "ctr": round(clicks / imps, 4) if imps else 0.0,
        "revenue": round(spend, 2),
        "by_position": by_pos,
        "by_policy": by_policy,
    }

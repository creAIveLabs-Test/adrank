"""Incident drills — the oncall game.

Start a drill, DO NOT read this file's output, go stare at /dashboard and
figure out what broke from the metrics alone. Then diagnose out loud,
check your answer with --reveal, and undo with --restore.

Drills:
  bot     a bot population starts clicking ~25% of everything it sees.
          Symptom: CTR spikes, calibration monitor shows actual >> predicted.
  drift   the world changes: every user's interests rotate to new categories.
          Symptom: CTR sags everywhere, calibration shows predicted >> actual
          (the model believes affinities that no longer exist). Fix after
          diagnosis: retrain (the flywheel absorbs the new world).
  stale   a bad model rollout: current artifacts replaced by a sabotaged
          model whose scores are scaled 3x hot. Symptom: CTR roughly holds
          (ordering survives a monotone distortion) but the calibration
          monitor goes red and GSP prices distort — the quiet incident.

Run:  uv run python -m adrank.chaos --drill bot|drift|stale|random
      uv run python -m adrank.chaos --reveal
      uv run python -m adrank.chaos --restore
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from sqlmodel import select

from .db import get_session
from .models import CATEGORIES, SimUser

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "models" / "chaos_state.json"
BACKUP = ROOT / "models" / "chaos_backup"


def drill_bot() -> dict:
    """Bots are injected as traffic, not state: run the simulator with bots."""
    print("Drill armed. Now generate traffic that includes the anomaly:")
    print("  uv run python -m adrank.simulator --sessions 1500 --direct --bot-users 120")
    print("Then watch /dashboard.")
    return {"drill": "bot", "note": "bot traffic via --bot-users 120"}


def drill_drift() -> dict:
    with get_session() as s:
        users = s.exec(select(SimUser)).all()
        backup = {u.id: u.interests for u in users}
        for u in users:
            vec = u.interest_vector()
            shift = random.randint(5, 15)
            u.interests = ",".join(str(v) for v in vec[shift:] + vec[:shift])
            s.add(u)
        s.commit()
    print("Drill armed. The world has changed. Generate traffic:")
    print("  uv run python -m adrank.simulator --sessions 1500 --direct")
    print("Then watch /dashboard.")
    return {"drill": "drift", "user_backup": backup}


def drill_stale() -> dict:
    import joblib

    from .pctr_model import PctrModel

    BACKUP.mkdir(exist_ok=True)
    src = ROOT / "models" / "pctr_model.joblib"
    shutil.copy(src, BACKUP / "pctr_model.joblib")

    model: PctrModel = joblib.load(src)
    # Sabotage: refit the calibrator to run 3x hot. A plain IsotonicRegression
    # stays picklable, so the bad model deploys exactly like a good one would.
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    xs = np.linspace(0, 0.5, 200)
    hot = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=0.95)
    hot.fit(xs, np.clip(model.iso.predict(xs) * 3.0, 0.0, 0.95))
    model.iso = hot
    joblib.dump(model, src)
    print("Drill armed: a bad model version is now live (restart or POST /admin/reload-models).")
    print("Generate traffic:  uv run python -m adrank.simulator --sessions 1500 --direct")
    return {"drill": "stale", "backup": str(BACKUP / "pctr_model.joblib")}


DRILLS = {"bot": drill_bot, "drift": drill_drift, "stale": drill_stale}


def restore() -> None:
    if not STATE.exists():
        print("No active drill.")
        return
    state = json.loads(STATE.read_text())
    kind = state["drill"]
    if kind == "drift":
        with get_session() as s:
            for uid, interests in state["user_backup"].items():
                u = s.get(SimUser, int(uid))
                if u:
                    u.interests = interests
                    s.add(u)
            s.commit()
        print("Users restored to original interests.")
    elif kind == "stale":
        shutil.copy(BACKUP / "pctr_model.joblib", ROOT / "models" / "pctr_model.joblib")
        print("Good model restored — POST /admin/reload-models to hot-swap it back.")
    else:
        print("Bot drill leaves no state — the polluted events remain in the log")
        print("(as they would in production; retraining on them is part of the lesson).")
    STATE.unlink()
    print("Drill cleared.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drill", choices=[*DRILLS, "random"])
    ap.add_argument("--reveal", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()

    if args.reveal:
        print(json.loads(STATE.read_text())["drill"] if STATE.exists() else "No active drill.")
        return
    if args.restore:
        restore()
        return
    if not args.drill:
        ap.print_help()
        return
    if STATE.exists():
        print("A drill is already active — --restore it first.")
        return
    kind = random.choice(list(DRILLS)) if args.drill == "random" else args.drill
    state = DRILLS[kind]()
    STATE.write_text(json.dumps(state))
    if args.drill == "random":
        print("(drill chosen at random — no peeking; use --reveal after you diagnose)")


if __name__ == "__main__":
    main()

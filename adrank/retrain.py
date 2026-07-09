"""Stage 5: the retrain loop — the system improves from its own logs.

Re-fits pCTR (champion selection + calibration) and the two-tower on all
accumulated events, rewrites models/, then hot-swaps the running server.
Schedule it (cron/launchd) nightly, or run by hand:

  uv run python -m adrank.retrain
"""
import httpx

from . import pctr, twotower


def recalibrate(base_url: str = "http://127.0.0.1:8100", last_n: int = 30000) -> None:
    """Light loop: refit ONLY the calibrator on the freshest served traffic.

    Why this exists (the winner's curse): the model serves what it scores
    highest, so its own over-predictions are selected into traffic — a
    calibrator fit on the previous policy mixture always runs slightly hot
    on the new model's own picks. Full retrains shift serving again;
    frequent light recalibration on the freshest window is the standard
    production answer. Run it often; full retrain nightly.
    """
    import json as _json

    import joblib
    from sklearn.isotonic import IsotonicRegression

    from .eval import load_examples
    from .pctr import MODELS_DIR
    from .pctr_model import featurize_serving

    df = load_examples().tail(last_n)
    model = joblib.load(MODELS_DIR / "pctr_model.joblib")
    # Featurize EXACTLY as serving does — saved aggregates, position 0 — or the
    # calibrator is fit on scores serving never produces (training/serving
    # skew; the first version of this function had that bug and made ECE worse).
    aggs = _json.loads((MODELS_DIR / "pctr_aggs.json").read_text())
    cands = [{"user_id": r.user_id, "ad_id": r.ad_id, "ad_category": r.ad_category,
              "bid": r.bid, "context_match": r.context_match}
             for r in df.itertuples()]
    raw = model.predict_raw(featurize_serving(cands, aggs))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=0.5)
    iso.fit(raw, df["clicked"].to_numpy())
    model.iso = iso
    joblib.dump(model, MODELS_DIR / "pctr_model.joblib")
    print(f"calibrator refit on last {len(df)} impressions")
    try:
        httpx.post(f"{base_url}/admin/reload-models", timeout=10).raise_for_status()
        print("server hot-swapped")
    except httpx.HTTPError:
        print("server not reachable — fresh calibrator loads on next start")


def main(base_url: str = "http://127.0.0.1:8100") -> None:
    print("=== retraining pCTR ===")
    pctr.main()
    print("\n=== retraining two-tower ===")
    twotower.main()
    try:
        r = httpx.post(f"{base_url}/admin/reload-models", timeout=10)
        r.raise_for_status()
        print(f"\nserver hot-swapped: {r.json()}")
    except httpx.HTTPError as e:
        print(f"\nserver not reachable ({e}) — models on disk are fresh; "
              "they load on next server start.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate-only", action="store_true")
    args = ap.parse_args()
    recalibrate() if args.calibrate_only else main()

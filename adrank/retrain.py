"""Stage 5: the retrain loop — the system improves from its own logs.

Re-fits pCTR (champion selection + calibration) and the two-tower on all
accumulated events, rewrites models/, then hot-swaps the running server.
Schedule it (cron/launchd) nightly, or run by hand:

  uv run python -m adrank.retrain
"""
import httpx

from . import pctr, twotower


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
    main()

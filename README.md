# AdRank

A miniature ads personalization platform, built end to end: personalized candidate
retrieval, a calibrated pCTR model, a second-price (GSP) auction, exploration, and
an experiments loop — running against simulated traffic with known ground truth,
so every model can be measured not just on CTR lift but on true regret.

```
page view
   |
   v
targeting + budget filter          (SQL)
   |
   v
candidate retrieval                (stage 3: two-tower embeddings + ANN)
   |
   v
pCTR prediction                    (stage 2: LightGBM -> Wide&Deep-lite, isotonic calibration)
   |
   v
auction: ad_rank = bid x pCTR      (stage 4: GSP pricing, budget pacing)
   |
   v
policy: caps, diversity, explore   (stage 5: epsilon -> Thompson)
   |
   v
render + log impression/click ------> events table ------> nightly retrain
                                          |
                                          v
                                   offline eval harness (stage 1)
                                   AUC / LogLoss / ECE / replay CTR / regret vs oracle
```

## Why simulation

Each synthetic user has a hidden interest vector; each ad a hidden quality. The
serving side never reads them — it only sees the event log, like any real ads
system. But evaluation can read them, which means AdRank can compute the oracle
policy and measure true regret. The pCTR architecture is additionally validated
on the public Avazu dataset (~40M real ad impressions).

## Build stages

- [x] Stage 0 — the world: ad inventory, synthetic users, feed UI, event logging, random-serving baseline
- [x] Stage 1 — offline eval harness + baselines (random / popularity / logistic regression / oracle)
- [ ] Stage 2 — pCTR model (LightGBM) + isotonic calibration + Avazu validation
- [ ] Stage 3 — two-tower retrieval + ANN
- [ ] Stage 4 — GSP auction, budget pacing, auction debug panel
- [ ] Stage 5 — experiments platform (bucketed A/B), Thompson exploration, nightly retrain loop

## Run it

```bash
uv sync
uv run python -m adrank.seed                                  # create the world
uv run uvicorn adrank.server:app --port 8100                  # start the ad server
open http://127.0.0.1:8100                                    # browse as a persona
uv run python -m adrank.simulator --sessions 500              # generate traffic
curl -s http://127.0.0.1:8100/stats | python3 -m json.tool    # watch counters
```

## Honest notes

- Traffic is simulated; the simulator is an instrument (ground truth, regret),
  not a claim about scale. Real-data validation happens on Avazu.
- SQLite by design: the demo must run anywhere with zero infrastructure. The
  data layer is SQLModel, so Postgres is a connection-string change.
- Position bias is built into the ground-truth click model on purpose, so the
  modeling stages have something real to correct for.

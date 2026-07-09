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
- [x] Stage 2 — pCTR model (champion selection: LR vs GBDT) + isotonic calibration
- [x] Stage 3 — two-tower retrieval (PyTorch, in-batch negatives + logQ correction, numpy serving export)
- [ ] Stage 4 — GSP auction, budget pacing, auction debug panel
- [ ] Stage 5 — experiments platform (bucketed A/B), Thompson exploration, nightly retrain loop

## Results so far

Offline (time-split test, 36k impressions): random AUC 0.500 · popularity 0.520
· GBDT 0.542 · logreg 0.642 · oracle ceiling 0.760. Lab note: GBDT loses to LR
here — the click function is smooth and near-log-linear in the CTR features, so
LR's inductive bias wins at ~1k positives; at real ads scale the ordering flips,
which is why deep CTR models exist. Reliability diagram: reports/.

Online (live serving): random-v0 CTR 3.07% (36k imps) -> pctr-v1 CTR 6.91%
(9k imps, 2.2x) -> twotower-v2 CTR 6.11% (9k imps). The funnel policy ranks
only ~50 retrieved candidates instead of all 500 eligible ads (10x less
ranking compute) and pays ~12% relative CTR for it — the retrieval/precision
tradeoff, measured. Retrieval recall@50: two-tower 0.36, popularity 0.38,
union 0.39, random 0.12; the logQ sampling correction was worth +0.05
recall@25 (in-batch negatives otherwise punish popular ads).

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

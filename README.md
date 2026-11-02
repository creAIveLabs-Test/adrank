# AdRank
<img src="./AdRank_%20Choosing%20the%20Winning%20Ad.png" width="100%" alt="AdRank shows how an auction selects a sponsored ad for a simulated user.">
AdRank simulates how a platform chooses which ad to show. It finds ads that match a user's interests, predicts how likely the user is to click each one, and combines that prediction with the advertiser's bid to run an auction.

You can compare two strategies on the same simulated traffic: choose the ad most likely to get a click, or choose the ad with the highest bid × predicted click probability. In one A/B test, the auction earned 43% more revenue per 1,000 impressions, while the click-through rate fell from 6.73% to 5.21%.
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
- [x] Stage 4 — GSP auction (ad_rank = bid x pCTR, second-price with reserve) + auction debug panel at /auction
- [x] Stage 5 — experiments platform (hash-bucketed A/B + z-test), Thompson exploration slots, retrain + hot-swap loop

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

Auction (rev per 1k impressions): auction-v3 $76.67 at 5.4% CTR vs pctr-v1
$50.76 at 6.9% CTR vs random $20.43 at 3.1% CTR. Ranking by bid x pCTR
trades ~1.5 points of CTR for +51% revenue per impression — even while GSP
charges second prices instead of full bids. CTR-optimal is not
revenue-optimal; the auction is where that tension is priced. The /auction
panel shows any request's full board (bid, pCTR, ad_rank, price paid) and a
calibration toggle that demonstrates how a hot model misprices clicks.

Experiment (clean window, 13.5k impressions, /experiment): pctr-v1 arm CTR
6.73% at $42.62/1k vs auction-v3 arm CTR 5.21% at $61.12/1k — the CTR drop is
statistically significant (z=-3.67, p=0.0002) and buys +43% revenue. Lab
note: the first readout silently absorbed pre-experiment traffic into both
arms; arm counts now filter on the experiment start time.

The flywheel: after the traffic above, `adrank.retrain` re-fit everything on
its own logs and hot-swapped the server. Two-tower recall@25 went 0.25 ->
0.53 (now well past popularity's 0.36) and pCTR calibration AUC 0.63 -> 0.70
— the system measurably improves itself from data it generated. Thompson
exploration slots (~8% of impressions at ~1.3% CTR) are the tax that keeps
training data honest.

## Feel the job (stage 6 — ops mode)

The dashboard at /dashboard is the working surface: stat tiles (impressions,
CTR, revenue, explore share, calibration ECE with a health pill), CTR and
revenue-per-1k by experiment arm over time, the live experiment readout with
significance, and a calibration monitor (current model vs recent live traffic,
per decile). Bulk traffic: `--direct` runs the funnel in-process (~200k+
impressions accumulate fast).

The oncall game — incident drills via `adrank.chaos`:

```bash
uv run python -m adrank.chaos --drill random   # something breaks. no peeking.
# stare at /dashboard, diagnose out loud, then:
uv run python -m adrank.chaos --reveal         # check your answer
uv run python -m adrank.chaos --restore        # undo
```

Drills: bot traffic (CTR spikes, actual >> predicted), interest drift (CTR
sags, predicted >> actual — fix by retraining), stale model rollout (CTR
holds but calibration goes red and GSP prices distort — the quiet incident).

Incidents already survived for real, in order: calibration drifted after the
flywheel retrain (the winner's curse — the model's own picks select its
over-predictions); the first recalibration made it WORSE (training/serving
skew: calibrator fit on features serving never computes); the skew-correct
`retrain --calibrate-only` took live ECE 0.033 -> 0.004.

## Run it

```bash
uv sync
uv run python -m adrank.seed                                  # create the world
uv run uvicorn adrank.server:app --port 8100                  # start the ad server
open http://127.0.0.1:8100                                    # browse as a persona
uv run python -m adrank.simulator --sessions 500              # generate traffic (HTTP)
uv run python -m adrank.simulator --sessions 5000 --direct    # bulk traffic (in-process)
uv run python -m adrank.retrain --calibrate-only              # light recalibration loop
open http://127.0.0.1:8100/dashboard                          # the ops surface
curl -s http://127.0.0.1:8100/stats | python3 -m json.tool    # watch counters
```

## Honest notes

- Traffic is simulated; the simulator is an instrument (ground truth, regret),
  not a claim about scale. Real-data validation happens on Avazu.
- SQLite by design: the demo must run anywhere with zero infrastructure. The
  data layer is SQLModel, so Postgres is a connection-string change.
- Position bias is built into the ground-truth click model on purpose, so the
  modeling stages have something real to correct for.

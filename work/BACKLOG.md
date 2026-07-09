# AdRank Sprint Backlog

Statuses: TODO / IN PROGRESS / IN REVIEW / DONE. One ticket in progress at
a time. P0 interrupts everything. The mix mirrors a real ads-quality team:
pipelines and infra, model and experiment work, launches, incidents, code
health.

---

## ADR-101 — Add context features to pCTR and run the experiment  [P1] [TODO]
**Type:** model + experiment (the bread-and-butter ML loop)
**Context:** the ranker's features ignore time and page context beyond
category match. Hour-of-day and page-category interactions may carry signal
(users browse topics correlated with interests).
**Task:** design doc first (template). Add 1–3 new features to the pCTR
feature pipeline (eval.py + pctr_model.featurize_serving — keep them
identical, remember the skew incident), retrain, compare offline on the
harness, then A/B the new model against the current one with the
experiments platform and write the analysis doc.
**Acceptance:** design doc reviewed; offline table shows the delta;
experiment ran ≥ 1500 sessions; analysis doc with a ship/no-ship decision.

## ADR-102 — Respond to a page and write the postmortem  [P0] [TODO]
**Type:** oncall
**Task:** say "page me" (or run `chaos --drill random` yourself, blind).
Diagnose from /dashboard only. Mitigate. Write the blameless postmortem
with a timeline and ≥ 2 action items. At least one action item must be
"make this detectable faster" (alerting improvement).
**Acceptance:** correct diagnosis before --reveal; postmortem filed.

## ADR-103 — Budget pacing  [P1] [TODO]
**Type:** infra / control systems
**Context:** campaigns can exhaust daily_budget early (check top spenders'
spent_today vs budget in the DB — some burn out by "morning"). Advertisers
hate this; it also starves late-day auctions of competition.
**Task:** design doc (small). Implement probabilistic throttling in
eligibility: participation probability shrinks as spend runs ahead of a
uniform-day schedule. Add a /admin/new-day endpoint that resets
spent_today. Verify: run traffic, plot/query spend distribution across the
"day" before vs after.
**Acceptance:** measurable smoothing of spend; no CTR regression > noise.

## ADR-104 — Bot filtering in the training pipeline  [P1] [TODO]
**Type:** data quality
**Context:** the bot drill left polluted events in the log (as production
bots do). Bots click ~uniformly; humans click by interest and position.
**Task:** write a detector (per-user CTR z-score, position-insensitivity,
or category-entropy — your call, justify it), exclude flagged users'
events from load_examples() behind a flag, retrain, measure the delta on
clean holdout. Short doc: detector choice + before/after.
**Acceptance:** detector catches planted bots at < 5% false positive rate
on honest users; offline metrics improve or are flat with cleaner
calibration.

## ADR-105 — Launch review: promote the auction to 100%  [P1] [TODO]
**Type:** launch
**Context:** the experiment shows auction-v3 wins revenue at a significant
CTR cost. Someone has to make the call and own the rollout.
**Task:** write the launch doc: metrics summary, guardrails (what CTR
floor is unacceptable?), rollback plan (exact commands), then execute:
end the experiment, set ACTIVE_POLICY, verify on the dashboard, monitor
for one traffic run.
**Acceptance:** launch doc + clean rollout + post-launch dashboard check
noted in snippets.

## ADR-106 — Tests for the money path  [P2] [TODO]
**Type:** code health
**Context:** auction.py computes prices and experiment.py assigns arms —
the two functions where a bug costs money or corrupts an experiment. Zero
tests exist.
**Task:** pytest suite: GSP pricing properties (winner pays ≤ own bid,
≥ reserve, price beats runner-up rank), assignment determinism and
~50/50 balance, calibration monitor label matcher on a synthetic log.
**Acceptance:** `uv run pytest` green; ≥ 8 meaningful assertions; one bug
found OR a property documented that surprised you.

## ADR-107 — Serving efficiency: cache the eligibility pool  [P2] [TODO]
**Type:** performance
**Context:** eligible_ads() loads every ad from SQLite on EVERY request.
At 500 ads it's invisible; at 5M it's an outage. Fix it the way you would
at scale: cache with a freshness bound.
**Task:** measure request latency first (add a timing log or use the
simulator clock). Cache the ad pool with a short TTL; budget checks stay
fresh-ish (document the staleness tradeoff — an ad may serve slightly past
budget; quantify the worst case). Measure after.
**Acceptance:** before/after p50+p95 numbers in the ticket notes; staleness
tradeoff written down.

## ADR-108 — Position bias study  [P2] [TODO]
**Type:** analysis / research
**Context:** ground truth decays 1.0/0.6/0.35 by slot. The logs contain it;
the models mostly ignore it at serve time.
**Task:** measure observed positional CTR ratios from the log, per policy
(selection effects will make them differ from ground truth — explain why).
Write a 1-pager proposing a propensity-weighting fix and what data you'd
need. No implementation required — this is a design exercise.
**Acceptance:** doc with the measured ratios, the selection-effect
explanation, and the proposal.

---

Suggested order: 102 (feel oncall early) → 101 (the core loop) → 105
(make the call) → 103 → 104 → 106 → 107 → 108. Reorder if your manager
(Claude) says otherwise at standup.

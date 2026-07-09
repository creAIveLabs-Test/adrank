# The Work Simulation Protocol

You are an SWE on the AdRank team. Claude plays every other role: manager,
senior reviewer, oncall pager, and (at the end) hiring manager. The goal is
to produce a week of REAL work artifacts — the same ones a Google engineer
produces — and to practice the daily loop until it is muscle memory.

## The rule that makes it work

**You drive.** Claude reviews, questions, and pages — it does not build for
you inside this simulation. If you get stuck, ask for a hint the way you
would ask a senior teammate ("I'm stuck on X, here's what I tried") — not
"do it for me." Everything you can't do without help goes in your snippets
as "learned this week," which is honest and normal.

## The daily loop (run this every work session)

1. **Dashboard first** (2 min): open /dashboard before anything else.
   Anything anomalous? CTR, revenue, calibration pill, explore share.
   Oncall engineers look at the dashboards before their coffee.
2. **Standup** (say "standup" to Claude): three lines — yesterday / today /
   blockers. Claude responds as your manager: priorities, questions.
3. **Ticket work**: one ticket at a time from work/BACKLOG.md.
   Nontrivial tickets require a design doc BEFORE code (see templates/).
   Claude reviews the doc like a senior engineer — expect pushback.
4. **Code review**: when a ticket's code is done, say "review my work" —
   Claude reviews the diff like a Google reviewer. Address comments
   before marking the ticket done. Commit with a real message.
5. **End of session** (2 min): append one line to work/snippets.md —
   what shipped, what you learned, what's next.

## Random events (do not schedule these — ask for them)

- **"page me"**: Claude (or you, blind) runs a chaos drill. You respond
  like oncall: diagnose from the dashboard, mitigate, then write a
  blameless postmortem (template) within the same session.
- **"quiz me on stage N"**: teach-back interrupts, like a colleague
  dropping by to ask how your system works.

## Artifact standards (what the hiring manager sees)

- Design docs: work/design-docs/ADR-xxx-title.md — context, goals,
  non-goals, design, alternatives considered, metrics, rollout plan.
- Experiment analyses: work/experiments/ADR-xxx-analysis.md — hypothesis,
  setup, results table, significance, decision, follow-ups.
- Postmortems: work/postmortems/YYYY-MM-DD-title.md — blameless: impact,
  timeline, root cause, what went well/poorly, action items.
- Snippets: work/snippets.md — one line per session, dated.
- Commits: small, one concern each, messages that explain WHY.

## The finale (when the backlog is done or the interview is near)

Say "demo day": you present the sprint to Claude playing the hiring
manager — 15 minutes: what you shipped, one design doc, one experiment
analysis, one postmortem, the dashboard. Claude grades it and drills the
weak spots. THIS presentation, plus the artifacts, is what you show in
the real interview when they say "tell me how you work."

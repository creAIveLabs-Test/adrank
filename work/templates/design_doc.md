# ADR-xxx: Title
**Author:** Varun · **Status:** Draft / In Review / Approved · **Date:**

## Context
What is true today, and why it's a problem. Link to dashboard numbers or
ticket. Two paragraphs max.

## Goals
Bulleted, measurable. ("Improve offline AUC" is not measurable; "AUC +0.005
on the harness test split" is.)

## Non-goals
What this deliberately does NOT do. (This section is where reviewers decide
you can scope.)

## Design
The change, concretely: what code paths, what data, what interfaces.
A small diagram or before/after table beats prose.

## Alternatives considered
At least one real alternative and why it lost. "Do nothing" is a valid
alternative to discuss.

## Metrics & validation
How you'll know it worked: offline metric + harness, online experiment arm,
guardrails that would abort.

## Rollout & rollback
How it ships (experiment %, flags) and the exact steps to undo it.

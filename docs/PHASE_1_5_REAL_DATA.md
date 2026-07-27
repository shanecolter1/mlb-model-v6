# Phase 1.5 — Real historical data adapter

## What is now implemented

The project can now fetch and normalize real MLB historical game feeds through
the MLB Stats API and translate them into the tables required by the Version 6
historical pipeline:

- games
- plate appearances
- official final-game batting orders
- actual first pitchers used
- final results

A GitHub Actions workflow is included so the fetch can run in an
internet-enabled repository environment.

## Critical limitation preserved

A final historical game feed tells us the lineup and starter that actually
appeared, but it does not by itself prove exactly when that information became
available before first pitch.

Therefore snapshots made solely from these feeds are labeled:

`retrospective_research`

They are suitable for event-engine and venue-feature development, but not for a
claim of fully live-realistic pregame validation.

Promotion-quality pregame backtests require archived lineup, starter, weather,
and market snapshots with timestamps no later than the model run.

## Recommended first sample

Use one completed regular-season month as an engineering sample. After all
normalization and leakage audits pass, expand to multiple seasons.

## Source policy

- MLB game metadata and play outcomes: MLB Stats API
- Venue factors: Baseball Savant
- Retrosheet may later be used as an independent reconciliation source
- No betting odds are used to construct baseball outcome probabilities

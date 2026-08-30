# Historical Baseball As-Of Feature Store

Status: RESEARCH INFRASTRUCTURE

## Objective

Build one reusable, leakage-safe historical baseball feature store that can support I1-I9 scoring models, full-game run models, team totals, pitcher/hitter matchup studies, run-line work, bullpen analysis, and future baseball research without reconstructing pregame statistics separately for every model.

The design principle is simple:

`raw chronological events -> lagged as-of features -> reusable feature store -> model-specific views`

The statistics do not need to have been physically archived before each historical game. They must be reconstructable using only information strictly available before the game being predicted.

## Core storage layers

### Layer A: entity as-of features

One row per entity/date for hitters, pitchers, and teams. Features are generated from observations strictly before the as-of game date.

Examples:
- PA/BF sample support;
- season-to-date event rates;
- trailing 30/90/365-day event rates;
- strikeout, walk, home-run, hit/contact and extra-base-hit rates;
- platoon splits where sample support permits;
- reliability/sample-strength fields;
- days since last appearance and recent workload fields where reconstructable.

### Layer B: game as-of features

One row per game with only pregame-eligible or explicitly flagged retrospective inputs. Includes:
- home/away team IDs and venue;
- announced/actual starter IDs with provenance flag;
- lineup-derived aggregates with provenance flag;
- starter quality and reliability summaries;
- lineup quality summaries;
- pitcher-vs-lineup interaction summaries;
- park/environment keys;
- feature lineage fields.

### Layer C: outcomes

Outcomes are stored separately from predictors to reduce accidental leakage. Includes:
- final score;
- inning-by-inning runs;
- half-inning runs when available;
- winner/margin/total labels;
- model-specific target views generated downstream.

## Leakage rules

1. Every cumulative or rolling feature uses rows with event date strictly earlier than the target game date.
2. Same-day earlier games are not used by default. Doubleheaders therefore use the same pre-day feature state unless an explicitly timestamped intraday research mode is requested.
3. Final-feed lineup and actual-first-pitcher data are permitted only as `retrospective_unverified` unless archived pregame availability is independently established.
4. Future-season park factors, player rates, team rates, injuries, weather, umpire assignments, or roster information are prohibited.
5. Outcomes live in a separate table and are never merged into feature construction before the as-of calculation is complete.
6. All model train/test standardization, shrinkage, and hyperparameter fitting remains downstream and must be fit on training folds only.

## Reuse contract

The feature store is deliberately model-agnostic. It should not contain I2-specific weights or sportsbook-derived probabilities. Model-specific code may derive:
- inning scoring probabilities;
- NRFI/YRFI-style first-inning research;
- I1-I9 full-inning probabilities;
- half-inning distributions;
- first-five/full-game totals;
- team totals;
- run-margin and moneyline baseball probabilities;
- pitcher/hitter matchup residuals;
- bullpen entry/retention models.

## Initial feature families

The first build focuses on event-rate features that can be reconstructed reliably from normalized MLB plate appearances:
- PA/BF count;
- 1B, 2B, 3B, HR, BB, HBP, K and BIP-out rates;
- hit rate;
- extra-base-hit rate;
- on-base-event rate;
- contact-event rate;
- season-to-date and trailing-window versions;
- empirical-Bayes sample support/reliability fields.

Advanced Statcast features can be appended later using the same as-of key contract rather than redesigning the store.

## Output contract

Default path: `data/derived/baseball_asof/`

Primary artifacts:
- `entity_asof.parquet`
- `game_asof_features.parquet`
- `game_outcomes.parquet`
- `feature_dictionary.csv`
- `manifest.json`

The manifest records source files, date range, row counts, build timestamp, leakage rules, and code/feature-schema version.

## Performance

Feature calculation is performed once and persisted. Downstream models consume the saved tables rather than recalculating historical player statistics. This is the key performance improvement that makes all-inning and future baseball analysis practical.

## Governance

This infrastructure inherits `analysis/MODEL_DEVELOPMENT_GOVERNANCE.md`. It is not itself a model and must not promote any predictive component. Any model consuming these features must independently pass chronological out-of-sample validation, calibration, complexity, and market-isolation gates.

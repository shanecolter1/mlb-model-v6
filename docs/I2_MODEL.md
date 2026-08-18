# MLB I2 Under/Over Model — v0.3 Total-Conditioned Research Build

## Objective

Project the full second-inning run distribution before first pitch:

- P(I2 = 0)
- P(I2 = 1)
- P(I2 = 2)
- P(I2 = 3)
- P(I2 >= 4)
- cumulative P(1+), P(2+), P(3+), P(4+)
- fair Under/Over 0.5 American odds

## Mandatory total-conditioned run environment

The official I2 projection must start from the historical I2 scoring prior conditioned on the game's pregame full-game total. The broad all-games I2 rate may not be used as the final prior when a total-specific bucket is required.

Historical benchmark source: 2021–2025 regular-season games reconciled to DraftKings pregame opening full-game totals.

- 10,730 games in total buckets 6.0–11.0
- broad I2 Over 0.5: 4,668 / 10,730 = 43.5042%
- broad I2 Under 0.5: 56.4958%

Exact total buckets are stored in `data/derived/i2/i2_total_conditioned_prior.json`.

Production enforcement:

1. Obtain an approved pregame full-game total through the isolated run-environment interface.
2. Retrieve the exact historical I2 prior for that total bucket.
3. Run the baseball-only I1->I2 simulator.
4. Measure the baseball model's log-odds deviation from the broad historical I2 reference.
5. Apply that baseball-only log-odds deviation to the total-conditioned prior.
6. Reweight the exact positive-run distribution proportionally and shift Top2/Bottom2 score probabilities so they recombine to the conditioned full-I2 probability.
7. If the approved total or exact historical bucket is missing, do not publish or rank an official I2 probability. The game remains pending.

There is **no broad-prior fallback** for an official projection.

### Narrow market-isolation amendment

The pregame full-game total point is now an explicit, narrowly scoped structural input to the I2 predictor because the historical benchmark demonstrates that I2 scoring probability is materially conditioned on the overall run environment.

This is the **only** market-derived input permitted before the I2 probability freeze.

Before freeze, the predictor may receive:

- the full-game total point only
- bookmaker/source identity required to map that point to the historical benchmark
- capture timestamp required for audit

Before freeze, the predictor may **not** receive:

- I2 Under/Over prices
- juice attached to the full-game total
- moneylines
- run lines
- alternate lines
- implied probabilities
- consensus prices
- line movement
- betting percentages
- bookmaker projections or any other market-derived feature

The isolated Netlify function `i2-run-environment` strips price/juice fields and returns only the DraftKings full-game total point plus matchup/time metadata. I2 derivative prices remain prohibited until the total-conditioned baseball probability artifact is frozen.

For live daily operation, the first observed DraftKings pregame total is locked in `data/runtime/i2/YYYY-MM-DD_run_environment.json` and reused on later reruns. `latestObservedTotal` is audit-only and does not replace the locked value. This first-observed live value is not guaranteed to equal the historical archive's exact opening tick, so that distinction must remain visible in the research audit.

## Core architecture

1. **Shared read-only baseball snapshot** — consume the same confirmed lineup, pitcher, Statcast, workload, park, defense, handedness, roof, and weather inputs used by the existing MLB retrieval layer.
2. **Isolated run-environment capture** — capture only the approved full-game total point needed to select the historical I2 prior.
3. **Plate-appearance event engine** — use the repository's existing mutually exclusive 1B/2B/3B/HR/BB/HBP/K/BIP-out event vectors.
4. **I1 state simulation** — simulate the first inning from lineup slot 1 to obtain a distribution of the I2 starting slot and pitches entering I2.
5. **Empirical play-state calibration** — runner advancement, outs added, runs scored, and pitch-count distributions are sampled from 2021–2025 Retrosheet PA transitions conditional on event, outs, and base state.
6. **I2 half-inning simulation** — simulate Top 2 and Bottom 2 separately using the actual batting-order sequence and starter/opener/bulk-pitcher mixture.
7. **Total-conditioned probability transform** — recenter the baseball-only result on the exact full-game-total historical I2 prior while retaining the baseball matchup delta.
8. **Full-I2 distribution** — publish exact and cumulative run probabilities after conditioning.
9. **Prediction freeze** — freeze the total-conditioned I2 probabilities before any derivative-market retrieval.
10. **Market/decision layer** — only after freeze, compare the frozen model with available I2 prices and calculate edge/EV/staking.

## Hard data-source priority rule

For every I2 run, audit, lineup check, starter check, or model update, **the user's own MLB data must be queried before public web sources**. This is a mandatory workflow rule, not a preference.

Source priority is:

1. **Shared/read-only upstream MLB model data layer** — live machine-readable MLB/Statcast/roster/lineup/pitcher data already retrieved by the user's MLB system.
2. **GitHub model/runtime artifacts** — current frozen snapshots, derived datasets, model outputs, and repository data needed by I2.
3. **User Library datasets** — historical joined game data, inning benchmarks, saved odds/history files, Retrosheet-derived data, park-factor files, and other approved model datasets.
4. **Official external baseball sources** — only when the required field is genuinely missing, stale, failed, or needs independent verification after steps 1–3.
5. **Other public sources** — last resort only, and the fallback must be disclosed.

Additional enforcement rules:

- Do not declare a lineup, starter, roster state, or other baseball input unavailable until steps 1–3 have been checked.
- For lineups, the shared machine-readable live MLB feed is authoritative ahead of consumer-facing starting-lineup webpages when it is fresher.
- Public web search must not be used merely for convenience when the same information already exists in the user's MLB data layer, GitHub, or Library.
- Every fallback to an external source must record the missing/stale/failed upstream field and the fallback source used.
- Do not silently substitute one data source for another.
- If sources disagree, preserve the freshest timestamped internal/upstream record, flag the conflict, and verify with an official source rather than automatically replacing the internal value.
- The only pre-freeze market exception is the isolated full-game total point described above. All derivative-market information remains prohibited until the I2 probability artifact is frozen.

## Why I1 -> I2 state is mandatory

I2 is not an average-lineup inning. The hitters most likely to bat in I2 depend on how many plate appearances occurred in I1. In the 2021–2025 regular-season dataset, I2 began at lineup slot 4 in 8,957 of 24,296 team-game observations and slot 5 in 6,952. The model therefore simulates I1 rather than applying whole-lineup OPS/xwOBA indiscriminately.

## Historical state dataset

Source: Retrosheet regular-season CSV packages, 2021–2025.

- 12,148 games
- 24,296 team/game I1->I2 state observations
- no missing I2 starting slots
- same I1 pitcher still pitching at the start of I2: 98.04%
- observed half-inning I2 scoring rate: 25.09%
- observed half-inning I2 mean runs: 0.4581

The committed compact datasets retain only the fields needed for state calibration. Player/event-level source rows are not duplicated in this repository.

## Empirical play calibration

The model removes the initial hard-coded base-running assumptions when `i2_play_calibration.json` is supplied. The calibration is derived from **913,349 regular-season plate appearances** across 2021–2025 and covers all **192 combinations** of the eight model event classes × three pre-PA out states × eight base-occupancy states.

For each state/event combination it stores the empirical probability distribution of:

- outs added
- post-play base occupancy
- runs scored

It also stores empirical pitch-count PMFs by event. This lets I1 pitch-count and I2 run simulations preserve real double plays, sacrifice outcomes, runner advancement, errors/fielder-choice outcomes included in the existing residual BIP class, and other state transitions instead of relying on fixed runner-advance percentages.

## Leakage rule

Historical `i1_pa`, `i1_runs`, `i1_pitches`, and `i2_start_slot` are targets/calibration states, not pregame features. A pregame prediction may not read the realized I1 state. It must simulate I1 from information available before first pitch.

The empirical play-transition table and total-conditioned historical prior are safe for pregame use because they contain pooled historical probabilities rather than realized future state from the game being predicted. Rolling OOS validation must keep all fitted batter/pitcher parameters time-safe.

## Current production status

**RESEARCH BUILD — not yet production betting-approved.**

Implemented:

- I1->I2 state simulation
- exact/cumulative full-I2 simulation
- empirical base/out transition engine
- empirical pitch-count distributions
- opener/bulk mixture hook
- user-data-first retrieval
- mandatory total-conditioned run environment
- no broad-prior fallback
- isolated full-game-total capture with price stripping
- reproducible postmortem diagnostics

The remaining major production blocker is the leakage-safe daily/as-of advanced batter/pitcher feature history. Until that layer is materialized and its interaction weights pass rolling out-of-sample validation, the existing event-engine batter/pitcher interaction weights remain research/shadow parameters.

## Next validation gates

- Re-run historical and live challenger tests comparing broad-prior, total-only, baseball-only, and total-conditioned models.
- Measure Brier score, log loss, calibration, and rank discrimination by total bucket.
- Materialize daily/as-of Statcast 2021–2025.
- Reconstruct as-of batter/pitcher handedness and pitch-type matchup features.
- Fit batter-vs-pitcher event interaction weights on rolling training windows.
- Test 2023, 2024, and 2025 as true unseen seasons.
- Validate start-slot distribution, exact-run distribution, and probability calibration.
- Add complete historical I2 sportsbook-price validation only if reliable timestamped derivative-market history becomes available.

## Retrosheet attribution

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

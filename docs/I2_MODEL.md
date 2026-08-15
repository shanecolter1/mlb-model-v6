# MLB I2 Under/Over Model — v0.2 Research Build

## Objective

Project the full second-inning run distribution before first pitch:

- P(I2 = 0)
- P(I2 = 1)
- P(I2 = 2)
- P(I2 = 3)
- P(I2 >= 4)
- cumulative P(1+), P(2+), P(3+), P(4+)
- fair Under/Over 0.5 American odds

The prediction engine is market-independent. Sportsbook data is not accepted by the I2 module and must only be retrieved after the baseball probability artifact is frozen.

## Core architecture

1. **Shared read-only baseball snapshot** — consume the same confirmed lineup, pitcher, Statcast, workload, park, defense, handedness, roof, and weather inputs used by the existing MLB retrieval layer.
2. **Plate-appearance event engine** — use the repository's existing mutually exclusive 1B/2B/3B/HR/BB/HBP/K/BIP-out event vectors.
3. **I1 state simulation** — simulate the first inning from lineup slot 1 to obtain a distribution of the I2 starting slot and pitches entering I2.
4. **Empirical play-state calibration** — runner advancement, outs added, runs scored, and pitch-count distributions are sampled from 2021–2025 Retrosheet PA transitions conditional on event, outs, and base state.
5. **I2 half-inning simulation** — simulate Top 2 and Bottom 2 separately using the actual batting-order sequence and starter/opener/bulk-pitcher mixture.
6. **Full-I2 simulation** — combine Top 2 + Bottom 2 into exact and cumulative run distributions.
7. **State diagnostics** — compare simulated I2 starting-slot and pitch-count distributions with the 2021–2025 Retrosheet state-transition dataset.
8. **Prediction freeze** — freeze baseball probabilities before any market retrieval.
9. **Market/decision layer** — only after freeze, compare the model with available I2 prices and calculate edge/EV/staking.

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

The model now removes the initial hard-coded base-running assumptions when `i2_play_calibration.json` is supplied. The calibration is derived from **913,349 regular-season plate appearances** across 2021–2025 and covers all **192 combinations** of the eight model event classes × three pre-PA out states × eight base-occupancy states.

For each state/event combination it stores the empirical probability distribution of:

- outs added
- post-play base occupancy
- runs scored

It also stores empirical pitch-count PMFs by event. This lets I1 pitch-count and I2 run simulations preserve real double plays, sacrifice outcomes, runner advancement, errors/fielder-choice outcomes included in the existing residual BIP class, and other state transitions instead of relying on fixed runner-advance percentages.

## Leakage rule

Historical `i1_pa`, `i1_runs`, `i1_pitches`, and `i2_start_slot` are **targets/calibration states**, not pregame features. A pregame prediction may not read the realized I1 state. It must simulate I1 from information available before first pitch.

The empirical play-transition table is safe for pregame use because it contains pooled historical transition probabilities, not future state from the game being predicted. Rolling OOS validation must nevertheless keep all fitted batter/pitcher parameters time-safe.

## Current production status

**RESEARCH BUILD — not yet production betting-approved.**

The I1->I2 state layer, exact/cumulative full-I2 simulation, empirical base/out transition engine, empirical pitch-count distributions, opener/bulk mixture hook, market-isolation rule, reproducible dataset builder, and CI tests are implemented.

The remaining major production blocker is the leakage-safe daily/as-of advanced batter/pitcher feature history. Until that layer is materialized and its interaction weights pass rolling out-of-sample validation, the existing event-engine batter/pitcher interaction weights remain research/shadow parameters.

## Next validation gates

- Materialize daily/as-of Statcast 2021–2025.
- Reconstruct as-of batter/pitcher handedness and pitch-type matchup features.
- Fit batter-vs-pitcher event interaction weights on rolling training windows.
- Test 2023, 2024, and 2025 as true unseen seasons.
- Validate start-slot distribution, exact-run distribution, Brier score, log loss, and probability calibration.
- Add complete historical I2 sportsbook-price validation only if reliable timestamped derivative-market history becomes available.

## Retrosheet attribution

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

# MLB I2 Under/Over Model — v0.1 Research Build

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
4. **I2 half-inning simulation** — simulate Top 2 and Bottom 2 separately using the actual batting-order sequence and starter/opener/bulk-pitcher mixture.
5. **Full-I2 convolution by simulation** — combine Top 2 + Bottom 2 into exact and cumulative run distributions.
6. **State diagnostics** — compare simulated I2 starting-slot and pitch-count distributions with the 2021–2025 Retrosheet state-transition dataset.
7. **Prediction freeze** — freeze baseball probabilities before any market retrieval.
8. **Market/decision layer** — only after freeze, compare the model with available I2 prices and calculate edge/EV/staking.

## Why I1 -> I2 state is mandatory

I2 is not an average-lineup inning. The hitters most likely to bat in I2 depend on how many plate appearances occurred in I1. In the 2021–2025 regular-season dataset, I2 began at lineup slot 4 in 8,957 of 24,296 team-game observations and slot 5 in 6,952. The model therefore simulates I1 rather than applying whole-lineup OPS/xwOBA indiscriminately.

## Current historical state dataset

Source: Retrosheet regular-season CSV packages, 2021–2025.

- 12,148 games
- 24,296 team/game I1->I2 state observations
- no missing I2 starting slots
- same I1 pitcher still pitching at the start of I2: 98.04%
- observed half-inning I2 scoring rate: 25.09%

The committed compact datasets retain only the fields needed for state calibration. Player/event-level source rows are not duplicated in this repository.

## Leakage rule

Historical `i1_pa`, `i1_runs`, `i1_pitches`, and `i2_start_slot` are **targets/calibration states**, not pregame features. A pregame prediction may not read the realized I1 state. It must simulate I1 from information available before first pitch.

## Current production status

**RESEARCH BUILD — not yet production betting-approved.**

The state-transition layer and I1->I2 simulation are now implemented. Full historical fitting of batter/pitcher interaction weights still requires the leakage-safe daily/as-of Statcast history previously identified in the repository data audit. Until that layer is materialized and passes rolling out-of-sample validation, the existing event-engine interaction weights remain research/shadow parameters.

## Next validation gates

- Materialize daily/as-of Statcast 2021–2025.
- Reconstruct as-of batter/pitcher handedness and pitch-type matchup features.
- Fit batter-vs-pitcher event interaction weights on rolling training windows.
- Test 2023, 2024, and 2025 as true unseen seasons.
- Validate start-slot distribution, half-inning exact-run distribution, Brier score, log loss, and calibration.
- Add complete historical I2 sportsbook-price validation only if reliable timestamped derivative-market history becomes available.

## Retrosheet attribution

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

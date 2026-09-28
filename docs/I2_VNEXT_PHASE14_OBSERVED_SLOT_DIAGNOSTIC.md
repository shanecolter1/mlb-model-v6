# I2 vNext Phase 14 — observed I2 starting-slot diagnostic

Status: **RESEARCH ONLY / TARGET-ONLY DIAGNOSTIC**. No live forecast,
production formula, full-I2 calibration, market workflow, or betting rule
changed.

**Comparison scope:** the canonical 2025 point-in-time walk-forward precision
replay predicted **57.1414% Under**, versus **55.1440% realized** (+1.9973
percentage points). The 56.6292% figure below is a *different* static
2024-trained exact research evaluator on the same 2,430 games. Its baseline
is 0.5121 percentage points below the canonical replay. The difference
combines downstream model cadence and exact versus 10,000-trial evaluation;
it has not been separately attributed. The observed-slot diagnostic is
paired only with its own exact static baseline and does not replace the
canonical replay score.

## Question and controlled calculation

Phase 13 showed that the fitted direct I1 PA candidate barely affects full-I2
Under. That result does not show whether a better I1 slot forecaster could
address the residual. This phase tests the fixed downstream I2 model with the
**actual I2 starting batting slots** inserted only after all pregame slot
distributions and scoreless-by-slot probabilities were computed. Actual slots
are available only after I1 and are never eligible pregame predictors.

The 2,430-game 2025 paired calculation uses the same frozen 2024-trained
direct-I2 event model, neutral I1 park treatment, prior-season I2 Savant
profile where available (2,264 games), and empirical transition table as
Phase 13. The game Under outcome is used only for scoring.

| Full I2 Under | Pregame slot distribution | Actual-slot diagnostic | Change |
| --- | ---: | ---: | ---: |
| Mean forecast | 56.6292% | 56.6520% | +0.0228 pp |
| Realized rate | 55.1440% | 55.1440% | — |
| Overprediction | +1.4852 pp | +1.5080 pp | +0.0228 pp |
| Brier | 0.2460874 | 0.2459575 | -0.0001299 |
| Log loss | 0.6852753 | 0.6850143 | -0.0002610 |

Calendar-date paired 95% intervals for the diagnostic minus pregame model
are [-0.0004343, +0.0001613] in Brier and [-0.0008801, +0.0003289]
in log loss. Both include zero.

| I2 half | Realized scoreless | Pregame forecast | Actual-slot diagnostic |
| --- | ---: | ---: | ---: |
| Top | 76.2963% | 75.2706% | 75.2741% |
| Bottom | 73.1276% | 75.2139% | 75.2391% |

The bottom-half scoreless forecast moves **farther** above realization after
the actual batting slot is supplied. The current I1 starting-slot error is
therefore not a plausible standalone fix for this specific Under residual
within the frozen static downstream model. This does not directly quantify
the same diagnostic under the canonical walk-forward replay. Knowing an
observed slot is also not a
formal upper bound on achievable pregame performance: it can carry game
context unavailable before first pitch, while the downstream event and
transition assumptions stay fixed. Do not turn the diagnostic into a
predictor or an adjustment.

## Reproducibility and boundary

- Compact result: `data/derived/i2_vnext/phase14_i2_observed_slot_oracle.json`.
- Optional target-only slot input to the exact evaluator:
  `src/research/run_i2_i1_state_ab_exact.mjs`.
- Paired scorer: `src/research/audit_i2_observed_slot_oracle.py`.
- Target source: `data/derived/i2/i2_state_compact_2025.csv`.

All 2,430 baseline pregame Under probabilities and I1 league/player slot
total-variation values exactly match the independently run Phase 13 replay.
The Phase 14 raw evaluator initially omitted two baseline half-scoreless
fields from its saved per-game rows; these were copied from Phase 13 only
after exact per-game baseline probability parity was checked. The evaluator
now writes those fields directly. The per-game source inputs and predictions
remain outside git; only the compact score is tracked.

The previously inspected 2025 outcomes cannot validate a promoted change.
Phase 7C already tested distinct strikeout/ball-in-play transitions without
resolving the residual; Phase 7D's home-side PA feature showed no clear gain.
The next useful work is independent prospective scoring of frozen pregame
forecasts, with special attention to bottom-half calibration at matched
venues and explicit exclusion of unvalidated temporary sites. No fixed
bottom adjustment or guessed shrinkage is supported here.

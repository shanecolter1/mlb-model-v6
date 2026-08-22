# MLB Pitch-Count State-Dependent Half-Inning Model

Status: research challenger; baseball-only; no sportsbook inputs.

## Objective

Replace heuristic live count adjustments with historically estimated probabilities for the full live state:

`outs × base occupancy × ball/strike count × runs already scored × batter/pitcher context`

The first historical layer is league-state calibration. Player/pitcher/context effects are layered on only after the league state table is validated.

## Primary historical sample

- Seasons: 2021–2025 MLB regular season
- Source target: Baseball Savant Statcast pitch-level feed
- Existing Retrosheet 2021–2025 base/out transition calibration remains in place and is not discarded.
- No market/odds fields are used anywhere in this analysis.

## Canonical count states

12 legal nonterminal counts:

`0-0, 1-0, 2-0, 3-0, 0-1, 1-1, 2-1, 3-1, 0-2, 1-2, 2-2, 3-2`

## Canonical live state

- inning
- top/bottom
- outs: 0,1,2
- base_mask: 0..7
- balls: 0..3
- strikes: 0..2
- runs_already_scored_in_half
- batter
- pitcher
- batter/pitcher handedness where available
- starter/reliever role when available
- pitcher pitch count / TTO when available

For the first challenger, state calibration is fitted on:
`outs × base_mask × balls × strikes`.

## Outcome definition

For every pitch-state observation, calculate **additional runs from that moment until the half inning ends**.

Output:
- P(additional 0)
- P(additional exact 1)
- P(additional exact 2)
- P(additional exact 3)
- P(additional 4+)
- P(additional 1+)
- P(additional 2+)
- P(additional 3+)
- P(additional 4+)
- expected additional runs

For live display, runs already scored are a deterministic floor:
`final half-inning runs = runs already scored + additional runs`.

## Weighting / dependence

Pitch states within the same half inning are correlated. Raw state frequencies are legitimate conditional-frequency estimates, but uncertainty must not pretend each pitch is an independent game.

Therefore:
- store pitch-state observation count N
- store unique half-inning count H
- compute confidence intervals using half-inning clustered bootstrap for production reports
- do not use naive pitch-level standard errors for promotion decisions

## Smoothing

Sparse exact states require hierarchical shrinkage.

Candidate hierarchy:
1. exact `outs × bases × count`
2. `outs × count`
3. `count`
4. all-state baseline

Raw empirical rates are always retained for audit.
No arbitrary monotonic constraints.

## PA outcome layer

Separately estimate from each count:
- strikeout
- walk
- HBP
- single
- double
- triple
- HR
- BIP out
- other

This lets the simulator update **every pitch** through historically estimated PA outcome probabilities instead of hard-coded count multipliers.

## Validation

Walk-forward:
- train 2021–2022 → test 2023
- train 2021–2023 → test 2024
- train 2021–2024 → test 2025

Compare:
1. current heuristic count adjustment
2. count-only historical challenger
3. count + outs challenger
4. full `count × outs × bases` challenger

Metrics:
- Brier: 1+/2+/3+/4+
- log loss
- Ranked Probability Score
- expected-runs MAE/RMSE
- calibration by count
- calibration by outs/base state
- calibration by inning range
- calibration by starter/reliever when added

## Promotion rule

This analysis does not automatically modify the live champion model.

A challenger may replace the heuristic count logic only after:
- out-of-sample improvement
- stable calibration
- adequate sample sizes
- no meaningful subgroup degradation
- reproducible versioned artifacts

## Output artifacts

- `count_state_raw.csv`
- `count_state_smoothed.csv`
- `count_only_summary.csv`
- `pa_outcome_by_count.csv`
- `state_model_metadata.json`
- `validation_report.json`
- `STATE_DEPENDENT_COUNT_MODEL_REPORT.md`

Retrosheet attribution remains required for any Retrosheet-derived supporting data retained in the project.

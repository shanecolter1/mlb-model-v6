# vNext-specific opening-total fit

Research checkpoint, 2026-09-30. This supersedes the *interpretation* of the prior v0.4-formula transfer test; it does not erase that diagnostic. No production forecast or pre-freeze vNext input changed.

## What changed from the transfer test

The earlier experiment transferred v0.4's exact-bucket prior plus a full-strength, fixed logit adjustment to vNext. It was not a vNext-specific training exercise and could double-count information already present in vNext. Here the full-game total effect is estimated **directly from earlier vNext replay probabilities and outcomes**, separately for top and bottom I2. For each half, the candidate is:

`logit(p_half) = logit(vNext_half_probability) + intercept_half + slope_half × (DK_opening_total − 8.5)`.

The total slope receives a fixed ridge penalty of .01. No slope size or penalty is selected on either test season. The corresponding control fits the intercept alone using the identical training rows. The two adjusted half no-score probabilities are multiplied to score the full-I2 Under; unadjusted full I2 uses the archived simulation probability. Opening total is the only sportsbook-derived feature; I2 prices are never used. This is a **post-freeze diagnostic**, consistent with the approved baseball-only vNext input boundary.

The versioned game-level opening-total archive has compressed SHA-256 `8512a4fc2fc8e8566d62080bcf006ab6dbdc953e95c3eb8725b55c47a8dafee5`. Game ID, date, and top/bottom/full observed outcomes must agree exactly with each vNext replay; mismatches fail the test. The archive ends 2025-08-16.

## Chronological cohorts

| Test season | Training seasons | Matched test games | Coverage |
|---|---|---:|---|
| 2024 | 2023 | 2,354 | Mar 28–Sep 30 |
| 2025 | 2023–24 | 1,660 | Mar 27–Aug 16 |

2022 is a pre-rule regime without an earlier vNext replay for this fit. 2023 supplies the first post-rule training season, not an out-of-sample test. 2026 lacks the canonical historical opening-total join and has already been inspected; it is excluded.

## Results

Lower log loss is better. Each season row scores exactly the same games across the three columns.

| Test | Raw full I2 | Intercept-only halves | Intercept + trained total slopes | Total vs intercept, day-cluster 95% interval |
|---|---:|---:|---:|---:|
| 2024 | .685514 | .686948 | .686975 | +.000027 [−.000736, +.000795] |
| 2025 through Aug 16 | .681057 | .681100 | .680726 | −.000374 [−.000932, +.000150] |

Against raw vNext, the trained candidate's paired log-loss difference is **+.001461** in 2024 (95% day-cluster interval [−.000677, +.003671]) and **−.000331** in 2025 ([−.001818, +.001156]). Brier scores and half-inning metrics are in the JSON result. The two total slopes learned from 2023 were +.0480 (top) and +.0421 (bottom) log-odds per total run; adding 2024 changed them to +.0489 and +.0100. Thus a modest vNext-specific total signal is plausible but the observed incremental value is small, variable by season and not established by these two tests.

The training/test years have already been studied in this project, so this is a historical diagnostic, not a pristine holdout. The v0.4 validation release is a separate historical OOS candidate with different features/training; it is not a timestamped frozen-production comparator. We should not promote the total input or claim vNext beats production. A prospective, same-game frozen comparison—including v0.4 and the predeclared vNext total challenger—is the decision gate.

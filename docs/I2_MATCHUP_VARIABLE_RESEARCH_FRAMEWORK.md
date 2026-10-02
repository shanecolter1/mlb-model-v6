# I2 Matchup Variable Research Framework

Status: RESEARCH ONLY / NOT PRODUCTION

## Objective

Use the empirical pregame full-game total as the I2 probability anchor, then allow matchup variables to explain only residual I2 variation among games with comparable pregame totals.

Core specification:

`logit(P(I2 Over)) = logit(P_empirical_total) + matchup_adjustment`

The current production research build in `src/model/i2_total_conditioning.js` is not modified by this framework. Promotion requires separate validation and user approval.

## Development principles

1. The empirical pregame-total model is M0 and remains the benchmark.
2. Exact total buckets are estimated from training data only and shrunk toward the broad training prior when support is limited.
3. Matchup variables are evaluated on residual value after the total anchor; they are not allowed to re-answer whether the game is generally high scoring.
4. Market prices, sportsbook I2 probabilities, EV, line movement, and market agreement are excluded from all baseball-model fitting and selection.
5. No random train/test split. Historical testing is chronological by season.
6. Regularized logistic fitting with a fixed empirical-total logit offset determines variable weights.
7. Candidate families enter sequentially and must improve out-of-sample log loss versus the simpler accepted model.
8. Final selected variables receive leave-one-variable-out ablation tests.
9. Calibration intercept/slope, Brier score, log loss, and total-bucket behavior are reported.
10. The distribution of model-implied matchup probability deltas is measured by pregame-total bucket. This provides the empirical standard-deviation scale for judging whether an adjustment is ordinary or extreme.

## Variable families

The registry is `config/i2_matchup_feature_registry.csv` and currently includes starter quality/reliability, expected I2 hitter quality, I1-simulated I2 batting-order path, platoon/contact/power/baserunner interactions, venue effects, starter-role risk, and guarded weather/umpire candidates.

A variable is not approved because it is intuitively sensible. It must be reconstructable as-of pregame, have sufficient sample support, show stable direction/magnitude, and improve chronological out-of-sample probability quality after already accepted variables are present.

## Testing stages

M0 is empirical total only. Candidate variables then pass univariate residual diagnostics, sequential family gates, ridge-regularized offset fitting, ablation, and matchup-adjustment dispersion analysis. Matchup deltas are reported in percentage points and in bucket-specific standard-deviation units.

## Historical holdout governance

The repository's hard model-development governance remains controlling. 2021–2024 are the principal historical development sample; 2025 is chronological validation evidence to the extent it has not already been consumed; subsequent 2026 games become forward evidence only when the architecture was frozen before results were inspected.

## Required historical feature matrix

The runner expects one row per game with game date, opening pregame full-game total, complete second-inning outcome, and leakage-free pregame candidate features. Matchup fitting must not fabricate unavailable historical pregame features.

## Generalization to all innings

This I2 architecture is now generalized to I1-I9 in `src/analysis/all_inning_matchup_variable_research.py`, with the broader design documented in `docs/ALL_INNING_MATCHUP_VARIABLE_RESEARCH_FRAMEWORK.md` and variables registered in `config/inning_matchup_feature_registry.csv`.

Each inning receives its own empirical total-conditioned baseline, feature eligibility, fitted residual coefficients, calibration diagnostics, feature-family ablation, and matchup-delta dispersion. I1 explicitly fixes the lineup start at slot 1; I2-I5 incorporate pregame batting-order path distributions; I6-I9 require pregame starter-still-active/bullpen mixtures rather than assuming the probable starter remains in the game.

## Promotion gate

No result from either research framework changes the daily model automatically. Historical feature lineage must pass leakage audit, hyperparameters must be frozen, the matchup model must beat its empirical-total-only benchmark chronologically, extreme adjustments must be supported by observed residual dispersion, and sportsbook prices remain post-freeze only.

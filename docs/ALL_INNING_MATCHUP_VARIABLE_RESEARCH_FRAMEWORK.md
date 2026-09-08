# MLB All-Inning Matchup Variable Research Framework

Status: RESEARCH ONLY — does not alter production daily probabilities.

## Objective

Generalize the I2 empirical-total-anchor architecture to innings 1 through 9. For each target inning `j`, the model starts from the empirical probability of at least one run conditioned on the pregame full-game total and inning number, then allows only validated matchup variables to explain residual variation around that anchor.

For inning `j` and game `i`:

`logit(P(run in inning j)_i) = logit(P0(total, inning j)_i) + f_j(X_i)`

`P0(total, inning j)` is estimated only from training-period games and is shrunk toward broader parents when support is thin. `f_j(X)` is the inning-specific matchup residual model.

## Why inning must be explicit

The historical scoring process is not exchangeable across innings. I1 always begins with lineup slot 1 and the starter; I2-I5 have evolving batting-order paths while the starter is usually active; later innings increasingly depend on pregame pitcher-removal/bullpen probabilities and lineup turnover. Therefore each inning receives its own empirical baseline, residual dispersion, feature eligibility, and fitted coefficients.

## Empirical baselines

For every opening total and inning, persist:

- N games reaching the inning;
- P(any run), P(no run);
- exact-run distribution when available;
- shrinkage-adjusted P(any run);
- sampling uncertainty;
- residual matchup-adjustment standard deviation;
- 5th/25th/50th/75th/95th percentiles of validated matchup deltas.

Thin exact-total × inning cells must shrink hierarchically. Preferred parent sequence:

1. exact total × exact inning;
2. adjacent-total pooled × exact inning;
3. all totals × exact inning;
4. broad all-inning prior only as last-resort research fallback.

The amount of pooling/shrinkage is selected chronologically, not by in-sample fit.

## Matchup variable eligibility by inning

### I1

No batting-order-path variable is needed because both teams begin with slot 1. Primary candidates are starter quality/reliability, top-of-order hitter quality, platoon/contact/power/baserunner interactions, and weakly weighted environmental residuals that prove value beyond the game total.

### I2-I5

Add simulated inning-start-slot distributions and their uncertainty. Starter variables remain primary, but starter-still-active probability becomes increasingly relevant. Expected hitters must be weighted by the pregame batting-order-path distribution rather than realized inning participants.

### I6-I9

The model must not assume the probable starter remains active. Pregame pitcher-state mixtures are required: starter-still-active probability plus bullpen quality/state estimates available before first pitch. Realized relief-pitcher identity or earlier-game events are prohibited in a pregame model. Late-inning variables may receive different coefficients from early-inning variables and must prove incremental value independently.

## Development sequence

M0: empirical pregame total × inning baseline only.

M1: + active-pitcher/starter quality and reliability appropriate for inning.

M2: + likely inning hitter quality and batting-order path.

M3: + hitter/pitcher interactions: platoon, contact, power, baserunner.

M4: + starter-removal/bullpen mixture for later innings.

M5: + park/environment/umpire only when they add out-of-sample residual information beyond the pregame-total anchor.

A later model is retained only if it improves the simpler predecessor chronologically on proper probability metrics.

## Validation

Use strictly chronological folds. The repository governance remains controlling: 2021-2024 are development evidence; 2025 is validation to the extent not already consumed; subsequent 2026 games become forward evidence only if the specification was frozen before outcomes were inspected.

Required metrics, overall and by inning:

- log loss;
- Brier score;
- calibration intercept and slope;
- calibration by predicted probability band;
- calibration by opening-total bucket;
- calibration by inning;
- matchup-delta dispersion by opening total × inning;
- coefficient sign/magnitude stability across folds;
- feature-family ablation.

Classification accuracy and historical wagering profit are not model-selection metrics.

## Standard-deviation control of matchup adjustments

For every inning and total environment, estimate the out-of-sample distribution of fitted matchup deltas. Report each prediction as both probability points and a standardized residual:

`matchup_z = matchup_delta / SD(validated matchup deltas for total × inning)`

The SD is a diagnostic and empirical adjustment envelope, not an arbitrary hard cap. Extreme adjustments must be supported by historical out-of-sample evidence. Sparse cells inherit/pool dispersion from the hierarchical parent state.

## Reliability shrinkage

Player and interaction effects must be reliability-weighted. Examples include starter BF/starts, bullpen BF, hitter PA, and lineup-path entropy. Small samples regress toward broader pitcher/hitter/league states. Reliability variables can control shrinkage without necessarily becoming additive scoring predictors.

## Leakage rules

All features must reflect information available before first pitch. Prohibited pregame features include realized inning-start slots, realized pitcher changes, realized pitch counts, actual relief-pitcher identity, runs or baserunners from earlier innings, and any later-updated player statistic that would not have existed at prediction time.

Retrospective final-feed lineups/starters may be used for exploratory feature discovery only when explicitly labeled non-production-eligible. They may not be represented as a clean pregame validation set without archived timing verification.

## Market isolation

Sportsbook prices are prohibited from feature selection, coefficient fitting, shrinkage tuning, calibration, and model acceptance. The only market-derived baseball context allowed by the established architecture is the isolated pregame full-game total point used to select the empirical baseline. I1-I9 inning prices remain post-freeze evaluation data only.

## Required outputs

For each inning 1-9, the research run should produce:

- empirical baseline table by opening total;
- walk-forward M0/M1/... metrics;
- fitted standardized coefficient table by fold;
- univariate residual diagnostics;
- ablation results by feature family;
- matchup-delta SD/quantiles by total;
- calibration tables;
- manifest with data lineage, feature availability, governance status, and holdout-consumption warnings.

A cross-inning summary should compare how much incremental matchup information survives beyond the pregame-total baseline in each inning. This is important: it may turn out that matchup variables are highly useful in I1-I3 and progressively less reliable later, or the opposite. The data—not a fixed inning weighting assumption—determines that conclusion.

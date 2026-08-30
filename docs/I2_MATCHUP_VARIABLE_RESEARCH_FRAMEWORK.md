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

The registry is `config/i2_matchup_feature_registry.csv` and currently includes:

- starter quality: K, BB, HR and non-HR hit rates;
- starter reliability/sample support;
- expected I2 hitter quality;
- I1-simulated I2 batting-order path;
- platoon interaction;
- contact, power and baserunner interactions;
- venue effects, tested only for residual value after total conditioning;
- starter-role/opener risk;
- weather and umpire candidates held behind a stronger redundancy/leakage gate.

A variable is not approved because it is intuitively sensible. It must be reconstructable as-of pregame, have sufficient sample support, show stable direction/magnitude, and improve chronological out-of-sample probability quality after already accepted variables are present.

## Testing stages

### M0 — empirical total only

For each chronological fold, estimate `P(I2 Over | opening total)` using training games only. Sparse exact buckets are Beta-binomial shrunk toward the broad training I2 rate.

### Univariate residual diagnostics

For every available candidate feature, split the held-out season into quantiles and report:

- expected I2 Over from the pregame-total baseline;
- actual I2 Over;
- residual percentage points;
- sample size.

This tests whether the variable explains I2-specific residuals rather than general scoring.

### Sequential family gate

Families are added in registry order. A family is provisionally retained only when mean chronological out-of-sample log loss improves over the previously accepted specification.

The sequential gate is a screening tool, not final causal proof. Correlated-variable interpretation is handled with regularization and ablation.

### Regularized weight estimation

Weights are fitted using ridge-penalized logistic regression with the total-bucket logit fixed as an offset. Feature values are standardized using training data only.

### Ablation

Remove each retained feature in turn and recalculate out-of-sample log loss and Brier score. A retained variable that contributes no independent value is a removal candidate.

### Matchup-adjustment dispersion

For every held-out prediction:

`matchup_delta = final_probability - empirical_total_probability`

Within each opening-total bucket, report the mean, standard deviation, 5th percentile and 95th percentile of matchup deltas. The standard deviation is therefore the dispersion of model-implied conditional matchup probabilities, not the mechanically determined Bernoulli standard deviation of 0/1 outcomes.

Future daily audit output should express a game-level adjustment both in percentage points and in bucket-specific standard deviations.

## Historical holdout governance

The repository's hard model-development governance remains controlling:

- 2021–2024 are the principal historical development sample;
- 2025 has already been used in prior work and is not perfectly pristine, but remains useful as chronological validation evidence;
- a new frozen specification should be evaluated prospectively on subsequent 2026 games not used to redesign the component;
- inspecting a forward result and then redesigning the component consumes that observation as development evidence for that component.

## Required historical feature matrix

The research runner expects one row per game with:

- game date or season;
- opening pregame full-game total;
- complete second-inning runs or separate Top 2 / Bottom 2 runs;
- leakage-free pregame candidate feature columns named in the registry.

The existing joined 2021–2025 master contains the total/outcome foundation, but its release manifest currently identifies the canonical release asset as pending upload. Matchup fitting must not fabricate missing historical pregame features. If no registered feature columns are available, the runner completes the M0 benchmark and returns `BLOCKED_MATCHUP_FEATURES_MISSING` for the matchup stage.

## Outputs

`src/analysis/i2_matchup_variable_research.py` writes:

- `m0_empirical_total_only_folds.csv`
- `univariate_residual_diagnostics.csv`
- `sequential_family_gate.csv`
- `final_walk_forward_folds.csv`
- `final_walk_forward_predictions.csv`
- `final_standardized_coefficients_by_fold.csv`
- `feature_ablation.csv`
- `matchup_adjustment_dispersion_by_total.csv`
- `manifest.json`

## Promotion gate

No result from this framework changes the daily I2 model automatically. Before promotion:

1. historical feature lineage and as-of timestamps must pass leakage audit;
2. the final specification and hyperparameters must be frozen;
3. it must beat M0 on proper scoring/calibration metrics across chronological folds;
4. extreme matchup adjustments must be supported by the observed bucket-specific adjustment distribution;
5. a forward 2026 evaluation must be run on observations not used to choose the specification;
6. only after the baseball probability is frozen may sportsbook prices be used for EV evaluation.

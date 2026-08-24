# MLB Model Development Governance — Anti-Overfitting Rules

Status: HARD GOVERNANCE
Effective: 2026-08-23
Applies to: all empirical parameter estimation, feature selection, calibration, validation, and production promotion for the MLB model.

## Hard rules

1. Chronological isolation. Training data may not use information that would not have been available at prediction time. Random train/test mixing is prohibited when it can leak future information.
2. Out-of-sample requirement. No fitted component may be promoted because it improves in-sample fit. It must improve performance on unseen chronological data.
3. Development / validation / forward-test separation. Default structure is 2021-2024 development, 2025 locked validation/model selection, and completed 2026 games as forward/out-of-time testing. Once 2026 outcomes are inspected for a component, they may not be repeatedly used to tune that same component and still be called a pristine holdout.
4. Freeze before forward evaluation. Architecture, features, transformations, shrinkage rules, and acceptance metrics must be frozen before evaluating the forward-test set.
5. Minimum sample support. Granular empirical states with inadequate support may not be used as raw unsmoothed probabilities. They must shrink to a broader empirical parent state or be flagged as insufficient.
6. Hierarchical shrinkage. Player-specific and matchup-specific estimates must regress toward broader empirical baselines as sample size declines. No arbitrary replacement constants are permitted.
7. Complexity gate. A more complex specification is accepted only when it provides meaningful out-of-sample improvement over a simpler benchmark. In-sample gain alone is insufficient.
8. Regularization. Multi-parameter fitted models must use an appropriate regularization or shrinkage method unless there is a documented reason it is unnecessary.
9. Calibration evaluation. Probability models must be evaluated with proper scoring rules and calibration diagnostics, including Brier score and/or log loss and probability-bucket calibration where sample size permits. Classification accuracy alone is not sufficient.
10. No betting-result tuning. Baseball-model parameters may not be selected or altered to maximize historical wagering profit, EV, sportsbook performance, or market agreement. Betting markets remain isolated until baseball probabilities are frozen.
11. No silent exceptions. If a governance rule prevents or materially limits development, the limitation must be reported explicitly. The rule is not weakened automatically.
12. Exceptions require user decision. A blocked development path may proceed under a temporary exception only after the issue, expected benefit, overfitting risk, and proposed safeguard are presented to the user.

## Required issue reporting

Every model-development analysis should classify governance status as PASS, WARNING, or BLOCKED.

A WARNING is required when, for example, sample size is marginal, validation periods are unusually different from training periods, a requested feature cannot be reconstructed leakage-free, or a holdout has already been partially examined.

A BLOCKED status is required when, for example, future information would leak into training, no genuinely unseen validation data exists for a proposed parameter, a complex model improves only in-sample, or a requested empirical state is too sparse to estimate responsibly without an approved pooling/shrinkage hierarchy.

When WARNING or BLOCKED occurs, report: affected component; rule triggered; why it matters; what development is still valid; recommended remedy; and whether proceeding would consume or contaminate a holdout.

## Current holdout interpretation

2021-2024: primary development/estimation sample.
2025: locked validation/model-selection period to the extent it has not already been consumed by prior design work. Any prior use must be disclosed.
2026 completed games: forward/out-of-time evaluation. Once examined and used to redesign a component, those observations become development evidence for that component and are no longer pristine forward-test observations.

Because 2021-2025 have already been used in prior historical model work, historical cross-validation remains useful but must not be described as a fully pristine final test. The strongest evidence for new components is a frozen specification evaluated on subsequent 2026 games not used in its design.

## Promotion standard

A component is production-eligible only when: data lineage is documented; leakage checks pass; sample support is adequate or an empirical shrinkage hierarchy is defined; the specification is frozen before testing; unseen chronological evaluation is completed; proper scoring/calibration metrics are reported; a simpler benchmark is reported; and no material governance BLOCKED issue remains.

## Committed future bullpen upgrades

These are explicit future production upgrades and must remain on the development roadmap even if the current structural bullpen models pass validation:

1. Authoritative starter identity. Replace the current first-pitcher proxy with a verified starter/game-start indicator from an upstream source such as MLB Stats API or another authoritative roster/game feed. Opener and bulk-reliever games must be identified explicitly rather than inferred from first appearance alone.
2. True numeric pitch-count workload. Add reliable pitcher pitch-count history for recent appearances and rest/workload features. Batters faced remains a valid independent workload feature, but it must never be relabeled or treated as equivalent to pitches thrown.
3. Live warming-pitcher signal. Maintain warming status as a separate live Bayesian update to pitcher-removal and reliever-selection probabilities. Absence of an observed warming signal must never imply that a pitching change cannot occur.
4. Revalidation requirement. When either authoritative starter identity or true pitch-count workload is added, affected bullpen components must be rerun through chronological locked-test, context-calibration, leakage, and p99 latency gates before production promotion.

# MLB Model Development Governance — Anti-Overfitting and Production Rules

Status: HARD GOVERNANCE
Effective: 2026-08-23
Applies to: all empirical parameter estimation, feature selection, calibration, validation, production implementation, and production promotion for the MLB model.

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
11. Hard live-latency requirement. Every production live-model update triggered by receipt of a new pitch state must complete within 1.000 second from model-input receipt to availability of the updated model output. This applies to all production model components participating in a live update.
12. Latency engineering target. Production implementations should target p95 model-compute latency of 500 ms or less and preferably 250-500 ms, preserving headroom for data-ingestion, serialization, network, and UI overhead. The hard acceptance limit remains 1.000 second.
13. Predictive value may not be sacrificed merely to meet latency. Sample reduction, state collapse, probability truncation, Monte Carlo substitution, omitted variables, or other approximations that can change predictive output are prohibited unless separately validated and explicitly approved under governance. Prefer mathematically equivalent precomputation, caching, dynamic programming, vectorization, compiled lookup tables, incremental state updates, and parallelization.
14. Numerical-equivalence gate for acceleration. Whenever a validated research calculation is replaced by a faster production implementation, the accelerated implementation must reproduce the reference probability outputs within a documented tight numerical tolerance on a representative regression suite before promotion. Any material discrepancy is BLOCKED until explained and validated as a deliberate model change.
15. Offline/live separation. Computationally expensive fitting, historical calibration, hyperparameter search, and parameter estimation must occur offline/pregame whenever possible. Live per-pitch execution should consume frozen or incrementally maintained parameters and recompute only state affected by the new pitch.
16. Latency regression is a production blocker. A model change that clears statistical validation but fails the live latency gate is not production-eligible. The latency failure must be surfaced and corrected before promotion.
17. Phased persistence for extensive production runs. Any production-development stage expected to perform at least 50,000 independent row/state/prediction evaluations, at least 10 full model fits/candidate fits, or at least 5 minutes of wall-clock computation must be separated into a persisted workflow phase before the next logically distinct stage begins. Each completed phase must save the exact model/data state, governance result, source/commit identity, and artifacts required for the next phase so a later failure or timeout does not require recomputing completed work.
18. Internal checkpointing for very large phases. A single persisted phase expected to exceed 250,000 independent evaluation units or 10 minutes of wall-clock computation must support resumable internal checkpointing at intervals no larger than 250,000 evaluations or 10 minutes, whichever occurs first, unless the computation is atomic and cannot be safely resumed. An atomic exception must be documented before execution.
19. Phase integrity. Downstream phases must consume artifacts produced by the exact upstream commit/model hash that passed the preceding gate. Re-fitting or silently regenerating upstream model state inside a downstream phase is prohibited unless the upstream phase itself is intentionally rerun and revalidated.
20. No silent exceptions. If a governance rule prevents or materially limits development, the limitation must be reported explicitly. The rule is not weakened automatically.
21. Exceptions require user decision. A blocked development path may proceed under a temporary exception only after the issue, expected benefit, overfitting risk, latency/predictive tradeoff, and proposed safeguard are presented to the user.

## Extensive-run evaluation units

For phased-workflow governance, an independent evaluation unit means one top-level historical observation/state/prediction processed by the model or validator, such as one PA feature/prediction evaluation, one half-inning distribution evaluation, one legal live-state latency evaluation, or another explicitly documented equivalent unit. Low-level arithmetic operations performed inside NumPy, matrix multiplication, recursion, or compiled libraries are not counted individually because they are implementation-dependent and not comparable across engines.

The 50,000-unit threshold is a minimum phase-separation trigger, not permission to combine logically distinct governance gates. Calibration, numerical-equivalence testing, locked validation, forward testing, and production latency validation should remain separate persisted phases whenever they produce independent promotion decisions, even if an individual phase falls below the numerical threshold.

## Required issue reporting

Every model-development analysis should classify governance status as PASS, WARNING, or BLOCKED.

A WARNING is required when, for example, sample size is marginal, validation periods are unusually different from training periods, a requested feature cannot be reconstructed leakage-free, a holdout has already been partially examined, or a proposed production implementation materially reduces latency headroom without violating the hard 1-second limit.

A BLOCKED status is required when, for example, future information would leak into training, no genuinely unseen validation data exists for a proposed parameter, a complex model improves only in-sample, a requested empirical state is too sparse to estimate responsibly without an approved pooling/shrinkage hierarchy, an accelerated implementation fails numerical equivalence, a live production update exceeds the 1.000-second hard limit, or an extensive production run exceeds the phased-persistence threshold without durable checkpointing.

When WARNING or BLOCKED occurs, report: affected component; rule triggered; why it matters; what development is still valid; recommended remedy; whether proceeding would consume or contaminate a holdout; and, for production performance issues, whether the proposed remedy changes predictive outputs or is computation-only.

## Current holdout interpretation

2021-2024: primary development/estimation sample.
2025: locked validation/model-selection period to the extent it has not already been consumed by prior design work. Any prior use must be disclosed.
2026 completed games: forward/out-of-time evaluation. Once examined and used to redesign a component, those observations become development evidence for that component and are no longer pristine forward-test observations.

Because 2021-2025 have already been used in prior historical model work, historical cross-validation remains useful but must not be described as a fully pristine final test. The strongest evidence for new components is a frozen specification evaluated on subsequent 2026 games not used in its design.

## Production latency validation standard

For any component participating in live per-pitch predictions, production promotion requires a reproducible latency benchmark using representative legal live states and realistic production data structures. Report at minimum median, p95, p99, maximum observed model-compute latency, benchmark hardware/runtime, sample count, and whether caches were warm or cold.

The hard pass condition is that all intended live update paths complete within 1.000 second in the approved production configuration. A p95 above 500 ms is a WARNING even if all observations remain under 1.000 second because it leaves insufficient operating headroom. The preferred implementation target is 250-500 ms p95 or faster.

Latency optimization must preserve the same market-isolation rules and predictive-governance rules as the research model. Market data may not be introduced merely as a computational shortcut or fallback for a slow baseball-model calculation.

## Promotion standard

A component is production-eligible only when: data lineage is documented; leakage checks pass; sample support is adequate or an empirical shrinkage hierarchy is defined; the specification is frozen before testing; unseen chronological evaluation is completed; proper scoring/calibration metrics are reported; a simpler benchmark is reported; no material governance BLOCKED issue remains; accelerated implementations pass numerical-equivalence testing; every live per-pitch execution path passes the hard 1.000-second latency requirement; and all extensive production runs complied with phased persistence and checkpointing requirements.
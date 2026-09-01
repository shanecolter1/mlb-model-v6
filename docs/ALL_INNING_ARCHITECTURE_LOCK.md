# MLB All-Inning Prediction Engine — Architectural Lock

Status: GOVERNING RESEARCH ARCHITECTURE

## Scope

The prediction engine targets innings 1 through 9 individually. I2 is not the primary model and is not privileged. For every game, the engine must produce an inning-specific discrete run distribution for I1, I2, ..., I9.

Core outputs per inning:
- P(0 runs)
- P(1 run)
- P(2 runs)
- P(3 runs)
- P(4+ runs)
- derived P(1+), P(2+), P(3+), P(4+)
- derived Under/Over 0.5 and any other inning market only after prediction freeze

## Governing baseline

M0 is the already validated empirical pregame full-game total x inning relationship. It is the locked anchor and must not be replaced by arbitrary smoothing, assumed shrinkage strengths, or a generic broad prior.

For inning i:

logit(P_i) = logit(P_M0(total, inning=i)) + f_i(matchup state)

The matchup residual must be estimated chronologically from baseball-only information.

## Internal modeling unit

The canonical research unit is half-inning level, not game-level I2.

Required keys:
- game_id
- game_date
- season
- inning 1..9
- half = top/bottom
- batting_team_id
- pitching_team_id

Required target fields:
- half_played
- runs_half
- scored_half

Full-inning distributions are formed from the corresponding top and bottom half-inning distributions. A half not played is not equivalent to a played scoreless half.

## Matchup architecture

The model must identify a probability distribution over pitcher x batter states for each inning.

### Pitcher state

For each inning i, estimate pregame:
- P(starter active in inning i)
- P(bullpen active in inning i)
- opener/bulk state when supportable
- bullpen quality/availability state

Do not use realized in-game pitcher changes in a pregame model.

Pitcher-state probability must first be validated against the historical pitcher-state outcome before being judged solely by downstream run prediction.

### Batter/order state

For each inning i, estimate the probability distribution over the batting-order slot that starts the half-inning:
- P(start slot 1), ..., P(start slot 9)

Use the existing empirically reconstructed batting-order path distributions. Preserve the full state distribution; do not collapse it prematurely to one generic lineup score.

Batter-path probability must first be validated against realized historical start-slot outcomes before downstream run-model promotion.

### Matchup assembly

Pitcher state and batting-order state must NOT be assumed independent. Development testing rejected the simple factorization P(pitcher state) x P(start slot), with stable dependence especially in middle and late innings. The matchup layer therefore uses an empirically fitted joint/conditional state distribution:

P(pitcher state, batter/order state | inning i, pregame)

The current development joint-state layer conditions batting-order state on starter-survival/bullpen state with inning-specific dependence strength selected chronologically on 2022-2024 folds. Any future richer dependence structure must also earn promotion empirically.

### Required point-vector AND full-distribution comparison

The matchup state must be tested in two parallel representations from the SAME joint pitcher x batter state distribution. The point-vector branch is a challenger/comparator; it must not silently replace the distributional architecture.

Point-vector branch M4-P:

Xbar_i = sum_z w_z X_z

where z is a latent joint pitcher x batter/order state and w_z is its pregame probability. This branch collapses the state distribution to probability-weighted expected matchup features before the nonlinear event/run transformation.

Full-distribution branch M4-D:

P(Y | inning i) = sum_z w_z P(Y | X_z)

Each latent matchup state must be passed through the fitted event/scoring transformation FIRST, and only then probability-weighted. Do not approximate this branch by evaluating the nonlinear model at the expected feature vector. In general E[f(X)] != f(E[X]).

Ultimately the required distributional run architecture is:

P(R_i=r) = sum_z w_z P(R_i=r | X_z, M0_i)

The M4-D branch must be compared directly with M4-P and locked M0 on identical chronological development/confirmation folds before architecture freeze.

A distribution-plus-dispersion/nonlinear extension M4-D+ may be considered only if the simpler full mixture demonstrates evidence that additional distributional shape information is useful. Variance or higher moments must not be added merely because they are plausible.

### Matchup feature-family selection governance

The current matchup event dimensions are strikeout, baserunner (BB/HBP), home run, and non-HR hit. These dimensions were NOT manually selected as final production weights. They were retained because PA-level chronological M1 validation using strictly prior-date player histories demonstrated predictive signal, and subsequent probabilistic pregame-state fidelity testing showed that the signal survives replacement of realized participants with M2/M3/M4 pregame participant distributions.

Current core dimensions:
- strikeout
- walk/HBP/baserunner
- HR/power
- non-HR hit/contact

These are event primitives rather than an assertion that conventional summary statistics are the optimal final feature set. The research roadmap must include systematic empirical challenger testing of additional pitcher and batter statistics available from the leakage-safe sources, including where available:
- handedness/platoon splits
- pitch mix and pitch-type matchup features
- velocity and velocity stability
- whiff/chase/contact measures
- hard-hit and barrel measures
- expected-stat measures such as xwOBA/xSLG/xBA where historically/as-of reconstructable
- batted-ball quality and launch characteristics
- walk/control measures
- workload/fatigue/recent role variables where they belong to pitcher-state rather than intrinsic skill

No candidate is promoted because baseball intuition says it should matter. Candidate families must be screened chronologically against the relevant PA event targets and then tested for incremental value beyond the existing validated dimensions. Feature selection, transformations, interactions, windows, shrinkage and regularization are development-selected only. 2025 must remain untouched throughout selection.

The same feature family should be available across I1-I9 unless evidence supports an inning-specific exclusion. Coefficients/effects may vary by inning, with partial pooling only if empirically validated.

### Bullpen state representation

When the bullpen branch is active, exact reliever identity is represented probabilistically rather than assumed known. Late-inning first-reliever identity ranking is empirically useful, but early-inning exact-identity stability is mixed. Therefore the matchup engine must support a probability-weighted bullpen skill mixture.

The validated bullpen skill state uses strictly prior-date pitcher histories and probability-weighted K, baserunner, HR, and non-HR-hit skill vectors. Exact identity may be used only where its development evidence supports it; otherwise use the broader skill mixture.

## Development sequence

M0: locked empirical pregame-total x inning baseline.

M1: establish residual pitcher/batter matchup skill dimensions across I1-I9 and systematically test candidate player-stat families for incremental PA-level predictive value.

M2: validate inning-specific pitcher-state probabilities across I1-I9.

M3: validate inning-specific batting-order/batter-state probabilities across I1-I9.

M4: assemble the joint pitcher x batter matchup state distribution and preserve it. Evaluate both M4-P point-vector and M4-D full-distribution branches. Do not interpret a point-vector failure as a distributional matchup failure.

M4 event layer: validate that pregame probabilistic matchup states retain predictive fidelity for PA events before translating them into runs.

M4 run layer: translate validated PA-event matchup information into half-inning and inning run distributions relative to locked M0, testing the full mixture directly.

M5+: add park, environment, umpire, or other variables only if they improve chronological out-of-sample performance beyond the total-conditioned baseline and validated matchup engine.

The previous I2-only M2 and M3 runs are archived as failed specific formulations only. They do not reject pitcher-state or batting-order-state modeling.

## Locked development findings through current M4 research

- M0 fidelity is verified across I1-I9 against the canonical historical outcomes.
- M1 PA-level retrospective matchup-skill testing identified stable predictive signal in K, baserunner, HR, and non-HR-hit dimensions using strictly prior-date histories.
- M2 starter-survival probabilities require empirical calibration/partial pooling; raw pitcher-specific retention probabilities are too extreme.
- M2 calibrated starter-survival models use development-selected history windows/weights, with 2025 untouched.
- M3 raw unconditional inning start-slot distributions remain the governing batter-path representation.
- Recursive transition propagation from I1 through I9 was tested directly against unconditional M3 distributions. It was effectively neutral through I5, trivially positive in I6 and I8, and worse in I7 and I9. It is therefore NOT promoted as a wholesale replacement for the unconditional M3 representation.
- Pitcher state and batting-order state are empirically dependent and must be modeled jointly/conditionally rather than factorized.
- Bullpen skill-mixture fidelity is validated as the preferred fallback when exact reliever identity is uncertain.
- The first integrated matchup residual test collapsed the joint distribution to expected/point-vector features. Its failure does NOT constitute a rejection of the full matchup distribution.
- An inning-specific point-vector residual challenger improved only I2 and I4 on 2024 confirmation and was negative in aggregate; it is not promoted globally.
- Direct PA-event fidelity testing of the probabilistic pregame matchup state remained positive on 2024 confirmation for K, baserunner, HR, and non-HR hit. Therefore useful baseball matchup information survives the M2/M3/M4 uncertainty layer; downstream event-to-run translation remains an active research question.
- The full M4-D matchup-distribution challenger is mandatory before concluding the matchup layer has failed or before architecture freeze.

## Chronological validation and holdout

Development seasons: 2021-2024 only.

Final holdout: 2025 remains unopened during architecture selection, feature selection, hyperparameter tuning, shrinkage tuning, interaction selection, residual scaling, point-vs-distribution selection, and event-to-run architecture selection.

The holdout must not be loaded into model-selection code paths. One frozen final architecture receives the final 2025 evaluation.

## Empirical-validation rule

Everything is empirically validated unless explicitly approved otherwise.

No numeric window, threshold, shrinkage strength, pooling strength, coefficient, interaction, nonlinear transform, reliability curve, feature inclusion, event-engine batter/pitcher blend weight, residual scaling factor, distributional moment, or point-vs-mixture architecture receives production status merely because it is reasonable.

## Market isolation

Before prediction freeze, the only market-derived baseball context allowed is the isolated pregame full-game total point used to select the empirical inning baseline.

No inning prices, juice, moneylines, run lines, alternate lines, implied probabilities, consensus, line movement, betting percentages, or betting commentary may enter feature construction, fitting, calibration, model selection, or prediction.

After predictions are frozen, market prices may be retrieved only for fair-value, EV, and staking decisions.

## I9 and unplayed-half treatment

Bottom 9 is conditional on the game state and may not be played. Historical target construction must distinguish:
- half played and scoreless
- half not played

The all-inning model must explicitly represent P(bottom 9 occurs) when producing a full-I9 distribution. The same principle applies to shortened games and other unplayed halves.

## Existing assets to preserve

Keep and reuse:
- canonical 2021-2025 historical master
- normalized Stats API feeds
- reusable as-of entity/team feature store
- batting-order path artifact
- player platoon artifact
- starter workload/retention artifact
- bullpen artifact
- event/run expectancy primitives where empirically validated

Do not modify existing 1Tap tooling as part of this rebuild.

Do not break the current I2 production pipeline while the all-inning challenger is developed. Build the new engine in parallel. Once validated, I2 becomes projection.innings[2] from the common I1-I9 engine.

## Research namespace

New work should live under all-inning-specific paths where practical, for example:
- src/analysis/all_inning/
- src/model/all_inning/
- data/derived/all_inning/
- data/runtime/all_inning/
- tests/all_inning/

Legacy I2 paths remain operational until explicit promotion/migration.

## Immediate implementation order

1. Build canonical half-inning research matrix for I1-I9 from leakage-safe historical sources. COMPLETE.
2. Verify M0 implementation fidelity across all nine innings. COMPLETE.
3. Generalize M1 matchup-skill research across all innings. COMPLETE FOR INITIAL EVENT DIMENSIONS; BROADER PLAYER-STAT CHALLENGER SCREEN REMAINS REQUIRED.
4. Build and validate pitcher-state model by inning. COMPLETE FOR STARTER SURVIVAL; BULLPEN IDENTITY/SKILL MIXTURE VALIDATED.
5. Build and validate batter/order-state model by inning. COMPLETE; UNCONDITIONAL M3 RETAINED OVER RECURSIVE TRANSITION REPLACEMENT.
6. Assemble joint pitcher x batter matchup state distribution. COMPLETE AT STATE-CONSTRUCTION LEVEL.
7. Run explicit M4-P point-vector versus M4-D full-distribution comparison from the same joint states. IN PROGRESS / REQUIRED GATE.
8. Systematically screen additional pitcher/batter statistic families for incremental PA-event predictive value on 2021-2024 only; integrate only validated additions into both P and D branches.
9. Translate the validated matchup event distribution into half-inning and full-inning P0/P1/P2/P3/P4+ relative to locked M0.
10. Compare chronologically by inning and aggregate; champion remains M0 unless challenger improves according to locked empirical governance.
11. Freeze feature set, state representation, hyperparameters, residual scaling, and distribution architecture.
12. Open 2025 holdout exactly once.
13. Only after prediction freeze retrieve inning markets for fair-value/EV/staking.

## Handoff / new-chat continuity contract

A new chat must read this file before changing or executing the research roadmap. It must not infer project scope from the project title or old I2 files. The engine is I1-I9.

On continuation, first inspect the latest GitHub Actions state on branch `research/all-inning-engine-rebuild`, especially the full-distribution challenger workflow and artifacts. Do not restart completed M0-M3 research. Do not open 2025. Do not modify 1Tap.

The next governing question is whether preserving the complete joint matchup distribution through the nonlinear event/scoring transformation improves upon the point-vector representation and locked M0. The broader player-stat feature screen is also required before final feature freeze.
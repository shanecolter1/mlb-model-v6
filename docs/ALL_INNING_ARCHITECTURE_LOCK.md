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

Use the joint state distribution to construct expected pitcher x batter event characteristics. The common matchup feature family includes, subject to empirical validation:
- strikeout
- walk/HBP/baserunner
- HR/power
- non-HR hit/contact
- platoon

No manually assigned matchup weight receives production status.

### Bullpen state representation

When the bullpen branch is active, exact reliever identity is represented probabilistically rather than assumed known. Late-inning first-reliever identity ranking is empirically useful, but early-inning exact-identity stability is mixed. Therefore the matchup engine must support a probability-weighted bullpen skill mixture.

The validated bullpen skill state uses strictly prior-date pitcher histories and probability-weighted K, baserunner, HR, and non-HR-hit skill vectors. Exact identity may be used only where its development evidence supports it; otherwise use the broader skill mixture.

## Development sequence

M0: locked empirical pregame-total x inning baseline.

M1: establish residual pitcher/batter matchup skill dimensions across I1-I9.

M2: validate inning-specific pitcher-state probabilities across I1-I9.

M3: validate inning-specific batting-order/batter-state probabilities across I1-I9.

M4: assemble probability-weighted joint pitcher x batter matchup states and test incremental residual run signal versus M0/M1.

M5+: add park, environment, umpire, or other variables only if they improve chronological out-of-sample performance beyond the total-conditioned baseline and validated matchup engine.

The previous I2-only M2 and M3 runs are archived as failed specific formulations only. They do not reject pitcher-state or batting-order-state modeling.

## Locked development findings through M4 state research

- M0 fidelity is verified across I1-I9 against the canonical historical outcomes.
- M1 PA-level retrospective matchup-skill testing identified stable predictive signal in K, baserunner, HR, and non-HR-hit dimensions using strictly prior-date histories.
- M2 starter-survival probabilities require empirical calibration/partial pooling; raw pitcher-specific retention probabilities are too extreme.
- M2 calibrated starter-survival models use development-selected history windows/weights, with 2025 untouched.
- M3 raw unconditional inning start-slot distributions remain the governing batter-path representation.
- Recursive transition propagation from I1 through I9 was tested directly against unconditional M3 distributions. It was effectively neutral through I5, trivially positive in I6 and I8, and worse in I7 and I9. It is therefore NOT promoted as a wholesale replacement for the unconditional M3 representation.
- Pitcher state and batting-order state are empirically dependent and must be modeled jointly/conditionally rather than factorized.
- Bullpen skill-mixture fidelity is validated as the preferred fallback when exact reliever identity is uncertain.

## Chronological validation and holdout

Development seasons: 2021-2024 only.

Final holdout: 2025 remains unopened during architecture selection, feature selection, hyperparameter tuning, shrinkage tuning, interaction selection, and residual scaling.

The holdout must not be loaded into model-selection code paths. One frozen final architecture receives the final 2025 evaluation.

## Empirical-validation rule

Everything is empirically validated unless explicitly approved otherwise.

No numeric window, threshold, shrinkage strength, pooling strength, coefficient, interaction, nonlinear transform, reliability curve, feature inclusion, event-engine batter/pitcher blend weight, or residual scaling factor receives production status merely because it is reasonable.

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
3. Generalize M1 matchup-skill research across all innings. COMPLETE FOR DEVELOPMENT SIGNAL IDENTIFICATION.
4. Build and validate pitcher-state model by inning. COMPLETE FOR STARTER SURVIVAL; BULLPEN IDENTITY/SKILL MIXTURE VALIDATED.
5. Build and validate batter/order-state model by inning. COMPLETE; UNCONDITIONAL M3 RETAINED OVER RECURSIVE TRANSITION REPLACEMENT.
6. Assemble pitcher x batter matchup distributions using the joint state representation. NEXT.
7. Fit residual run effects and discrete run distributions chronologically.
8. Freeze architecture.
9. Open 2025 holdout once.

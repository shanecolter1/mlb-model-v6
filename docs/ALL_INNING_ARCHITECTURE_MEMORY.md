# MLB I1-I9 Prediction Engine — Canonical Architecture Memory

Status: LOCKED RESEARCH ARCHITECTURE. This document is the durable source of truth for the all-inning project and supersedes I2-only research interpretations when they conflict.

## Objective

Build one pregame baseball prediction engine that produces an independent discrete run distribution for every regulation inning I1 through I9. I2 is one output of the engine, not the governing scope.

For each inning i, the core output is:
- P(0 runs)
- P(1 run)
- P(2 runs)
- P(3 runs)
- P(4+ runs)
- derived P(1+), P(2+), P(3+), P(4+)
- fair Under/Over 0.5 and any other market derived only after the baseball prediction is frozen.

Internally, model Top and Bottom halves separately and combine them to full-inning distributions. Do not average away/home matchup states before half-inning modeling.

## Locked baseline M0

M0 is the already validated empirical pregame full-game opening-total x inning relationship. It is the champion anchor and is not to be redefined by arbitrary smoothing or a newly assumed prior strength.

For each inning i:

    baseline_i = P_empirical(runs | opening_total, inning=i)

The implementation must faithfully reproduce the validated estimator in leakage-safe chronological development. Any pooling, shrinkage, interpolation, or fallback strength must itself be empirically selected on 2021-2024 development data or explicitly approved.

## Canonical modeling unit

The canonical research matrix is half-inning indexed:

    game_id x inning x half

Required identity keys:
- game_id
- game_date
- season
- inning 1..9
- half = top|bottom
- batting_team_id
- pitching_team_id

Required outcome fields:
- half_played
- half_runs
- half_scored = 1[half_runs >= 1]

Required context:
- opening full-game total
- home/away team identity
- pregame-safe lineup and starter identity timing class
- target availability class

Full-inning probabilities are formed by convolution/combination of the two half-inning distributions, with explicit treatment of unplayed Bottom 9 states.

## State-model architecture

### Pitcher-state model

Before using bullpen or starter-retention information as run predictors, estimate and validate the pregame probability distribution of the pitcher state for each inning.

For each pitching team and inning i estimate, as supported by data:
- P(starter active in inning i)
- P(bullpen active in inning i)
- opener/bulk state where relevant
- later, probability mass over reliever-quality classes or specific relievers only if pregame-identifiable and empirically supported.

Use only pregame-safe inputs such as starter history, BF/start, retention history, rest, role, workload, bullpen usage/availability and team tendencies. Realized pitcher changes, realized pitch counts, actual relief identity and earlier-game events are prohibited in a pregame model.

Pitcher-state accuracy/calibration must be validated as a state-prediction problem before downstream run-model promotion.

### Batter/order-state model

For each batting team and inning i estimate the probability distribution over inning-start batting-order slot:

    P(S_i = s), s=1..9

I1 is deterministic at slot 1. I2-I9 use the empirical prior-date batting-order-path model. Preserve the full distribution; do not collapse it to one expected slot or a small number of generic lineup features before matchup construction.

From each possible start slot, derive the implied sequence of likely hitters. Validate slot/path probability accuracy first, then use the distribution in downstream matchup construction.

### Pitcher x batter matchup distribution

The matchup state for inning i is the probability-weighted joint distribution of:

    pitcher_state_i x batter_path_i

For each possible PA matchup compute common empirically testable dimensions such as:
- strikeout tendency
- walk/HBP tendency
- HR/power
- non-HR hit/contact
- on-base tendency
- platoon advantage
- any higher-order interaction only after empirical validation.

Integrate over state uncertainty rather than pretending the exact future pitcher and batter sequence are known.

## Development layers

M0: locked empirical opening-total x inning baseline.

M1: matchup skill foundation. Identify which pitcher and hitter characteristics explain residual scoring beyond M0, with inning-specific effects and leakage-safe data.

M2: pitcher-state identification across I1-I9. This is not a generic bullpen feature block.

M3: batter/order-state identification across I1-I9. This is not a generic batting-order feature block.

M4: assemble probability-weighted pitcher x batter matchup states and fit only the residual effect beyond M0.

M5+: park, environment, umpire or other residual families only if they add chronological out-of-sample information beyond the existing architecture.

The earlier I2-only M2 and M3 runs are archived as failed specific formulations, not evidence that bullpen or batting-order concepts are useless.

## Residual fitting

Do not generalize the current I2 total-conditioning formula that adds the full raw-baseball logit delta to the empirical prior.

The all-inning challenger must fit the residual directly:

    link(P_i) = link(M0_i) + f_i(matchup_state)

where f_i is estimated chronologically from historical residual outcomes. Magnitude must come from empirical out-of-sample behavior, not from subtracting one internally generated probability from another.

Matchup-delta dispersion should be measured by opening-total x inning environment and reported as both probability-point deltas and standardized residuals. Dispersion is a diagnostic/envelope, not an arbitrary cap.

## Event engine governance

The reusable PA event engine may support simulation, but its default 50/50 batter/pitcher weights are not production-approved. Batter/pitcher event-combination weights, reliability formulas, windows, shrinkage strengths and nonlinear interactions require empirical validation on development data or explicit approval.

## I9 and incomplete-half handling

Bottom 9 not played is not equivalent to Bottom 9 played and scoreless. The target layer must preserve half_played and model P(B9 occurs) explicitly when constructing the full-I9 distribution. The same principle applies to shortened games and any other unplayed half inning.

## Validation and holdout governance

Development evidence: 2021-2024 only.

Final holdout: 2025 remains untouched for architecture, feature, hyperparameter, window, shrinkage, pooling and coefficient selection. No development workflow may intentionally inspect 2025 target outcomes.

Chronological development folds should be walk-forward within 2021-2024. Required metrics include log loss, Brier score, calibration, coefficient/family stability, ablations and state-model calibration by inning.

A feature or layer survives only if it demonstrates chronological out-of-sample improvement and stability under the agreed promotion rules.

## Market isolation

Before prediction freeze, sportsbook data is prohibited except the isolated pregame full-game total point that selects M0. No inning prices, juice, implied probabilities, moneylines, run lines, alt lines, consensus, movement or betting commentary may enter feature construction, fitting or selection.

After the baseball distributions are frozen, the Market/Decision Engine may retrieve inning prices and calculate fair odds, break-even probability, EV and staking.

## Source and artifact policy

Keep and reuse:
- canonical 2021-2025 historical master
- normalized MLB Stats API game/PA/inning tables
- reusable leakage-safe entity/team as-of rates
- batting-order-path artifact
- platoon artifact
- starter workload/retention artifact
- bullpen historical artifact

Archive but do not promote:
- I2-only M1 matrix as the final all-inning architecture
- I2-only M2 bullpen challenger result
- I2-only M3 batting-order challenger result

Do not modify existing 1Tap tooling as part of this work.

## Parallel implementation rule

Do not break or rename the existing I2 production pipeline while the all-inning challenger is under development. Build the new engine in a parallel all-inning namespace. When validated, I2 should become projection.innings[2] from the unified engine.

Preferred namespaces:
- src/analysis/all_inning/
- src/model/all_inning/
- data/derived/all_inning/
- data/runtime/all_inning/
- tests/all_inning/

## Immediate implementation sequence

1. Build and audit the canonical development half-inning matrix for 2021-2024.
2. Reproduce the locked M0 opening-total x inning baseline on that matrix.
3. Build M1 matchup-skill diagnostics across I1-I9.
4. Validate pitcher-state probabilities by inning.
5. Validate batting-order-path probabilities by inning.
6. Assemble pitcher x batter state distributions.
7. Fit residual run models and discrete 0/1/2/3/4+ distributions by inning.
8. Freeze architecture.
9. Open 2025 once for final holdout evaluation.

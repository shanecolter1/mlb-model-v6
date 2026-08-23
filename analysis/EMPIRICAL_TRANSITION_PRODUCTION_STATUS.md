# Empirical PA Transition Engine — Production Status

Status: **PRODUCTION PASS**
Promotion date: 2026-08-23
Production merge: `a13bd7cfcfcd3898cba05eda36088ba7b5a3c261`

## What is now production

The live half-inning `runDistribution()` recursion no longer uses the legacy hand-coded `walkTransition()` or `hitTransitions()` runner-advancement rules. All six modeled PA outcomes (`out`, `bb`, `single`, `double`, `triple`, `hr`) now transition through the empirical state table keyed by:

`event × outs_before × base_mask_before`

Each transition is a probability distribution over:

- outs added
- resulting base occupancy
- runs scored

Generic `out` is the observed-count empirical pool of strikeout and ball-in-play-out outcomes for each outs/base state.

## Data and anti-overfitting split

- 2021-2023: development / estimation
- 2024: shrinkage hyperparameter selection only
- 2025: locked chronological validation
- market inputs used: **false**

Selected hierarchical shrinkage:

- exact state → event+outs prior strength: **5**
- event+outs → event prior strength: **1280**

## Locked 2025 validation

- shrunken transition log loss: **0.2324239968**
- raw unsmoothed transition log loss: **0.2335943284**
- improvement vs raw: **0.0011703315 log-loss points**

Governance result: **PASS**. The shrunken model outperformed the raw state model on unseen 2025 data after shrinkage strength was selected using 2024 only.

## Production regression gate

Before promotion, CI verified:

1. all 144 required model states exist (6 events × 3 outs × 8 base states);
2. every transition distribution normalizes to 1;
3. outs/base/run state values are legal and finite;
4. recursive half-inning distributions normalize across every outs/base starting state under a fixed probability test mixture;
5. empirical advancement materially differs from the legacy fixed runner rules where historical evidence requires it;
6. the generated live `index.html` contains the empirical model and routes `out` as well as hit/walk events through it;
7. legacy `walkTransition()` and `hitTransitions()` functions are absent from the promoted production path.

CI production integration result: **PASS**.

## Files retained

- `data/derived/model_calibration/seasonal/transitions_2021.json`
- `data/derived/model_calibration/seasonal/transitions_2022.json`
- `data/derived/model_calibration/seasonal/transitions_2023.json`
- `data/derived/model_calibration/seasonal/transitions_2024.json`
- `data/derived/model_calibration/seasonal/transitions_2025.json`
- `data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json`
- `data/derived/model_calibration/seasonal/shrinkage_grid_2024.csv`
- `data/derived/model_calibration/seasonal/model_development_manifest.json`
- `analysis/seasonal_transition_shrinkage.py`
- `analysis/live_engine_transition_regression.py`
- `analysis/apply_empirical_transition_patch.py`

## Remaining model-development priorities

This promotion removes the hand-coded runner/base-out transition heuristics. It does **not** make the entire probability engine fully empirical. Remaining high-priority heuristic components include the batter/pitcher event-probability blend, player-rate shrinkage priors, live pitcher deterioration adjustments, pitcher continuation outside the validated early-game scope, and confidence/probability shrinkage. These remain subject to `analysis/MODEL_DEVELOPMENT_GOVERNANCE.md` and must pass chronological out-of-sample gates before production replacement.

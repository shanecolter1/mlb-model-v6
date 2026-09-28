# I2 vNext Phase 13 — direct I1 candidate through full-I2 Under

Status: **RESEARCH ONLY / SHADOW**. No live model, full-I2 calibration,
market workflow, staking, or betting threshold changed.

**Comparison scope:** the canonical 2025 walk-forward precision replay
predicted 57.1414% Under against 55.1440% realized. This phase's 56.6292%
baseline uses a static 2024-trained I2 model with exact evaluation on the
same games. Its scores are paired only within that static research setup;
they do not supersede the canonical walk-forward score.

Phase 14 substituted actual I2 batting-order starting slots as a target-only
diagnostic. The mean Under bias persisted. See
`docs/I2_VNEXT_PHASE14_OBSERVED_SLOT_DIAGNOSTIC.md`.

## Paired 2025 full-I2 replay

The frozen 2022–23 direct first-inning PA model from Phase 12 replaces only
the I1 event probabilities used to forecast each team's I2 starting batting
slot. The baseline retains the existing 50/50 I1 PA formula. Both arms then
use the same exact scoreless-by-start-slot calculation, frozen direct-I2
event model trained through 2024, prior-season Savant venue profiles, and
empirical base/out transitions. I1 park context is neutral in both arms, as
in the Phase 12 slot check. No odds or observed I2 slot enters prediction.

| 2,430 games in 2025 | Existing I1 PA | Direct I1 PA | Direct minus existing |
| --- | ---: | ---: | ---: |
| Mean raw Under 0.5 probability | 56.6292% | 56.6333% | +0.0041 pp |
| Brier | 0.24608740 | 0.24608067 | -0.00000673 |
| Log loss | 0.68527534 | 0.68526136 | -0.00001398 |

The realized Under rate was 55.1440%. Paired calendar-date 95% intervals
for direct minus existing were [-0.00002358, +0.00000977] in Brier and
[-0.00004833, +0.00001934] in log loss. Both cross zero. The candidate's
mean Under forecast is slightly higher despite the existing baseline's
overprediction in this exact setup. The top-half mean scoreless probability
fell from 75.2706% to 75.2661%; bottom rose from 75.2139% to 75.2238%.

**Decision:** retain the existing live formula. The PA-level improvement
does not yield a material full-I2 improvement or repair the Under bias.
These 2025 outcomes were inspected before the candidate was specified, and
the paired intervals omit model-fit uncertainty. This replay uses one
frozen 2024-trained downstream I2 model and exact transitions; it is not the
2025 point-in-time walk-forward, Monte Carlo precision replay and does not
apply a full-I2 calibration. Its scores must not be substituted for that
replay's reported scores.

## Evaluator correction and verification

The older exact I2 evaluator matched Retrosheet team codes directly against
Savant team names and therefore applied **zero** park profiles. We replaced
that lookup with the repo's established Retrosheet **site** mapping. It now
matches 2,264/2,430 games (93.17%). Temporary and unmapped venues remain
neutral. The earlier zero-park run was discarded. The paired scorer rejects
unexpectedly low park coverage.

The I1 slot changes from the corrected exact full-I2 run match Phase 12's
separately generated slot distributions for all 4,860 halves to numerical
precision. This verifies that the integrated arm uses the same I1 candidate
and the only downstream difference between arms is the I1 starting-slot
distribution.

## Reproduction and next boundary

- Compact paired result:
  `data/derived/i2_vnext/phase13_i2_direct_i1_exact_comparison.json`.
- Exact evaluator with optional `--direct-i1-pa-model`:
  `src/research/run_i2_i1_state_ab_exact.mjs`.
- Paired calendar-date scorer:
  `src/research/compare_i2_direct_i1_exact.py`.

The large pregame replay input, Savant profile, and per-game forecasts remain
outside git. They were the same input snapshots used in earlier Phase 9–12
work. No price feed was called. An independently frozen future sample is
needed before reconsidering this I1 arm. The remaining full-I2 bias should
be studied in downstream event probabilities, transitions, and missing venue
coverage with point-in-time inputs; this comparison does not justify a new
shrinkage or side correction.

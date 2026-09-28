# I2 vNext Phase 15 — localize bottom-I2 miss

Status: **SHADOW DIAGNOSTIC ONLY**. No prediction parameter, price workflow,
calibration, or betting decision changed.

This checkpoint uses the **canonical** 2025 point-in-time walk-forward,
10,000-trial replay (mean full-I2 Under 57.14%, realized 55.14%). It scores
the bottom and top halves separately on exactly the same 2,430 games.
Subgroups are descriptive after inspecting 2025; calendar-date intervals
condition on frozen forecasts and do not establish causes.

| Group | Games | Bottom forecast scoreless | Bottom realized | Forecast minus realized | Top error |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 2,430 | 75.56% | 73.13% | +2.43 pp | -0.70 pp |
| Valid prior-season park | 2,264 | 75.53% | 73.01% | +2.52 pp | -1.56 pp |
| Neutral venue | 166 | 75.89% | 74.70% | +1.19 pp | +11.06 pp |
| Matched park, away starter actually began I2 | 2,234 | 75.54% | 73.01% | +2.53 pp | — |

At matched parks, the date-cluster 95% interval for bottom error is
[+0.67, +4.45] points. Bottom-minus-top residual is +4.08 points,
interval [+1.43, +6.81]. The 30 matched games where the away starter
did not begin I2 are too few to estimate a reliable separate effect;
observed continuation is never supplied to the pregame model.

## Risk and calendar localization

Bottom forecasts at matched parks ranged from about 65.8% to 84.7%
scoreless. Quintiles formed *only from pregame probabilities* have bottom
errors of +0.21, +2.45, +2.69, +3.81 and +3.43 percentage points from
lowest to highest predicted scoreless. Individual quintile intervals are
wide and mostly include zero. The direction suggests checking the
highest-scoreless forecasts; it is not evidence for an arbitrary threshold
or calibration curve.

| Matched-park period | Games | Bottom error | Date-cluster 95% interval |
| --- | ---: | ---: | ---: |
| March–April | 420 | +6.51 pp | [+1.71, +11.56] pp |
| May–June | 751 | -2.85 pp | [-5.90, +0.24] pp |
| July–September | 1,093 | +4.67 pp | [+2.35, +7.16] pp |

The sign reversal makes a fixed bottom-half offset especially suspect. The
bottom error persists when the expected starter actually appears, and the
unprofiled-venue problem is concentrated in the **top** half. Neither I1
starting slots nor ordinary starter replacement appears to be a complete
explanation of the matched-park bottom miss.

## Test that can identify a mechanism

The current compact replay contains half-inning outcomes and probabilities,
but not the observed I2 **PA event sequence**. The next diagnostic must
join the frozen monthly event-model version and pregame player/lineup state
to official 2025 I2 play events, then retain the prediction before comparing
the realized event:

1. Score the first three PAs of each I2 half by event class and modeled
   reach. This avoids selecting only longer innings. Separate top and bottom,
   valid parks, and known pregame opener plans.
2. Decompose expected scoreless changes into the PA event mix versus
   conditional base/out-to-run transitions. The latter must be checked by
   observed state and event, without using those observations as predictors.
3. Check pregame pitcher identity/continuation forecasts separately from
   PA talent. Actual continuation is a target-only diagnostic.
4. Fit **one** baseball-only candidate on prior data if a mechanism survives
   these checks. Freeze full-I2 outputs and compare on genuinely new games.

The one-feature home-batting-side PA ablation from Phase 7D was weak, and
the separate K/BIP transition full-I2 ablation from Phase 7C was tiny and
uncertain. This phase does not claim either is the cause or supply a new
correction. The next work needs event-level attribution, not another
constant side adjustment.

Compact result: `data/derived/i2_vnext/phase15_bottom_i2_residual_localization.json`.
Reproduction script: `src/research/audit_bottom_i2_residual.py`. The large
2025 replay and pregame input snapshot remain outside git.

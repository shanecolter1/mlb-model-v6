# I2 vNext Phase 17 — full-season half-inning calibration audit

Status: **RESEARCH ONLY / PREVIOUSLY INSPECTED 2025**. No live model, production calibration, market workflow, thresholds, EV, or staking logic changed.

## Question

Estimate the adjustment, if any, required for the **top of the 2nd** and **bottom of the 2nd** separately. This is a model-residual test, not a raw home/away scoring comparison.

The response variable is whether the half inning scored at least one run. The predictor is the canonical pregame half-inning scoring probability. Calendar month is **not** used as a feature, control, split, or adjustment.

## Full-season matched-home-venue result

The primary diagnostic uses 2,264 2025 games whose ordinary home venue maps to the prior-season Savant profile.

| Half | Predicted scoring | Actual scoring | Actual - predicted |
| --- | ---: | ---: | ---: |
| Top / away batting | 24.3930% | 22.8357% | **-1.5573 pp** |
| Bottom / home batting | 24.4690% | 26.9876% | **+2.5186 pp** |

A logit-offset model allowing a separate half effect estimates:

- top offset: **-0.08679**, date-cluster p = **0.0926**;
- bottom-minus-top offset: **+0.21927**, date-cluster p = **0.0030**;
- implied bottom offset: **+0.13248**.

Thus the evidence supports a genuine **half-specific residual difference**. It does not support treating the two halves as having the same calibration.

## Candidate adjustment

The parsimonious candidate leaves top unchanged and applies only the bottom offset:

```
p_bottom_adjusted = logistic(logit(p_bottom_raw) + 0.1324771)
```

Examples:

| Raw bottom score probability | Candidate adjusted |
| ---: | ---: |
| 15% | 16.77% |
| 20% | 22.20% |
| 25% | 27.57% |
| 30% | 32.85% |
| 35% | 38.07% |

The bottom-only coefficient itself is nonzero with date-cluster robust p = **0.0076**.

A more flexible intercept+slope calibration curve does **not** outperform the simple offset strongly enough to justify the extra degrees of freedom. AIC is 5070.56 for bottom-only, 5069.52 for separate top/bottom offsets, and 5071.31 for the half-specific slope curve.

## Generalization check inside 2025

To avoid simply reporting in-sample fit, the audit uses 20 repeated 10-fold cross-validation runs grouped by complete calendar date. Dates are randomized across folds; month is irrelevant.

For the bottom-only candidate at matched home venues:

- half-inning Brier change vs raw: **-0.000205**;
- half-inning log-loss change: **-0.000603**;
- reconstructed full-I2 Brier change: **-0.000156**;
- reconstructed full-I2 log-loss change: **-0.000328**.

All four date-cluster bootstrap 95% intervals still include zero. Therefore the residual itself is clear, while the forecast-score improvement is not yet independently established.

## Interpretation

The correct current conclusion is:

1. **Top and bottom calibration are not the same.**
2. The principal miss is the model **underpredicting bottom/home I2 scoring**.
3. There is no strong reason to force a top adjustment merely to make the framework symmetric.
4. A **bottom-only logit offset** is the cleanest candidate adjustment.
5. Do **not** promote the +0.13248 offset from 2025, because the 2025 residual was already inspected before this test. It requires independent/prospective validation.

Compact result: `data/derived/i2_vnext/phase17_half_calibration_audit.json`.

Reproducible audit: `src/research/audit_i2_half_calibration.py`.

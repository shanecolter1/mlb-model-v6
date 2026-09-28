# I2 vNext Phases 18–21 — half-inning calibration decision

Status: **VALIDATED HISTORICAL SHADOW CANDIDATE / NOT PRODUCTION**.

This document supersedes the Phase 17 recommendation to consider a bottom-only adjustment. Phase 17 remains a valid description of the inspected 2025 residuals, but the 2024 replication changed the adjustment structure.

## Objective

Estimate and validate top-of-2nd and bottom-of-2nd calibration adjustments directly against the model's own pregame half-inning scoring probabilities.

The governing tests:

- use **actual minus predicted** half-inning scoring, not raw home/away scoring rates;
- use complete seasons, not month controls;
- keep market/price inputs out of the baseball model;
- preserve one final full-I2 calibration layer rather than stacking a common half-level correction and another global correction.

## Phase 18 — 2024 full-season replication

A 10,000-trial point-in-time replay was built for 2,429 games in 2024. The ordinary matched-home-venue sample contains 2,421 games.

| Half | Predicted scoring | Actual scoring | Actual - predicted |
| --- | ---: | ---: | ---: |
| Top / away batting | 25.5757% | 24.0397% | **-1.5361 pp** |
| Bottom / home batting | 25.5694% | 26.8897% | **+1.3203 pp** |

The separate half-offset fit estimates:

- top logit offset: **-0.0828131**;
- bottom logit offset: **+0.0685976**;
- bottom-minus-top offset: **+0.1514106**, date-cluster p = **0.01618**.

The bottom-only offset is not independently significant in 2024 (p = **0.1335**). Therefore the replicated signal is the **difference between halves**, not a stable positive bottom-only intercept.

Historical caveat: the governed vNext hyperparameters (C = 0.05, 730-day half-life) were originally selected using 2024 H2. Phase 18 is replication evidence for the half residual, not a pristine untouched model holdout.

## Phase 19 — zero-sum half contrast

Decompose the 2024 fitted offsets into:

```
common = (top_offset + bottom_offset) / 2
h      = (bottom_offset - top_offset) / 2
```

For 2024:

- common component = **-0.0071077359**;
- half contrast `h` = **0.0757053178**.

The common component is near zero and is **not** applied at the half-adjustment stage. The candidate is:

```
top_adjusted    = logistic(logit(top_raw)    - h)
bottom_adjusted = logistic(logit(bottom_raw) + h)
h = 0.0757053177820764
```

This keeps the half adjustment zero-sum on the logit scale and leaves common/global calibration to the final full-I2 calibration layer.

### Cross-season validation: fit 2024, test 2025 unchanged

Matched ordinary home venues in 2025: **2,264 games**.

| Half | Raw actual - predicted | After 2024-fitted contrast |
| --- | ---: | ---: |
| Top | -1.5573 pp | **-0.1950 pp** |
| Bottom | +2.5186 pp | **+1.0990 pp** |

Proper-scoring changes, adjusted minus raw:

| Metric | Delta | Date-cluster 95% interval |
| --- | ---: | ---: |
| Half Brier | **-0.0003773** | [-0.0007507, -0.0000007] |
| Half log loss | **-0.0010163** | approximately zero-bound on upper end |
| Full-I2 Brier | **-0.0001519** | [-0.0002501, -0.0000505] |
| Full-I2 log loss | **-0.0003099** | [-0.0005118, -0.0001013] |

The 2024-fitted contrast therefore improved 2025 full-I2 Brier and log loss with date-cluster intervals below zero. 2025 was not used to estimate `h`.

## Phase 20 — shadow implementation verification

The governed artifact is:

`data/derived/i2_vnext/i2_vnext_half_contrast.json`

The shadow runner order is:

1. simulate raw top and bottom I2 score probabilities;
2. preserve the raw probabilities for audit;
3. apply the zero-sum half contrast only when the ordinary venue has a matched Savant profile;
4. reconstruct the half-adjusted full-I2 Under probability;
5. apply the **single final full-I2 calibration layer**;
6. keep the result shadow-only and betting-ineligible.

The JavaScript adapter used by the shadow runner was replayed against all 2,264 matched 2025 games and matched the Phase 19 Python research result numerically.

The September 29, 2026 next-slate smoke run remained fail-closed because pregame inputs were not yet frozen:

- remaining games: 4;
- frozen projections: 0;
- pending probable starter: 3;
- pending confirmed lineup: 1;
- betting-eligible outputs: 0.

This is an input-state result, not a failure of the half-adjustment implementation.

## Phase 21 — prospective cohort isolation

The prospective scorer is now `i2-vnext-prospective-score-v2`.

It:

- selects the latest valid pregame snapshot for each game;
- then restricts the report to **one latest declared prospective-validation cohort**;
- excludes prior calibration cohorts instead of pooling them;
- fails closed if multiple calibration versions share the latest validation start date;
- reports both adjusted and raw top/bottom metrics;
- reports the half-adjusted versus raw full-I2 contribution separately.

Current cohort:

- half calibration version: **i2-vnext-half-contrast-v1**;
- prospective validation start: **2026-09-28**;
- `h = 0.0757053177820764`.

Because the 2026 regular season ended before prospective collection began, postseason observations can be archived as prospective evidence but should not be treated as a substitute for a full regular-season validation sample.

## Current decision

**Governing half-inning adjustment candidate**

```
TOP I2:    logit(p) - 0.0757053177820764
BOTTOM I2: logit(p) + 0.0757053177820764
```

Scope:

- matched ordinary home venues only;
- baseball-only, market-isolated;
- common calibration component withheld;
- final full-I2 calibration remains a separate single layer;
- shadow only;
- no production promotion yet.

Do not implement the superseded Phase 17 bottom-only offset.

Primary checkpoints:

- `data/derived/i2_vnext/PHASE18_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE19_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE20_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE21_CHECKPOINT.json`

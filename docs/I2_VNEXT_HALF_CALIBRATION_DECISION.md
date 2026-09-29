# I2 vNext Phases 18–21 and 19B–19D — half-inning calibration decision

Current decision: **NO HALF-INNING ADJUSTMENT** for the full-I2 betting model.
The 2024-fitted contrast is archived in the retired artifact and is disabled
in the vNext shadow runner; its favorable 2025 result was superseded by the
2022–2025 development and untouched 2026 full-season comparison below.
Production remains unchanged.

The Phase 17 bottom-only proposal and the later 2024-fitted zero-sum proposal
describe historical experiments. The five-season decision below governs the
current model choice.

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

The historical shadow-runner experiment used this order:

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

The originally declared cohort was:

- half calibration version: **i2-vnext-half-contrast-v1**;
- prospective validation start: **2026-09-28**;
- `h = 0.0757053177820764`.

Because the 2026 regular season ended before prospective collection began, postseason observations can be archived as prospective evidence but should not be treated as a substitute for a full regular-season validation sample.

## Phase 19B–19D — five-season decision

The recovered replay used 10,000 trials per game, no prices or month controls,
and each team's season-specific primary home venue. Development used 2022–2025;
2021 supplied the prior training/transition foundation, and 2020 supplied
arsenal history only. The completed 2026 regular season was held out from
candidate coefficient fitting and shrinkage selection. The 2026 Stats API
manifest contained 2,459 records for 2,430 unique regular-season games; 29
duplicate game IDs were removed before as-of player updates. Of 2,425 eligible
replays, 2,414 were at primary home venues.

| Season | Primary-home games | Top actual − predicted | Bottom actual − predicted |
| --- | ---: | ---: | ---: |
| 2022 | 2,427 | −2.19 pp | +1.11 pp |
| 2023 | 2,425 | +2.48 pp | +1.22 pp |
| 2024 | 2,421 | −0.92 pp | +1.93 pp |
| 2025 | 2,426 | −0.58 pp | +2.48 pp |
| 2026 | 2,414 | +0.25 pp | +0.92 pp |

The full-season 2023 top residual changed sign. The previously chosen 2024
contrast of ±0.0757053 logit units was therefore not treated as a general
correction. A 2022–2025 fit yielded `h = 0.0531313`; development-only
leave-one-season-out selection shrank it by 0.6404547 to an applied contrast
of **±0.0340282** for the untouched 2026 test.

The candidates were specified before the corrected 2026 replay completed.
The governing rule chose the lowest 2026 full-I2 log loss, with Brier as a
tie-break. Full-I2 probabilities retained each game's raw joint-zero
dependence factor when the half probabilities were changed.

| 2026 candidate versus raw | Full-I2 log-loss change | Full-I2 Brier change |
| --- | ---: | ---: |
| No half adjustment | **0** | **0** |
| Zero-sum contrast | +0.0000282 | +0.0000143 |
| Bottom-only offset | +0.0002974 | +0.0001477 |
| Separate affine-logit curves | +0.0004504 | +0.0002217 |

Positive changes are worse. In 2026 the shrunk zero-sum candidate reduced the
bottom scoring residual from +0.92 to +0.30 pp, but increased the top residual
from +0.25 to +0.87 pp. Raw full-I2 Under 0.5 averaged 57.337% versus 57.249%
observed; the zero-sum candidate averaged 57.325% and worsened both proper
scores slightly. These data do not support adding a half adjustment to the
full-I2 betting model. The exact selected value is
`PHASE19D_CHECKPOINT.json:selected_half_adjustment = NO_HALF_ADJUSTMENT`.

This selection compares candidate forms on one held-out season and does not
establish that every future half adjustment is harmful. The 2022–2025
development seasons also test a fixed model specification chosen partly with
2024 data, so they are robustness evidence rather than pristine historical
holdouts. The 2024→2025 shadow artifact remains available for its originally
declared prospective comparison; the current artifact has `enabled: false`
and retains that original specification under `retirement.prior_artifact`.

## Prior shadow candidate (superseded for model selection)

**Frozen 2024-fitted half-inning experiment**

```
TOP I2:    logit(p) - 0.0757053177820764
BOTTOM I2: logit(p) + 0.0757053177820764
```

Scope:

- matched ordinary home venues only;
- baseball-only, market-isolated;
- common calibration component withheld;
- final full-I2 calibration remains a separate single layer;
- retired from the shadow runner and betting-ineligible;
- not selected by the 2026 full-I2 holdout.

Do not implement the superseded Phase 17 bottom-only offset.

Primary checkpoints:

- `data/derived/i2_vnext/PHASE18_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE19_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE20_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE21_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE19B_CHECKPOINT.json`
- `data/derived/i2_vnext/PHASE19D_CHECKPOINT.json`

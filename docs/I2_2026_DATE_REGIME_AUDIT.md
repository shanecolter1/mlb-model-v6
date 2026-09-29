# I2 2026 date regime audit

The reported mid-2026 decrease in baseball drag motivated an exploratory
date check. The comparison uses **every game** at a primary home venue in
the archived, market-isolated, 10,000-trial I2 replay. It does not condition
on a ball being put in play, a fly ball, or a home run. Each prediction was
made from the historical pregame inputs. The split is **May 25**, the reported
start of the carry change; it was not learned from the I2 outcomes.

Full I2 Under 0.5, 2026:

| Dates | Games | Actual Under | Raw predicted Under | Actual minus predicted |
| --- | ---: | ---: | ---: | ---: |
| Before May 25 | 789 | 59.82% | 57.12% | +2.71 pp |
| May 25 onward | 1,625 | 56.00% | 57.44% | -1.44 pp |

The post-minus-pre residual moved **-4.15 percentage points**. A 95% day
cluster bootstrap interval is **-8.35 to +0.14 pp**. The observed I2 Over
rate rose, but this interval narrowly includes zero; it does not establish
a discrete change at May 25 or identify the baseball as the cause.

The same calendar-date split in the earlier available seasons (post-minus-pre
Under residual, with 95% day cluster interval) was: 2022 **+2.71 pp**
[-1.13, +6.43], 2023 **-0.77 pp** [-5.03, +3.59], 2024 **+1.43 pp**
[-2.18, +5.19], and 2025 **-2.25 pp** [-6.61, +2.08]. The 2025 direction
shows that a date movement can arise without the reported 2026 carry event.

Top and bottom scoring probabilities in 2026 changed from actual-minus-predicted
residuals of **-0.90 and -0.43 pp** before the cutoff to **+0.82 and +1.58 pp**
after it. The Under movement is therefore not confined to one half.

The fixed, non-overlapping date windows show how the full I2 Under residual
developed. These are descriptive windows, not fitted curve points:

| 2026 dates | Games | Actual Under | Predicted Under | Residual |
| --- | ---: | ---: | ---: | ---: |
| Mar 25–Apr 21 | 352 | 60.51% | 57.11% | +3.40 pp |
| Apr 22–May 19 | 372 | 59.95% | 57.06% | +2.89 pp |
| May 20–Jun 16 | 358 | 57.54% | 57.46% | +0.09 pp |
| Jun 17–Jul 14 | 351 | 57.26% | 57.42% | -0.15 pp |
| Jul 15–Aug 11 | 357 | 57.42% | 57.30% | +0.13 pp |
| Aug 12–Sep 8 | 376 | 54.52% | 57.46% | -2.94 pp |
| Sep 9–Sep 27 | 248 | 52.02% | 57.66% | -5.65 pp |

The largest residuals occur late, rather than as a stable step immediately
after May 25. The 2026 season-average near-zero calibration masks opposing
early and late residuals. This is a legitimate warning for a current betting
model, but an in-sample date curve would use the same games to discover and
fit the pattern. **No production adjustment is selected by this audit.**

Reproduce from the five Phase 19C season ZIP artifacts with
`src/research/audit_i2_date_regime.py`. The exact machine-readable metrics,
including Brier and log loss, are in
`data/derived/i2_vnext/i2_date_regime_audit.json`.

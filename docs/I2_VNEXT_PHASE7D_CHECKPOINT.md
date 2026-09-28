# I2 vNext Phase 7D checkpoint — 2026-09-27

Status: **SHADOW ONLY**. No prediction formula, full-I2 calibration, production
workflow, pricing workflow, or betting threshold was promoted.

## Phase 7C transition result

The paired 2,430-game, 10,000-trial replay finished. Separating strikeout
and ball-in-play out transitions reduced full-I2 Brier by 0.000080 and log
loss by 0.000155 over the pooled transition control. The 95% paired calendar
date intervals both include zero. In later 2025, the predicted Under rate is
57.49% versus 53.99% realized. The correction does not solve the bias.

Source: `data/derived/i2_vnext/phase7c_transition_ablation.json`. The
2025 sample was already used for calibration selection, so this is a
descriptive ablation, not a new promotion test. The distinct transition
artifact remains research-only.

## Earlier-season half-inning outcome audit

`src/research/audit_i2_half_asymmetry.py` pairs top and bottom second innings
in the existing Retrosheet compact files. Two suspended games with halves on
different dates are excluded. Across 9,716 games in 2021–2024, top second
innings were scoreless 75.94% of the time and bottom second innings 73.99%:
a bottom-minus-top gap of -1.95 percentage points. A descriptive paired date
cluster interval is [-3.16, -0.73] points. The gap reversed in 2023 (+1.23
points), so a fixed offset is not supported by every season.

Among the 9,327 games in which both starters began the second inning, the gap
was -1.97 points. When both halves began at the same lineup slot (2,544
games), the gap was -2.71 points. Those are *observed* post-first-inning
groups used for diagnosis only. The 2025 gap was -3.17 points, and its
outcomes were already inspected when the hypothesis was formed. This audit
does not establish a model residual on 2021–2024 or isolate a home-side cause.

Source: `data/derived/i2_vnext/phase7d_half_asymmetry.json`.

## One-feature PA component ablation

`src/research/ablate_i2_home_side_pa.py` trains the existing joint multinomial
PA model on 40,812 Statcast I2 PAs from 2023–2024, then trains the same model
with one categorical home batting-side term. Both use the previously selected
730-day half-life and C=0.05, with the same prior-season arsenal features.
On 20,188 2025 PAs, the feature reduces log loss by 0.000135 and multiclass
Brier by 0.000043. The descriptive paired date intervals are
[-0.000393, +0.000126] and [-0.000150, +0.000063], respectively. Both
include zero. This weak PA component result is insufficient to add the
feature or infer a full-I2 improvement. No artifact was promoted or refit for
the live runner.

Source: `data/derived/i2_vnext/phase7d_home_side_pa_ablation.json`. The
2025 outcomes were already inspected and the live shadow model has monthly
walk-forward refits; this is component research only.

## Venue safety in the live shadow runner

The live vNext runner now applies Savant factors only when the normalized
actual game venue uniquely matches the profile venue name. It will not borrow
the ordinary home-club park for an unmatched temporary or neutral venue.
An unmatched/ambiguous park stays an explicit neutral fallback and remains
ineligible for betting. The rule and match status are written to the audit.
Three targeted venue tests pass, including a relocated home club and an
unmatched temporary venue.

## Next validation boundary

Keep the existing canonical vNext forecast in shadow mode. The current
data do not support adding a home batting-side coefficient or a fixed
bottom-half adjustment. Collect immutable pregame forecasts and subsequent
results on future games, preserving venue and lineup/source status, and
compare against a same-cutoff baseball-only baseline with Brier, log loss,
calibration, and date-cluster uncertainty. Model changes prompted by the
2025 diagnostics must be judged on later independent games before any
promotion decision.

# I2 vNext Phase 12 — integrated I1 PA research candidate

Status: **RESEARCH ONLY / SHADOW**. No production or live vNext model,
full-I2 calibration, price workflow, or betting rule changed.

## Candidate and chronological design

Phase 11 found a repeatable top/bottom event-mix residual in the opening
three first-inning PAs. This phase fitted a **single multinomial I1 PA
event model** using strictly pregame batter and pitcher event rates, a
prior-season league event offset, and an event-specific home batting-side
term. The shared batter/pitcher coefficients and event terms are estimated
jointly by maximum likelihood. The model replaces the old 50/50 PA formula
in the research arm; there is no additional shrinkage or separate PA
calibration curve.

Training used 2022 opening PAs for a 2023 chronological check, then 2022–23
for a frozen 2024 check. The exported research model contains only the
2022–23 parameters, training metadata, and event names; it contains no
2024 outcomes. First-three batter and starter identities had to match the
pregame snapshot. Six 2022, two 2023, and five 2024 games were excluded
for opening identity changes.

| Opening-PA check | PAs | Existing PA log loss | Direct, no home | Direct with home |
| --- | ---: | ---: | ---: | ---: |
| Train 2022, score 2023 | 14,568 | 1.505221 | 1.501287 | **1.500199** |
| Train 2022–23, score 2024 | 14,544 | 1.493688 | 1.489698 | **1.488989** |

The home feature's incremental 2023 log-loss difference from the direct
no-home model was -0.001088, with a paired calendar-date 95% interval of
[-0.002010, -0.000174]. In 2024 it was -0.000709, interval
[-0.001520, +0.000061]. The full direct-with-home model improved 2024
multiclass Brier from 0.707494 to 0.705437 and log loss from 1.493688 to
1.488989 versus the old PA formula, with a paired log-loss interval
[-0.006309, -0.003182]. These intervals condition on fitted parameters
and are exploratory: the hypothesis and model form followed inspection
of historical outcomes.

## Does the PA gain reach the I1 starting-slot target?

The exact state evaluator used the frozen 2022–23 research model in place of
the existing PA formula, with the same neutral park and pooled transition
table. Actual I2 starting slots were joined only after forecasts were made.

| Season | Slot log loss: existing → candidate | Paired difference and 95% interval | Top late-slot error: existing → candidate | Bottom late-slot error: existing → candidate |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 1.504287 → 1.503591 | -0.000696 [-0.002920, +0.001585] | +0.91 → -0.75 pp | -2.67 → -0.75 pp |
| 2025 | 1.529319 → 1.529003 | -0.000316 [-0.002587, +0.001794] | -1.42 → -2.62 pp | -2.34 → +0.08 pp |

Both slot log-loss intervals include zero. The fitted home effect largely
corrects the 2024 bottom-slot residual but worsens the 2025 top-slot
residual. The 2025 result shows that a fixed side pattern is not stable
enough to promote from these retrospective samples. No full-I2 Under
probabilities have been evaluated for this candidate.

## Artifacts and next boundary

- Joint PA fit and first-three scoring:
  `data/derived/i2_vnext/phase12_i1_direct_pa_candidate.json`.
- Target-free frozen research model:
  `data/derived/i2_vnext/i1_direct_pa_home_research_2022_2023.json`.
- Exact slot comparison:
  `data/derived/i2_vnext/phase12_i1_direct_slot_comparison.json`.
- Reproduction scripts:
  `src/research/export_i1_opening_pa_vectors.mjs`,
  `src/research/fit_i1_pa_direct_model.py`,
  `src/research/run_i1_state_ab.mjs`, and
  `src/research/compare_i1_direct_slot.py`.

Large Retrosheet ZIPs, pregame input snapshots, and per-PA/per-game rows stay
outside git; the compact Phase 11 result records raw source checksums.
The 2024 and 2025 I1 outcomes were inspected before this candidate was
specified, so neither is independent promotion evidence. Freeze an
end-to-end version and collect future immutable pregame forecasts before
any live change. A full-I2 replay is required to learn whether this I1
component affects the actual Under target; do not infer that from PA or
slot metrics alone.

# I2 vNext Phase 24 — Individual Variable Audit

**Status:** complete research audit; production unchanged.

The audit scores each active I1 player-rate and Savant event-level park input against the actual target: **full second-inning Under 0.5**. Variable selection uses **2022–2025** only; **2026 is replication/sensitivity**, not a selection year. No market prices are used.

## Main result

The player-specific first-inning event-rate layer adds no measurable full-I2 value. Neutralizing all hitter-specific I1 rates, all pitcher-specific I1 rates, or both together produces essentially zero change. This supports keeping the I1→I2 batting-slot/state engine while simplifying its player-specific I1 talent inputs.

The strongest event-level venue inputs are Savant **single** and **triple** factors: both improve log loss in all four development years and again in 2026. Double is positive but less stable. HR is unstable; full removal is not validated, and shrinkage remains research-only.

## Individual full-I2 ablations

Positive delta means removing the variable made prediction worse, so the variable helped.

| Variable | 2022–25 Δ log loss | 2022–25 Δ Brier | Positive dev years | 2026 Δ log loss | Decision |
|---|---:|---:|---:|---:|---|
| park_single | +0.0003836 | +0.0001870 | 4/4 | +0.0005867 | RETAIN_STRONG |
| park_double | +0.0002680 | +0.0001309 | 3/4 | +0.0001876 | RETAIN_LOW_CONFIDENCE |
| park_triple | +0.0001731 | +0.0000846 | 4/4 | +0.0003060 | RETAIN_STRONG |
| hitter_double | +0.0000019 | +0.0000009 | 3/4 | -0.0000016 | NO_RELIABLE_INCREMENTAL_VALUE |
| pitcher_triple | +0.0000010 | +0.0000005 | 4/4 | -0.0000017 | NO_RELIABLE_INCREMENTAL_VALUE |
| pitcher_hit_by_pitch | +0.0000010 | +0.0000005 | 3/4 | +0.0000013 | NO_RELIABLE_INCREMENTAL_VALUE |
| hitter_hit_by_pitch | +0.0000009 | +0.0000004 | 3/4 | +0.0000030 | NO_RELIABLE_INCREMENTAL_VALUE |
| pitcher_single | +0.0000006 | +0.0000003 | 1/4 | +0.0000043 | NO_RELIABLE_INCREMENTAL_VALUE |
| hitter_single | +0.0000001 | +0.0000001 | 2/4 | -0.0000026 | NO_RELIABLE_INCREMENTAL_VALUE |
| hitter_strikeout | +0.0000000 | +0.0000000 | 0/4 | +0.0000000 | REMOVE_MECHANICALLY_REDUNDANT |
| pitcher_strikeout | +0.0000000 | +0.0000000 | 0/4 | +0.0000000 | REMOVE_MECHANICALLY_REDUNDANT |
| pitcher_walk | -0.0000004 | -0.0000002 | 2/4 | -0.0000076 | NO_RELIABLE_INCREMENTAL_VALUE |
| hitter_triple | -0.0000005 | -0.0000002 | 1/4 | +0.0000011 | NO_RELIABLE_INCREMENTAL_VALUE |
| hitter_walk | -0.0000008 | -0.0000004 | 2/4 | -0.0000065 | NO_RELIABLE_INCREMENTAL_VALUE |
| hitter_home_run | -0.0000012 | -0.0000006 | 1/4 | -0.0000003 | NO_RELIABLE_INCREMENTAL_VALUE |
| pitcher_double | -0.0000014 | -0.0000007 | 1/4 | +0.0000025 | NO_RELIABLE_INCREMENTAL_VALUE |
| pitcher_home_run | -0.0000020 | -0.0000010 | 1/4 | -0.0000014 | NO_RELIABLE_INCREMENTAL_VALUE |
| park_home_run | -0.0004082 | -0.0002014 | 2/4 | +0.0001573 | SHRINKAGE_RESEARCH_ONLY |

## Group tests

| Group neutralized | 2022–25 Δ log loss | 2026 Δ log loss | Interpretation |
|---|---:|---:|---|
| All hitter-specific I1 rates | +0.0000005 | -0.0000071 | No measurable contribution |
| All pitcher-specific I1 rates | -0.0000010 | -0.0000024 | No measurable contribution |
| All hitter + pitcher I1 rates | -0.0000005 | -0.0000112 | No measurable contribution |
| All four park event factors | +0.0003430 | +0.0010867 | Aggregate venue layer helps, but components differ sharply |

## Direct I2 feature families already audited

Prior chronological refits found batter identity and pitcher identity to be the durable direct-I2 contributors. Standalone handedness/platoon did not show reliable incremental value, and individual arsenal interaction terms were much smaller with uncertainty spanning zero. A prior 2024–2025 full-inning audit likewise found no reliable individual arsenal gain. Exact numeric direct-feature ablation values were not durably persisted, so this checkpoint deliberately does not invent them.

## Decisions

1. **Retain:** direct batter identity, direct pitcher identity, Savant single park factor, Savant triple park factor.
2. **Retain but lower confidence:** Savant double park factor.
3. **Simplify next candidate:** neutralize the player-specific I1 hitter and pitcher event-rate layer while preserving batting order and the I1→I2 state engine. I1 strikeout is mechanically redundant and can be removed outright.
4. **Do not remove HR park factor outright yet:** the direction changes by season. Shrink-to-neutral is a valid challenger, but this audit does not promote it.
5. **Production remains unchanged** until the simplified candidate is replayed under the existing promotion governance.

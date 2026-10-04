# I2 vNext Phase 26A — Direct Feature Ablation

**Status:** screening complete; production unchanged.

This phase evaluates the direct-I2 PA event model using full-season chronological holdouts. Development years are 2022–2025; 2026 is replication only. No market data are used.

## Aggregated result

Positive delta means removing the feature made held-out prediction worse.

| Feature removed | 2022–25 Δ log loss | Positive dev years | 2026 Δ log loss | Interpretation |
|---|---:|---:|---:|---|
| Batter identity | +0.005340 | 4/4 | +0.006605 | Strong retain |
| Pitcher identity | +0.001637 | 4/4 | +0.002257 | Strong retain |
| Batter + pitcher identity | +0.006935 | 4/4 | +0.008786 | Strong retain |
| Platoon categorical term | -0.000039 | 1/4 | -0.000075 | Candidate removal |
| Home-team nuisance | -0.000675 | 1/4 | -0.000433 | Candidate removal |
| All arsenal interactions | +0.000039 | 2/4 | +0.000254 | Small/unstable; research only |

Batter identity is the strongest direct-I2 feature. Pitcher identity is also consistently valuable. Their removal is materially worse in every development year and again in 2026, with 2026 confidence intervals excluding zero.

Standalone platoon and home-team nuisance do not show durable incremental value. Arsenal effects are much smaller than identity effects and are not stable enough for promotion decisions at this stage.

## Governance

These are **PA-event screening results only**. No feature is removed from production based on this phase alone. Candidate removals must be replayed against the actual **full-I2 Under 0.5** target before any promotion.

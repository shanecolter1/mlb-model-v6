# I2 temporal confounds: first forecast comparison

This phase asks whether a stable position-in-season curve forecasts the full
second-inning Over better than the existing pregame model. It uses all primary
home games and outcomes, with no batted-ball-type filter and no sportsbook
inputs. The date basis is fixed before coefficient fitting: an early ramp from
the first game through 28 days and a late ramp in the final 42 days. The
candidate adds these terms, plus one intercept, to the logit of the frozen
pregame full-I2 Over probability. A constant-only offset is a simpler
comparator. Both candidates select one ridge penalty by leave-one-season-out
log loss over 2023–2025. The 2026 games are scored only after that selection.

2022 is excluded from this candidate's training because it precedes the 2023
rule package. It remains in the earlier date audit. This rule-era distinction
does not identify the pitch clock's effect separately from the other changes.

| Candidate | 2023–25 leave-one-season-out log loss | Change vs raw | 2026 log-loss change vs raw | 2026 Brier change vs raw |
| --- | ---: | ---: | ---: | ---: |
| Raw pregame model | 0.688326 | 0 | 0 | 0 |
| One constant logit offset | 0.687883 | -0.000442 | +0.000492 | +0.000244 |
| Early/late season-position curve | 0.687923 | -0.000403 | +0.000226 | +0.000112 |

Negative changes improve scoring. The selected season-position fit used ridge
0.1, with coefficients approximately +0.03977 for the common intercept,
+0.01167 for the early ramp, and **-0.00358** for the late ramp. Its late
effect is essentially zero and has the opposite sign from a recurring late
Over increase. The constant fit selected ridge 0.01. The curve was worse than
the simpler constant in the development comparison; both worsened full-season
2026 proper scores relative to raw.

The day-clustered 95% confidence interval for the curve's 2026 log-loss
change is [-0.000590, +0.001046]. The candidate does not explain the
within-2026 change: its actual-minus-predicted Over residual is **-3.74 pp**
before May 25 and **+0.49 pp** thereafter. The raw residuals are **-2.71**
and **+1.44 pp**, respectively. A stable curve estimated on 2023–2025 cannot
be used as an explanation for the 2026 change.

This is a forecast test of a deliberately simple calendar pattern, not an
estimate of ball causality. The May 25 date came from a reported carry shift
and was not used to fit a 2026 coefficient. The 2026 results had already been
inspected, so this is a replication diagnostic rather than a pristine holdout.
No model or calibration is promoted by this phase.

## Next model test

The direct-I2 event model currently has a season-level prior for league event
rates and monthly as-of player refits. It does not expose a continuously
updated league environment as an independent pregame input. A forecastable
regime candidate needs event observations available **before** each game's
freeze: league-wide strikeout, walk, batted-ball-hit and home-run rates, plus
measured drag/carry if reliably available. Estimate player and pitcher
deviations relative to those as-of league rates, then replay each top/bottom
inning with the changed event probabilities. Select a single joint shrinkage
policy on earlier seasons and compare full-I2 scores on later seasons. That
test must be built from point-in-time event records; the current replay ZIPs
contain pregame forecasts and I2 outcomes, not the full league contact history
needed to fit it safely. A postgame 2026 date label is not a substitute.

Code: `src/research/validate_i2_season_position.py`.
Machine-readable scores: `data/derived/i2_vnext/i2_season_position_validation.json`.

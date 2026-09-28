# I2 vNext Phase 10 — earlier I1 state and replay parity

Status: **SHADOW ONLY**. No live formula, production model, calibration, pricing,
or staking change. This phase checked the Phase 9 first-inning state residual
against the archived 2024 pregame A/B artifact, earlier observed seasons, and
two fixed 2025 component ablations.

## Earlier first-inning pattern

"Late" means the second inning starts at batting slot 7, 8, or 9, usually
after a longer first inning. The paired observed bottom-minus-top late-slot
gap, with a calendar-date cluster 95% interval, was:

| Season | Paired games | Bottom minus top late-slot rate |
| --- | ---: | ---: |
| 2021 | 2,428 | +2.47 pp [+0.29, +4.66] |
| 2022 | 2,430 | +2.35 pp [+0.50, +4.27] |
| 2023 | 2,430 | +3.99 pp [+1.95, +6.14] |
| 2024 | 2,428 | +3.54 pp [+1.40, +5.65] |
| 2025 | 2,430 | +0.91 pp [-1.01, +2.88] |

The 2021–23 combined gap was +2.94 pp [+1.77, +4.23]. The actual bottom
first inning also had 0.11–0.20 more PAs on average in every season. The
2025 top/bottom gap was smaller and its interval included zero. These are
observed outcomes, not an estimated causal home-side effect.

The exact pregame I1 simulator's **2024 player-as-of** arm assigned late-slot
probability 14.50% top and 14.46% bottom, versus 13.59% and 17.13% realized.
Among 2,428 paired games it predicted a bottom-minus-top gap of -0.04 pp;
the actual gap was +3.54 pp. The paired gap error was -3.58 pp
[-5.58, -1.40]. In 2025 the simulator predicted 14.26% top and 14.24%
bottom, versus 15.68% and 16.58% realized. The 2025 paired gap error was
-0.92 pp [-2.80, +1.06], while bottom's absolute late-slot error remained
-2.34 pp. The 2024 A/B selection chose player-as-of rates based on overall
slot log loss and Brier; this side calibration audit was not part of that
precommitted selection.

The archived compact I1 PA counts and I2 start slots agree exactly under
`slot = (I1 PA mod 9) + 1` in every 2021–25 half. Two suspended games with
top and bottom halves on different dates were excluded from paired summaries.

## Fixed mechanism checks on the inspected 2025 sample

| I1-only change from archived neutral, pooled replay | Change in predicted late-slot rate | Change in slot log loss |
| --- | ---: | ---: |
| Existing separate strikeout / ball-in-play transitions | +0.021 pp | -0.000004 |
| Apply prior-season Savant park profile to I1 | +0.074 pp | -0.000099 |

Both paired date-cluster log-loss intervals include zero. The K/BIP artifact
used fixed shrinkage strengths but was trained through 2024, so it was tested
only on 2025. The venue arm uses the same actual-site mapping as the replay;
166 games at unprofiled 2025 venues remain explicit neutral fallbacks. Neither
fixed change explains the bottom I1 state residual. This does **not** rule out
other event-rate or transition errors.

## Replay-to-live finding

The archived 2025 vNext replay passed `environmentalContext: null` for I1,
while applying venue factors to I2. The live shadow runner passes a matched
venue context to both I1 and I2. The research replay now offers an explicit
`--i1-environment prior_season_park` arm, defaulting to `neutral` so previous
artifacts remain reproducible. The exact I1 evaluator uses the same venue
mapping for its park-aware component check. Switch hitters with Retrosheet
`B` batting side are represented as `S` when the simulator applies a park
split. Shard merging now rejects mixed I1 environment modes. This aligns a
research option with the live environmental rule; **no full-I2 outcome replay
or promotion has been claimed**. Other historical/live input differences
still prevent calling the archived comparison an exact live replay.

## Reproduction and next boundary

Compact results: `data/derived/i2_vnext/phase10_i1_side_state_audit.json`.
Sources are the tracked 2021–25 Retrosheet compact files; the archived 2024
I1 A/B artifact from Actions run `36275634254`; the Phase 4A 2025 pregame
inputs from run `36277316668`; and the 2024 park profile from the data-phase
artifact in run `36273567216`. The original 2025 neutral slot rows were made
in Phase 9. Generate the separate K/BIP artifact using
`src/research/build_i2_separate_out_transitions.py`, then run the exact I1
evaluator with `--play-calibration` or `--parks` for each component arm. The
summary script is `src/research/audit_i1_side_state.py`.

The next model-development question is whether pregame PA event rates or
other base/out behavior produce the persistent bottom-side residual. Study
PA-level first-inning evidence from training years before specifying a
single integrated candidate. The observed 2024 and 2025 results are already
inspected; independent future frozen forecasts must decide promotion.

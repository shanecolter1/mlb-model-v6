# I2 vNext Phase 9 — I1 state and venue audit

Status: **SHADOW ONLY**. The production model, live vNext formula, full-I2
calibration, betting workflow and thresholds are unchanged.

## Question and method

The 2025 vNext precision replay predicts too many full-second-inning Unders.
This phase separates a possible first-inning batting-order error from residuals
at actual venues with no prior-season park profile. It reuses the frozen
2,430-game, 10,000-trial precision replay and its strictly pregame 2025 player
inputs. The existing exact I1 evaluator propagates the pregame PA vectors
through the same pooled base/out transition table as the replay. Actual I2
starting slots from Retrosheet are joined **after** computing each slot
distribution, solely to score it. The observed slots never enter I2 prediction.

| 2025 I1 halves | Predicted slot 7–9 | Actual slot 7–9 | Predicted minus actual |
| --- | ---: | ---: | ---: |
| Top, 2,430 | 14.26% | 15.68% | -1.42 pp |
| Bottom, 2,430 | 14.24% | 16.58% | -2.34 pp |
| Combined, 4,860 | 14.25% | 16.13% | -1.88 pp |

The combined 95% calendar-date cluster interval for the 7–9 error is
[-2.84, -0.84] pp. Pregame player-as-of rates have slot log loss 1.529319 and
Brier 0.740623, versus 1.531446 and 0.741210 for league rates in this
already-inspected sample. The I1 state model is missing some longer first
innings. That alone does not establish the size or direction of its effect on
full-I2 Under probabilities, because I2 hitter quality varies by batting slot.

## Where the full-I2 error sits

| Group | Games | Predicted Under | Actual Under | Error | Excess expected Unders |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 2,430 | 57.14% | 55.14% | +2.00 pp | 48.54 |
| Prior-season venue matched | 2,264 | 57.13% | 55.83% | +1.30 pp | 29.37 |
| Actual venue has no prior profile | 166 | 57.33% | 45.78% | +11.55 pp | 19.17 |
| Sacramento (SAC01) | 81 | 56.48% | 39.51% | +16.97 pp | 13.74 |
| Tampa temporary venue (TAM02) | 81 | 58.17% | 54.32% | +3.85 pp | 3.12 |

The 166 neutral-venue games account for 39.5% of aggregate excess expected
Unders, though only 6.8% of games. Their calendar-date cluster 95% error
interval is [+3.74, +19.18] pp. The matched-venue error interval is
[-0.84, +3.37] pp. Small special-event site groups are descriptive only.

At Sacramento, the top-half scoreless forecast was 74.37%, versus 59.26%
realized; the bottom-half forecast was 75.94%, versus 74.07% realized. As a
within-team context check, when the Athletics visited other parks, opponents'
bottom-half scoreless forecast was 73.93%, versus 70.37% realized (81 games).
The larger Sacramento miss is consistent with a venue-specific problem, but
opponents, starters, schedule and weather also differ. This is not an estimate
of a park effect and is not grounds to invent a venue adjustment.

Even in the 2,264 matched-venue games, top scoreless was underpredicted by
1.56 pp while bottom scoreless was overpredicted by 2.52 pp. Venue coverage
does not resolve the remaining half-inning asymmetry. June 25 onward full-I2
Under overprediction is 3.43 pp on the same precision replay.

All intervals are exploratory calendar-date cluster bootstraps from a 2025
sample whose outcomes informed earlier diagnostics and selection; they are not
promotion evidence or adjusted for multiple comparisons. There is no new
full-I2 candidate in this phase.

## Reproduction and next boundary

The compact result is
`data/derived/i2_vnext/phase9_i1_venue_diagnosis.json`. The exact evaluator
now accepts an observed-slot CSV as a target-only input. The source inputs are
the archived Phase 4A prep artifact from Actions run `36277316668`, the Phase 5
precision artifact from run `36278930125`, and the tracked Retrosheet compact
2025 file. After extracting the two artifacts to an input directory:

```bash
node src/research/run_i1_state_ab.mjs \
  --input INPUT_DIR/replay_2025_inputs.json \
  --observed-slots data/derived/i2/i2_state_compact_2025.csv \
  --output INPUT_DIR/i1_state_2025_rows.json
python src/research/diagnose_i1_venue_2025.py \
  --slot-rows INPUT_DIR/i1_state_2025_rows.json \
  --precision-replay INPUT_DIR/replay_2025_precision_10000.json \
  --replay-input INPUT_DIR/replay_2025_inputs.json \
  --output data/derived/i2_vnext/phase9_i1_venue_diagnosis.json
```

Next, evaluate the longer-I1 residual on the earlier, pre-2025 start-slot
selection data and inspect whether its PA event rates or base/out transitions
are responsible. Any candidate change needs an independently frozen later
validation set. Temporary venues remain neutral and ineligible for betting
until a genuinely pregame, venue-specific baseball source is validated.

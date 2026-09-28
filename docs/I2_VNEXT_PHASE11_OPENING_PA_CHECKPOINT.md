# I2 vNext Phase 11 — first-inning opening PA evidence

Status: **SHADOW ONLY**. No prediction, production, calibration, market,
staking, or betting threshold change. This phase examined the first three
plate appearances in each first-inning half. These batters appear before
longer-innings selection can distort the PA event mix.

Phase 12 fitted a single research-only I1 PA model with the home-side feature
and scored it chronologically at the PA level and descriptively on I1 slots.
See `docs/I2_VNEXT_PHASE12_DIRECT_I1_PA_CHECKPOINT.md`.

## Observed event mix in earlier seasons

"Modeled reach" is 1B, 2B, 3B, HR, walk, or HBP in the I1 simulator's
eight-event taxonomy. Reached-on-error and fielder's-choice outcomes remain
in the simulator's ball-in-play bucket. Calendar-date paired 95% intervals:

| Season | Paired games | Bottom minus top modeled-reach rate | Bottom minus top K rate |
| --- | ---: | ---: | ---: |
| 2021 | 2,429 | +2.35 pp [+0.97, +3.73] | -1.99 pp [-3.34, -0.68] |
| 2022 | 2,430 | +3.79 pp [+2.29, +5.15] | -3.44 pp [-4.82, -2.16] |
| 2023 | 2,430 | +3.17 pp [+1.55, +4.75] | -3.79 pp [-5.21, -2.33] |
| 2024 | 2,429 | +2.26 pp [+0.79, +3.78] | -3.17 pp [-4.54, -1.83] |

For the opening three 2024 PAs, bottom-minus-top outs added per PA was
-0.02319. A symmetric descriptive identity assigns -0.02238 to the event
mix and -0.00081 to outs conditional on event. That calculation includes
base-state differences inside the conditional term; it does not establish
a causal home effect or prove every base/out transition is correct. Runner
outs on non-PA plays were approximately 0.028 per top half and 0.026 per
bottom half, too similar to explain this opening-PA gap descriptively.

## Pregame 2024 forecast check

The archived, strictly pregame 2024 player-as-of inputs generated the
existing 50/50 batter/pitcher log-odds event vectors for batting slots 1–3.
Five games with a changed batter or pitcher identity during those opening
PAs were excluded; 2,424 paired games and 14,544 PAs remain.

| First-inning half | Predicted modeled reach | Observed modeled reach | Predicted minus observed |
| --- | ---: | ---: | ---: |
| Top | 31.87% | 30.71% | +1.17 pp |
| Bottom | 31.83% | 32.95% | -1.12 pp |

The bottom-minus-top *forecast error* was -2.29 pp, with a paired date-cluster
95% interval of [-3.77, -0.80] pp. Predicted strikeout rates were about
21.9% on both sides; observed rates were 24.41% top and 21.26% bottom.
Thus the first three PA outcomes already show a side-specific event-rate
residual before the transition engine can generate a long inning. Team,
starter, lineup, park, and other context may contribute; this audit does not
isolate the cause.

## Governance and next experiment

The reproducible compact result is
`data/derived/i2_vnext/phase11_i1_opening_pa_audit.json`. Retrosheet parsed
2021–24 play ZIP checksums are in that artifact. The 2024 pregame vector
export is `src/research/export_i1_opening_pa_vectors.mjs`, reading the archived
I1 A/B input from Actions run `36275634254`. The target-only audit is
`src/research/audit_i1_opening_pa.py`; the large inputs and row export stay
outside git. No sportsbook service was called.

The event-mix result makes a PA-model investigation more promising than a
standalone I1 starting-slot offset. A research candidate should put any
home-batting-side effect *inside* the I1 PA event model and estimate it from
pre-2024 data, with pregame talent and venue handled in the same model.
Do not add a fixed slot correction or another independent shrinkage layer.
The 2024 sample was used for earlier I1 selection, and this hypothesis was
formed after inspecting it; comparisons on 2024 or 2025 remain descriptive.
Freeze a candidate before judging it on independent future games.

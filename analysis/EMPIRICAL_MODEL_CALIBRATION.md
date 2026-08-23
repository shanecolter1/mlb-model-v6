# Empirical Model Calibration Build

## Purpose
Replace hand-set live half-inning model assumptions with coefficients and transition probabilities estimated from historical baseball outcomes. This analysis uses baseball data only; no sportsbook or betting-market inputs are permitted.

## Historical sources already in repository
- `data/derived/i2/i2_play_calibration.json`
  - Retrosheet 2021-2025 regular seasons
  - 913,349 plate appearances
  - 192 event × outs × base-state transition cells
- `data/derived/i2/i2_state_compact_2021.csv` through `_2025.csv`
  - First-inning / second-inning continuation observations
  - Includes I1 PA, I1 runs, I1 pitch count, I2 lineup start slot, same-pitcher indicator, I2 PA/runs/pitches

## Completed empirical replacement analysis
The legacy runner-advancement rules are not supported by the observed data and should be removed from production rather than retuned as global constants.

### Runner on first, double: probability runner scores
| Outs | Legacy | Empirical | N |
|---:|---:|---:|---:|
| 0 | 45.0% | 28.90% | 2,242 |
| 1 | 45.0% | 32.85% | 2,712 |
| 2 | 45.0% | 52.15% | 2,577 |

### Runner on second, single: probability runner scores
| Outs | Legacy | Empirical | N |
|---:|---:|---:|---:|
| 0 | 60.0% | 37.31% | 2,643 |
| 1 | 60.0% | 52.00% | 3,477 |
| 2 | 60.0% | 78.90% | 4,157 |

### Runner on first, single: clean first-to-third advancement
| Outs | Legacy | Empirical | N |
|---:|---:|---:|---:|
| 0 | 30.0% | 25.48% | 7,593 |
| 1 | 30.0% | 26.95% | 9,238 |
| 2 | 30.0% | 34.85% | 8,683 |

## Required architecture change
Do not replace the old constants with three new constants. Replace `hitTransitions()` with the empirical conditional transition distribution:

`event type × outs before play × pre-play base mask -> P(outs added, post-play base mask, runs scored)`

This preserves the actual historical dependency of advancement on outs and base state and also captures runner putouts and uncommon transitions.

## Outcome-state expansion
The live transition engine should distinguish at least:
- strikeout
- ball-in-play out
- walk
- hit by pitch
- single
- double
- triple
- home run

The existing generic `out` state discards materially different runner-advancement behavior between strikeouts and balls in play.

## Pitcher continuation calibration
`analysis/empirical_model_calibration.py` combines the five season-level I1->I2 files and produces:
- empirical continuation by I1 pitch-count bin
- empirical continuation by I1 runs allowed
- a dependency-free logistic fit using I1 pitches, I1 runs, I1 PA and I2 lineup start slot

Important limitation: those files identify first-inning-to-second-inning continuation only. They must not be extrapolated to later-inning starter/reliever continuation without a full appearance-level historical dataset.

## Remaining parameters that require additional historical reconstruction
### Batter-pitcher blend
The current 68% batter / 32% pitcher blend cannot be validly fitted from aggregate transition tables. Required historical row fields:
- date/game ID
- batter ID
- pitcher ID
- PA outcome
- leakage-safe batter history available before the PA/game
- leakage-safe pitcher history available before the PA/game

Fit event-specific batter/pitcher effects rather than one common blend for every outcome.

### Live pitcher deterioration
Required pitch/appearance-level historical fields:
- pitcher ID and appearance ID
- pitch count/workload
- batters faced / times through order
- velocity versus baseline
- hard-hit/barrel/contact quality
- command/count variables
- subsequent PA outcomes

### Confidence shrinkage
Required out-of-sample prediction ledger:
- raw model probability
- realized result
- data-coverage fields
- projected-versus-live flag
- sample-size / player-history measures

Confidence should then be calibrated from observed forecast error rather than fixed High/Medium/Low multipliers.

## Build command
From repository root:

```bash
python analysis/empirical_model_calibration.py
```

Generated files are written under `data/derived/model_calibration/`.

## Production recommendation
The first production change should be the empirical event/base/out transition engine. It removes the largest demonstrated structural error while relying entirely on historical baseball observations already stored in the repository.

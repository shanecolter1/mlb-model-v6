# I2 vNext Phase 7B checkpoint — 2026-09-26

Status: SHADOW ONLY. No live model, staking, market workflow or prediction calibration promoted.

## Completed matched-input comparison

Actions run 36280824619 succeeded on all 2,430 games with 10,000 trials per game. The control uses production's 50/50 log-odds event formula, strictly pregame player rates, and the same prior-season park, league baseline, I1 engine and transition table as vNext. This is a controlled I2-formula comparison, not an exact deployed-production replay: the live production baseline/transition artifact pools through 2025 and cannot be used unchanged to test 2025.

| Period | vNext Brier | Control Brier | vNext log loss | Control log loss |
| --- | ---: | ---: | ---: | ---: |
| All 2025 | 0.247057 | 0.246654 | 0.687262 | 0.686427 |
| Later validation | 0.249705 | 0.248390 | 0.692652 | 0.689938 |

All paired calendar-date bootstrap intervals for the loss differences include zero. The data were previously used for calibration selection; these are descriptive comparisons, not a fresh independent promotion test.

Canonical results: `data/derived/i2_vnext/phase7b_bias_diagnosis.json`.

## Bias localization

- Top-I2 scoreless bias: -0.70 percentage points; bottom-I2 scoreless bias: +2.43 points.
- Neutral park fallback games: 166 games, +11.55 points full-I2 Under bias. Matched parks: 2,264 games, +1.30 points.
- Both starters began I2: 2,353 games, +1.94 points. At least one starter changed before I2: 77 games, +3.60 points. Starter changes explain only a small share of the aggregate residual sum.
- These are descriptive groups. Actual starter continuation is used only for diagnostic grouping, never as a pregame predictor.
- A missing home-offense effect is a candidate explanation for the half-inning asymmetry, not a demonstrated cause. Do not fit a correction from these diagnostics.

## Confirmed transition export mismatch and research correction

The original transition fitting/scoring code distinguishes strikeout and ball-in-play out. Its exported table pools them into `out`; the vNext adapter consequently uses the same transition distribution for both events.

Example: runner on first, no outs. Archived 2021–2024 observed multiple-out rates are 1.806% for strikeouts (7,972 observations) and 23.539% for balls-in-play outs (18,178 observations). Both currently receive 16.911% in the pooled replay.

`src/research/build_i2_separate_out_transitions.py` reconstructs the 48 separate K/BIP states from existing archived counts. It reuses the frozen strengths 5 and 1280; it does not retune or introduce another shrinkage layer. It preserves all original states and leaves walk/HBP treatment unchanged. `run_i2_vnext_replay.mjs` now prefers exact event states when supplied, falling back to the existing pooled mapping for old artifacts. The default artifact path is unchanged.

The research artifact is deterministically regenerated with:

```bash
python src/research/build_i2_separate_out_transitions.py
```

The resulting file is `data/derived/i2_vnext/i2_distinct_out_transitions.json`. The compact saved audit is `data/derived/i2_vnext/distinct_out_transition_audit.json`.

2025 conditional transition log loss:

| Event | N | Pooled | Separate |
| --- | ---: | ---: | ---: |
| Strikeout | 40,645 | 0.130293 | 0.054973 |
| Ball-in-play out | 83,299 | 0.286174 | 0.273157 |

This is component-level evidence only. Full-I2 probability impact has not been replayed. All 48 new states normalize, original states are unchanged, and the research replay adapter passed a smoke test with the separate-state artifact.

## Next bounded phase

Run a paired 10,000-trial full-I2 replay with the same frozen player models, I1 state, parks, game set and seeds, changing only the transition artifact to the generated separate-state table. Compare against the existing precision replay; do not refit calibration or promote from this reused 2025 sample. Then determine whether remaining bottom-half and neutral-venue bias warrants a separately preregistered study on older training/selection data.

## Workflow governance

Superseded full rebuild and Phase 4B workflows now require manual dispatch for computation. Their workflow-file push event runs only a checkpoint job. The two redundant runs triggered by the shared replay edit were cancelled by superseding workflow runs. Phase 7B reused existing artifacts and accessed no odds services.

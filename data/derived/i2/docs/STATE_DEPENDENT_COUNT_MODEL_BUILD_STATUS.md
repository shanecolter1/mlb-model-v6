# State-Dependent Pitch-Count Historical Analysis — Build Status

## Build completed
A reproducible analysis pipeline has been created to replace the current heuristic pitch-count adjustments with historically estimated state-dependent probabilities.

## Existing historical foundation
The existing I2/base-out engine already uses approximately 913,349 regular-season plate appearances from 2021–2025 across all 192 event × outs × base-state combinations. That engine remains intact.

## Missing layer being built
The new layer conditions the remainder-of-half-inning run distribution on:

`outs × base occupancy × balls × strikes`

for all 12 legal nonterminal counts.

The output target is the distribution of **additional runs from the current pitch state until the half inning ends**, including:
- 0
- exact 1
- exact 2
- exact 3
- 4+
- 1+
- 2+
- 3+
- 4+
- expected additional runs

## Data acquisition
The analysis script is configured to retrieve 2021–2025 regular-season Baseball Savant Statcast pitch-level data in small cached chunks through `pybaseball`.

This environment does not currently contain the full 2021–2025 pitch-level Statcast corpus, so the multi-million-row historical table has not yet been executed here. The pipeline can alternatively consume a pre-materialized parquet without downloading again.

## Statistical safeguards
- Pitch observations are grouped to half innings.
- Unique half-inning counts are retained.
- Production uncertainty will use half-inning-clustered bootstrap rather than naive pitch-level standard errors.
- Sparse exact states use a transparent first-stage empirical-Bayes shrinkage toward count-only probabilities.
- Raw state estimates remain retained for audit.
- No monotonicity is imposed by intuition.

## Validation design
Walk-forward:
- 2021–2022 -> 2023
- 2021–2023 -> 2024
- 2021–2024 -> 2025

Compare current heuristic count logic against:
1. count-only historical
2. count + outs
3. count × outs × bases

Promotion requires out-of-sample improvement; this package does not modify the production model.

## Market isolation
No sportsbook or derivative-market inputs are used.

## Intended GitHub paths
- `docs/STATE_DEPENDENT_COUNT_MODEL_PLAN.md`
- `research/count_state/README.md`
- `research/count_state/requirements-count-state.txt`
- `research/count_state/scripts/build_count_state_dataset.py`
- `research/count_state/scripts/validate_count_state_model.py`

Derived files after execution:
- `data/derived/count_state/count_state_raw.csv`
- `data/derived/count_state/count_state_smoothed.csv`
- `data/derived/count_state/count_only_summary.csv`
- `data/derived/count_state/pa_outcome_by_count.csv`
- `data/derived/count_state/state_model_metadata.json`
- `data/derived/count_state/validation_report.json`

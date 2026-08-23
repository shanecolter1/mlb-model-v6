# Seasonal Transition Correction Status

Status: IN PROGRESS / GOVERNANCE-CORRECTED BUILD CREATED
Date: 2026-08-23

## What was corrected

1. The pooled 2021-2025 transition calibration is no longer sufficient for temporal validation.
2. A new season-specific extractor downloads official Retrosheet parsed play-by-play ZIPs for 2021, 2022, 2023, 2024, and 2025.
3. Development split is fixed before validation:
   - 2021-2023: estimation
   - 2024: shrinkage hyperparameter selection only
   - 2025: locked chronological validation
4. Sparse exact event/outs/base states are handled with hierarchical empirical shrinkage rather than raw probabilities or arbitrary constants.
5. Shrinkage strength is selected on 2024 only, then frozen before 2025 evaluation.
6. The workflow verifies required artifacts and fails loudly if any are missing.
7. The workflow uploads build logs and generated artifacts even on failure for diagnosis.
8. A model development manifest is generated with source URLs, sample counts, split boundaries, selected shrinkage, 2025 validation scores, and governance status.

## Files added

- `analysis/seasonal_transition_shrinkage.py`
- `.github/workflows/seasonal_transition_shrinkage.yml`

Expected generated outputs:

- `data/derived/model_calibration/seasonal/transitions_2021.json`
- `data/derived/model_calibration/seasonal/transitions_2022.json`
- `data/derived/model_calibration/seasonal/transitions_2023.json`
- `data/derived/model_calibration/seasonal/transitions_2024.json`
- `data/derived/model_calibration/seasonal/transitions_2025.json`
- `data/derived/model_calibration/seasonal/shrinkage_grid_2024.csv`
- `data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json`
- `data/derived/model_calibration/seasonal/model_development_manifest.json`

## Promotion gate

Do not integrate the new transition table into the live dashboard until all expected generated outputs exist and the 2025 locked-validation result is available. If the shrunken model does not improve or at least remain competitive with the raw transition benchmark on 2025 log loss, classify the component as WARNING and review the shrinkage hierarchy before production promotion.

## Remaining issue

The workflow has been committed and explicitly triggered by a follow-up commit. The generated outputs must still be confirmed in the repository before the pipeline BLOCKED status can be cleared.

# I2 postmortem availability and incident, 2026-10-16

## Diagnosis

Run 38919784366 failed before the seasonal rollup while loading the absent October 15 frozen file. Run 38757825450 failed for October 14. Repository Actions history contains daily postmortem failures October 2–16 (including the reported October 5–16 range), with a successful October 1 evaluation of September 30. No production producer runs appear after September 30 in that history. October research runs are separate and do not establish production forecasts. Main's latest frozen prediction is September 30; full git history contains no October frozen prediction paths. This supports intentionally ungenerated forecasts, not lost October forecasts or attempted October production failures.

Both production workflows are manual-only. The daily workflow has been manual-only since September 26; v0.4 was enforced manual-only September 30 (8e40143). The v0.4 workflow persists v04_predictions, not the canonical frozen_predictions path. That naming gap is surfaced as missing canonical data if a v0.4 forecast exists; no implicit promotion or historical reconstruction is performed.

## Behavior

- The evaluator defaults to required mode. An explicit manual workflow dispatch defaults to require_forecast=true.
- Scheduled/push checks (and explicit dispatch with require_forecast=false) use auto mode. Present artifacts always undergo date, games, identity and probability validation.
- Absent artifacts can yield NO_FORECAST only in the verified manual-only era starting October 1, with no git history of that file, no sibling forecast/run-environment/empirical artifact or prior postmortem, no scheduled producer, and a complete successful Actions-history lookup showing no relevant main-branch producer run.
- New producer run titles identify requested dates, including backfills. Undated legacy runs in this era are conservatively treated as potential evidence for any date. Failed, cancelled, pending and successful runs all establish expectation; none are silently skipped.
- API failures, incomplete pagination (maximum 1,000 runs), dates older than 85 days, and unavailable credentials remain errors. Git history must be complete. These conservative limits require operator investigation rather than a fabricated no-forecast result.
- NO_FORECAST emits only a log/job-summary status: no frozen file, postmortem, observations or performance measurements are created.
- Seasonal aggregation runs even when daily grading fails, using existing postmortems only. Its successful output may be committed while the daily job remains red. Malformed seasonal JSON is fatal rather than silently omitted.
- Missing inning scores are never interpreted as zero. Only final games with explicit nonnegative inning scores are graded. Feed retrieval failures preserve the existing report and fail the daily step.
- Report commits never reset main or regenerate predictions. A concurrent push is rejected visibly and requires rerunning reporting against current main.

Prediction formulas, calibration, eligibility, sportsbook isolation and research-production boundaries are unchanged. The only producer modification is descriptive run naming. No paid prediction workflow is triggered by this repair.

## Validation and operation

Run `npm run test:i2-postmortem` or `npm test`. Tests include actual September 30 artifact validation, synthetic feed fixtures for grading, missing scores and retrieval errors, no-forecast scheduling, failed/successful/pending producers, missing persisted files, malformed predictions and seasonal inputs, API failure and pagination, research isolation and independent seasonal rollup. Synthetic observations exist only in temporary test directories.

To reevaluate October 15 on repaired code, dispatch the postmortem on main with date=2026-10-15 and require_forecast=false. A push of the workflow also evaluates yesterday. Rerunning the original failed run uses its old workflow revision and is not validation of the repair. Inspect the new run's NO_FORECAST summary, rollup and commit steps. For genuinely expected forecasts leave require_forecast=true; investigate producer logs and retained original artifacts without regenerating retrospective predictions.

# Version 6 event-engine architecture decision record

## Decision

The model will estimate mutually exclusive plate-appearance events before runs
and game outcomes. Phase 1.3 is shadow-only.

## Three comparison arms

### A — Legacy control
Uses the existing generic `park_score`.

### B — Savant run-factor-only
Removes `park_score` and applies only the Savant run multiplier to neutral
expected runs.

### C — Savant event vector
Removes `park_score`; adjusts 1B, 2B, 3B and HR event probabilities; then
renormalizes the mutually exclusive event vector and derives a run distribution.

## Double-counting correction

The aggregate Savant run factor is **not** applied inside the event-vector arm.
It is retained as a diagnostic/calibration target. Applying the run factor and
the component event factors in the same arm would count overlapping venue
information twice.

## Candidate features

Venue BB and SO factors remain disabled by default. They can be promoted only
after time-separated tests show incremental value beyond batter/pitcher walk and
strikeout models.

## Current limitations

The included base/out simulator is an engineering scaffold, not a validated
production simulator. Advancement rules and event interaction coefficients must
be estimated and validated from historical play-by-play data before promotion.

# I2 Derived Training / Calibration Data

These files are reproducible derivatives of the Retrosheet 2021–2025 regular-season CSV packages and support the MLB I2 Under/Over model.

## State-transition files

- `i2_state_compact_2021.csv`
- `i2_state_compact_2022.csv`
- `i2_state_compact_2023.csv`
- `i2_state_compact_2024.csv`
- `i2_state_compact_2025.csv`
- `i2_start_slot_summary.csv`
- `i2_state_benchmark.csv`
- `i2_state_manifest.json`

One compact state row represents one team/game side and links its observed I1 progression to its I2 outcome. Historical I1 state is calibration/target information only; a pregame model must simulate I1 rather than read the realized state.

## Play-state calibration files

- `i2_play_calibration.json`
- `i2_play_calibration_summary.csv`

`i2_play_calibration.json` is derived from 913,349 regular-season PAs across 2021–2025. It contains empirical transition PMFs conditional on:

`event class × outs before PA × base occupancy before PA`

for all 192 possible state combinations used by the model. Each PMF records outs added, post-play base occupancy, and runs scored. It also contains pitch-count PMFs by event.

The event normalization exactly matches the existing Version 6 trailing-event-rate engine:

- single
- double
- triple
- home_run
- walk
- hit_by_pitch
- strikeout
- ball_in_play_out (residual PA outcome class)

## Retrosheet attribution

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

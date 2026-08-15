# I2 State-Transition Derived Data

These files are derived from the Retrosheet 2021–2025 regular-season CSV packages and support the MLB I2 Under/Over model.

## Files

- `i2_state_compact_2021.csv`
- `i2_state_compact_2022.csv`
- `i2_state_compact_2023.csv`
- `i2_state_compact_2024.csv`
- `i2_state_compact_2025.csv`
- `i2_start_slot_summary.csv`
- `i2_state_benchmark.csv`
- `i2_state_manifest.json`

## Compact row schema

One row = one team/game side, linking its first-inning progression to its second-inning outcome.

- `season`
- `gid` — Retrosheet game ID
- `date`
- `half` — `top` for visiting offense, `bottom` for home offense
- `i1_pa`
- `i1_runs`
- `i1_pitches`
- `i2_start_slot`
- `same_pitcher_i2`
- `i2_pa`
- `i2_runs`
- `i2_pitches`

`i2_runs` is sufficient to derive exact 0/1/2/3/4+ and cumulative 1+/2+/3+/4+ outcomes.

## Governance

The observed I1 fields are historical calibration data. They are prohibited as realized inputs to a pregame prediction. Production must simulate I1 from pregame baseball information.

## Retrosheet attribution

The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.

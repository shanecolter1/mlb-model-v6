# Historical data contract

## Plate-appearance table

Required fields:

- `date`
- `game_id`
- `batter_id`
- `pitcher_id`
- `batting_team`
- `pitching_team`
- `event`

## Prediction output table

Required fields:

- `game_id`
- `prediction_timestamp`
- `model_arm`
- `predicted_home_runs`
- `predicted_away_runs`
- `predicted_total_runs`
- `home_win_probability`
- `home_win_by_2_probability`
- `home_win_by_3_probability`
- `venue_profile_window`
- `venue_profile_as_of`
- `lineup_status`
- `simulation_seed`
- `production_output_changed` — must remain false during shadow mode

## Result table

Required fields:

- `game_id`
- `home_runs`
- `away_runs`

Outcome fields must never be present in the pregame snapshot consumed by the
prediction engine.

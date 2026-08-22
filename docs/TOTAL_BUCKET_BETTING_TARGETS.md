# Historical Betting Targets by Opening Game Total

## Purpose

Build empirical probabilities and fair American odds for derivative MLB scoring markets, conditioned on the DraftKings pregame opening full-game total.

Benchmark window: **2021-2025 regular season**.

Opening-total buckets currently governed by the I2 model: **6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0**.

## Committed baseline

`data/derived/i2/total_bucket_i2_fair_values_2021_2025.csv` contains the exact existing total-conditioned I2 Yes Run / No Run probabilities and fair American odds derived from the approved 10,730-game historical population.

The governing source is `data/derived/i2/i2_total_conditioned_prior.json`.

## Complete target matrix

`src/analysis/build_total_bucket_betting_targets.py` builds the following directly from the row-level joined historical master:

- full-inning Yes Run / No Run probability and fair odds for innings 1-9;
- half-inning exact 0, 1, 2, 3, 4+ run probabilities;
- half-inning cumulative 1+, 2+, 3+ run probabilities and fair odds;
- first-five full-game exact-run distribution;
- F5 Over/Under fair odds at half-run thresholds;
- combined team F5 scoring distributions and fair odds;
- empirical number-of-scoreless-innings-per-game distribution by opening total.

All outputs are **empirical row-level counts**. The builder explicitly avoids Poisson reconstruction and inning-independence assumptions.

## Required source asset

The complete build requires the row-level 2021-2025 joined game master containing both:

1. DraftKings pregame opening full-game total; and
2. inning-level runs, preferably separated into away/home half innings.

That row-level join is **not currently committed to this repository**. The repository retains its aggregated total-conditioned I2 prior, but that aggregate is insufficient to reconstruct F5 and 2+ distributions without making assumptions.

When the historical master is restored, run:

```bash
python src/analysis/build_total_bucket_betting_targets.py \
  --input <joined_2021_2025_master.csv> \
  --output-dir data/derived/i2/total_bucket_targets
```

Expected outputs:

- `inning_any_run_by_total.csv`
- `half_inning_runs_by_total.csv`
- `f5_game_total_by_total.csv`
- `f5_team_total_by_total.csv`
- `scoreless_innings_distribution_by_total.csv`
- `manifest.json`

## Fair odds convention

For probability `p`:

- if `p > 0.5`, fair American odds = `-100 * p / (1-p)`;
- if `p <= 0.5`, fair American odds = `100 * (1-p) / p`.

Odds are rounded to the nearest whole American-odds point in committed CSV outputs.

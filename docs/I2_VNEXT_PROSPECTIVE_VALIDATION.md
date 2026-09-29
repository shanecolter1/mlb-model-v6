# I2 vNext prospective shadow ledger

Status: **SHADOW ONLY**. The current prospective cohort begins **2026-09-29**
and evaluates the finalized raw vNext probability stack:

- half-inning calibration: **disabled**
- final full-I2 calibration: **identity**
- probability stack version: `i2-vnext-raw-identity-prospective-v1`

This ledger records forecasts before first pitch and scores them after the game
from official MLB final feeds. It never reads odds or changes the production
betting workflow.

## Fixed selection and scoring rules

1. Run `src/pipeline/run_i2_vnext_today.mjs` before first pitch with a distinct
   `I2_OUTPUT` filename for each run. Preserve the file exactly as generated.
2. Archive it immediately. Archive entries are content-addressed and immutable.
   Any projected game generated after first pitch is rejected.
3. For each game, score the **latest valid pregame projection** within the
   single latest declared probability-stack cohort. A clean pre-freeze source
   recheck is required; invalidated snapshots are excluded before outcomes are
   read.
4. Once games are final, fetch the official MLB game feed once per game.
   Saved outcome files are not overwritten. Nonfinal games remain pending.
5. Report full-I2 Under Brier, log loss, mean prediction versus realized rate,
   AUC, fixed calibration bins, separate top/bottom summaries, and a
   confirmed-input subset. The scorer also verifies that the retired half
   layer changed no probability.
6. Prior probability-stack cohorts are never pooled into the current cohort.

## Current validation boundary

The 2022-2025 seasons were used for model/calibration development and 2026
was inspected during retrospective half-calibration research. Therefore none
of those seasons is a clean prospective promotion sample for the finalized
stack. The first clean cohort starts with games on or after **2026-09-29**.

Prospective scoring remains baseball-only. Prices, EV, and staking are not
read until a projection is frozen and the separate market-isolation boundary
is crossed.

## Commands

Before first pitch:

```bash
python src/research/i2_vnext_prospective.py archive \
  --snapshot data/runtime/i2/DATE_vnext_predictions.json \
  --archive-dir data/runtime/i2_vnext_prospective/forecasts
```

After games are final:

```bash
python src/research/i2_vnext_prospective.py fetch-final \
  --archive-dir data/runtime/i2_vnext_prospective/forecasts \
  --outcome-dir data/runtime/i2_vnext_prospective/mlb_final_feeds

python src/research/i2_vnext_prospective.py score \
  --archive-dir data/runtime/i2_vnext_prospective/forecasts \
  --outcome-dir data/runtime/i2_vnext_prospective/mlb_final_feeds \
  --prior-csv data/derived/i2/i2_state_compact_PRIOR_YEAR.csv \
  --output data/runtime/i2_vnext_prospective/score.json
```

# I2 vNext prospective shadow ledger

Status: **SHADOW ONLY**. This ledger records forecasts before first pitch and
scores them after the game from official MLB final feeds. It never reads odds
or changes the production betting workflow.

## Fixed selection and scoring rules

1. Run `src/pipeline/run_i2_vnext_today.mjs` before first pitch with a distinct
   `I2_OUTPUT` filename for each run. Preserve the file as generated.
2. Archive it immediately. An archive entry is content-addressed and created
   exclusively, so the same run can be archived again but cannot be changed
   in place. A projected game generated after first pitch is rejected.
3. For each game, score the **latest valid pregame projection**, regardless
   of whether its lineup source was confirmed or provisional. A clean
   pre-freeze source recheck is required; invalidated snapshots are excluded.
   This rule is applied before final outcomes are read.
4. Once games are final, fetch the official MLB game feed once per game.
   Files already saved are not overwritten. Nonfinal games stay pending.
5. Compare full-I2 Under forecasts against the prior regular season's
   full-I2 Under rate. Report Brier, log loss, mean prediction versus realized
   rate, AUC, fixed probability calibration bins, separate half-inning
   summaries, and a confirmed-input subset. A paired
   calendar-date interval is withheld until at least 20 scored dates.

The source snapshot SHA-256 and chosen timestamp are retained in each scored
row. A model-versus-production comparison requires a separate, genuinely
pregame production snapshot with the same game and cutoff; the prior-season
constant does not stand in for that comparison. Betting EV requires archived
post-freeze prices and is outside this baseball-only scorer.

## Commands

For a future regular-season date, after running the live shadow runner and
before first pitch:

```bash
python src/research/i2_vnext_prospective.py archive \
  --snapshot data/runtime/i2/DATE_vnext_predictions.json \
  --archive-dir data/runtime/i2_vnext_prospective/forecasts
```

After those games end:

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

Replace `DATE` and `PRIOR_YEAR` with the run date and the year preceding the
forecasts. The `fetch-final` command requests one MLB Stats API game feed for
each forecast game without an already saved feed. It does not poll a sportsbook.
Archive files and final feeds need to be persisted by the caller's future
workflow; this tool does not schedule or publish a run by itself.

## Validation boundary

The 2025 replay was inspected and used for calibration selection. It cannot
validate the Phase 7D hypotheses independently. Future prospective dates and
the unchanged model must be evaluated before changing calibration, adding a
home-side feature, or considering promotion. The full-I2 matched production
benchmark and price/EV audit remain separate required work. The reported
calibration bins and AUC are descriptive until the prospective sample is
large enough for stable comparisons.

# MLB Period Markets — Quick Reference

This folder contains live TheOddsAPI Business `/period-markets/` audits for MLB. These queries are post-freeze market enumeration only and must never feed back into the I2 prediction engine.

## Latest reference

- Human-readable sportsbook × market chart: `2026-08-30_mlb_period_markets.md`
- Filterable outcome-level CSV: `2026-08-30_mlb_period_markets.csv`
- Raw event-by-event API response: `../../data/runtime/i2/2026-08-30_period_markets_raw.json`
- Builder: `../../src/pipeline/build_period_markets_reference.mjs`
- Fast refresh workflow: `../../.github/workflows/period_markets_reference.yml`

## Live market keys found on 2026-08-30

- `h2h_1st_5_innings`
- `spreads_1st_5_innings`
- `totals_1st_5_innings`
- `alternate_totals_1st_5_innings`

No individual second-inning market key was returned in the live Business `/period-markets/` responses queried event-by-event for the MLB slate.

## Refresh

Run the GitHub Actions workflow **Build MLB period markets reference**. It queries each MLB event ID separately, writes the raw responses, rebuilds the sportsbook availability chart and outcome-level CSV, and commits the refreshed reference to this folder.

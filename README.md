# MLB Version 6 — Phase 1.5 real historical data adapter

This release moves the project from abstract historical data contracts to an
executable public-data ingestion workflow.

## Added

- MLB Stats API schedule and game-feed downloader
- Raw immutable JSON storage
- Game, PA, lineup, starter and result normalization
- Historical data audit
- Research snapshot builder
- GitHub Actions workflow
- Normalizer fixture test
- Explicit distinction between retrospective and verified-pregame snapshots

## Run locally or in GitHub Actions

```bash
python src/ingestion/fetch_mlb_statsapi.py   --start-date 2025-04-01   --end-date 2025-04-30   --output-dir data/raw/mlb_statsapi

python src/ingestion/normalize_mlb_feeds.py   --raw-dir data/raw/mlb_statsapi   --output-dir data/processed/mlb

python src/pipeline/audit_historical_data.py   --processed-dir data/processed/mlb
```

## Test

```bash
npm test
python tests/test_mlb_feed_normalizer.py
```

## Status

The real-data adapter is implemented. This environment did not permit the
external API download itself, so the package contains the executable workflow
and a realistic local fixture rather than claiming that a live historical
sample has already been fetched.

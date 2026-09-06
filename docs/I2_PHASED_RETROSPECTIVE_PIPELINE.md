# I2 retrospective analysis: phased durable pipeline

The historical I2 retrospective workflow is split into three independently restartable phases. Durable outputs are published as GitHub Release assets with SHA-256 checksums; Actions artifacts are secondary convenience copies only.

## Phase 1 — Frozen historical store
Workflow: `.github/workflows/i2_phase1_frozen_historical_store.yml`
Release tag: `i2-phase1-historical-store-v1`

Builds and freezes the reusable 2021–2025 source layer:
- normalized MLB games
- plate appearances
- inning results and other merged parquet tables produced by the normalizer
- strictly prior-date team feature store (`team_asof.parquet`)
- checksum-verified canonical joined master
- manifest with row counts and per-file SHA-256 hashes

This is the expensive acquisition/build phase. It is intended to be reused by future I2 analyses without reacquiring MLB feeds.

## Phase 2 — Strict chronological replay
Workflow: `.github/workflows/i2_phase2_strict_replay.yml`
Release tag: `i2-phase2-strict-replay-v1`

Consumes only the frozen Phase 1 release and produces:
- strict chronological OOS I2 predictions
- total-only baseline predictions
- walk-forward metrics
- coefficients by fold
- strict replay manifest and checksums

No historical lineup identity, actual starter identity, I2 derivative price, moneyline, runline, or juice is used.

## Phase 3 — Under/Over threshold calibration
Workflow: `.github/workflows/i2_phase3_threshold_calibration.yml`
Release tag: `i2-phase3-threshold-calibration-v1`

Consumes only the frozen Phase 2 release and produces:
- independent Under threshold sweep
- independent Over threshold sweep
- season breakdowns
- Wilson confidence intervals
- calibration-gap statistics
- total-only benchmark calibration
- leakage/market-isolation audit
- manifest and checksums

Threshold qualification remains price-blind. Sportsbook price and EV are separate downstream decision-engine steps.

## Persistence policy
GitHub Releases are the canonical permanent archive for large reusable phase outputs. GitHub Actions artifacts are convenience copies with 90-day retention and are not the canonical source. Each release asset is accompanied by a SHA-256 checksum and contains a manifest so future analyses can identify the exact frozen input version.

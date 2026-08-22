# Historical Joined Master Storage

## Canonical dataset

`MLB_Game_Stats_Joined_2021_2025.csv.gz` is the canonical large historical master for the 2021-2025 MLB regular seasons.

It should be stored as an immutable GitHub Release asset under tag:

`historical-mlb-2021-2025-v1`

The normal Git tree stores only code, derived outputs, documentation, and the dataset manifest. Do not commit the large CSV directly to the main branch.

## Provenance

The validated source package contains:

- 12,148 Retrosheet regular-season games (2021-2025)
- 10,933 eligible DraftKings regular-season final records with an opening total
- 10,911 reconciled market matches
- 22 deliberately excluded market records where exact reconciliation was not defensible
- 10,730 matched games with DraftKings opening total from 6.0 through 11.0
- 96,466 reached inning 1-9 observations in that 6.0-11.0 population

Primary reconciliation rule: archive date + away team + home team + final score, with documented team-code normalization and explicit exception handling.

## Required release assets

1. `MLB_Game_Stats_Joined_2021_2025.csv.gz`
2. `MLB_Game_Stats_Joined_2021_2025_README.txt`
3. `MLB_Game_Stats_Joined_2021_2025_reconciliation_audit.csv`

Recommended additional asset:

- `MLB_Game_Stats_Joined_2021_2025.sqlite`

## Integrity

After uploading the release assets, compute SHA-256 for the compressed CSV and uncompressed CSV and populate:

`data/external/MLB_Game_Stats_Joined_2021_2025.manifest.json`

The repository fetcher intentionally refuses to use the asset until a checksum is present and verified.

## Retrieval

Once the release asset and checksum are present:

```bash
python src/ingestion/fetch_historical_joined_master.py
```

The fetcher downloads the versioned release asset and verifies SHA-256 before exposing it to downstream analysis.

## Versioning rule

Never silently replace a historical master release asset. Any source correction, reconciliation change, added market history, or schema change requires a new version/tag and a new manifest version. Derived tables remain under `data/derived/i2/` and should record the input dataset version used to create them.

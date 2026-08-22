# Count-State Historical Analysis Package

This package builds the missing historical pitch-count state layer for the MLB live half-inning model.

## What it produces

The core table estimates the probability distribution of **additional runs remaining in the half inning** at each:

`outs × base occupancy × balls × strikes`

state.

It also estimates PA terminal-outcome frequencies by count.

## Run

```bash
python -m pip install -r requirements-count-state.txt
python scripts/build_count_state_dataset.py \
  --start 2021-04-01 \
  --end 2025-09-30 \
  --out data/derived/count_state
```

For a pre-downloaded Statcast parquet:

```bash
python scripts/build_count_state_dataset.py \
  --input-parquet /path/to/statcast_2021_2025.parquet \
  --out data/derived/count_state
```

## Important

- Research/challenger only.
- No sportsbook data is used.
- Existing 2021–2025 empirical base/out transition engine remains preserved.
- Do not replace production count logic until walk-forward validation is complete.
- Full confidence intervals should use half-inning clustered bootstrap because pitch-state observations within a half inning are correlated.

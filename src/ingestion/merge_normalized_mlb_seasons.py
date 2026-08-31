#!/usr/bin/env python3
"""Merge per-season normalized MLB tables into one reusable historical dataset."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

TABLES = ["games", "plate_appearances", "lineups", "starters", "results", "inning_results"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", type=Path, required=True,
                   help="Root containing one directory per season")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read_one(path: Path) -> pd.DataFrame:
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    if path.with_suffix(".csv").exists():
        return pd.read_csv(path.with_suffix(".csv"))
    raise FileNotFoundError(path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    season_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir()])
    if not season_dirs:
        raise SystemExit(f"No season directories under {args.input_root}")

    counts = {}
    for table in TABLES:
        parts = []
        for season_dir in season_dirs:
            stem = season_dir / table
            try:
                x = read_one(stem)
            except FileNotFoundError:
                continue
            x["source_partition"] = season_dir.name
            parts.append(x)
        if not parts:
            raise SystemExit(f"No data found for required table {table}")
        out = pd.concat(parts, ignore_index=True)
        key = [c for c in ["game_id","play_index","inning","half_inning","team_side","player_id","pitcher_id"] if c in out.columns]
        if key:
            out = out.drop_duplicates(subset=key, keep="last")
        out.to_parquet(args.output_dir / f"{table}.parquet", index=False)
        counts[table] = int(len(out))

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_partitions": [p.name for p in season_dirs],
        "tables": counts,
        "market_data_used": False,
        "purpose": "model-agnostic historical baseball source layer",
    }
    (args.output_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

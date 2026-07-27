#!/usr/bin/env python3
"""Audit normalized historical data before model execution."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path, required=True)
    args = p.parse_args()

    checks = []
    for name in ["games", "plate_appearances", "lineups", "starters", "results"]:
        path = args.processed_dir / f"{name}.parquet"
        exists = path.exists()
        rows = len(pd.read_parquet(path)) if exists else 0
        checks.append((name, exists, rows))

    pa = pd.read_parquet(args.processed_dir / "plate_appearances.parquet")
    valid_events = {
        "single", "double", "triple", "home_run", "walk",
        "hit_by_pitch", "strikeout", "ball_in_play_out"
    }
    invalid = sorted(set(pa["event"].dropna()) - valid_events)

    for name, exists, rows in checks:
        print(f"{name}: exists={exists} rows={rows}")
    print(f"invalid_events={invalid}")

    if any(not exists or rows == 0 for _, exists, rows in checks):
        raise SystemExit("Audit failed: one or more required tables are empty")
    if invalid:
        raise SystemExit("Audit failed: invalid normalized events")
    print("Historical data audit passed")

if __name__ == "__main__":
    main()

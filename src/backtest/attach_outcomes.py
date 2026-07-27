#!/usr/bin/env python3
"""Attach post-game outcomes to prediction rows after predictions are frozen."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    pred = pd.read_csv(args.predictions)
    results = pd.read_csv(args.results)

    required = {"game_id", "home_runs", "away_runs"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Missing result fields: {sorted(missing)}")

    merged = pred.merge(
        results[["game_id", "home_runs", "away_runs"]],
        on="game_id",
        how="inner",
        validate="many_to_one",
    )
    merged["actual_total_runs"] = merged["home_runs"] + merged["away_runs"]
    merged["home_win_outcome"] = (merged["home_runs"] > merged["away_runs"]).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()

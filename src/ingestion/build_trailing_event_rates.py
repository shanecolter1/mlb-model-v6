#!/usr/bin/env python3
"""Build leakage-safe trailing event-rate snapshots from plate-appearance data.

Expected input columns:
date, game_id, batter_id, pitcher_id, batting_team, pitching_team, event

Supported event labels are normalized to:
single, double, triple, home_run, walk, hit_by_pitch, strikeout, ball_in_play_out

This script deliberately uses only rows strictly before each snapshot date.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

EVENT_MAP = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    "walk": "walk",
    "intent_walk": "walk",
    "hit_by_pitch": "hit_by_pitch",
    "strikeout": "strikeout",
    "strikeout_double_play": "strikeout",
}

EVENTS = [
    "single", "double", "triple", "home_run", "walk",
    "hit_by_pitch", "strikeout", "ball_in_play_out"
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window-days", type=int, default=365)
    p.add_argument("--min-pa", type=int, default=50)
    return p.parse_args()

def normalize_event(value: str) -> str:
    return EVENT_MAP.get(str(value), "ball_in_play_out")

def rate_table(frame: pd.DataFrame, entity_col: str, cutoff: pd.Timestamp,
               window_days: int, min_pa: int) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=window_days)
    hist = frame[(frame["date"] < cutoff) & (frame["date"] >= start)].copy()
    counts = (
        hist.groupby([entity_col, "normalized_event"])
        .size().unstack(fill_value=0)
    )
    for event in EVENTS:
        if event not in counts.columns:
            counts[event] = 0
    counts["pa"] = counts[EVENTS].sum(axis=1)
    league = counts[EVENTS].sum() / counts["pa"].sum()

    # Transparent empirical-Bayes shrinkage to the league mean.
    strength = float(min_pa)
    rates = counts[EVENTS].add(league * strength, axis=1)
    rates = rates.div(counts["pa"] + strength, axis=0)
    rates["pa"] = counts["pa"]
    rates["as_of"] = cutoff.isoformat()
    return rates.reset_index()

def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"])
    df["normalized_event"] = df["event"].map(normalize_event)

    snapshots = []
    for cutoff in sorted(df["date"].dt.normalize().unique()):
        cutoff = pd.Timestamp(cutoff)
        batter = rate_table(df, "batter_id", cutoff, args.window_days, args.min_pa)
        batter["entity_type"] = "batter"
        batter = batter.rename(columns={"batter_id": "entity_id"})

        pitcher = rate_table(df, "pitcher_id", cutoff, args.window_days, args.min_pa)
        pitcher["entity_type"] = "pitcher"
        pitcher = pitcher.rename(columns={"pitcher_id": "entity_id"})

        snapshots.extend([batter, pitcher])

    out = pd.concat(snapshots, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

if __name__ == "__main__":
    main()

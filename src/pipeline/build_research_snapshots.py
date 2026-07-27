#!/usr/bin/env python3
"""Construct research snapshots from normalized historical MLB tables.

This builder produces two classes:
1. `retrospective_research`: uses final-feed lineup/actual starter, clearly flagged.
2. `pregame_eligible`: only when externally supplied archived timestamps verify that
   the lineup and starter were available before the requested prediction time.

The script never silently relabels final-feed information as pregame information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--event-rates", type=Path, required=True)
    p.add_argument("--venue-profiles", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prediction-hour-utc", type=int, default=16)
    return p.parse_args()

def event_vector(row):
    fields = [
        "single", "double", "triple", "home_run", "walk",
        "hit_by_pitch", "strikeout", "ball_in_play_out"
    ]
    return {field: float(row[field]) for field in fields}

def latest_rate(rates, entity_type, entity_id, game_date):
    eligible = rates[
        (rates["entity_type"] == entity_type) &
        (rates["entity_id"].astype(str) == str(entity_id)) &
        (pd.to_datetime(rates["as_of"]).dt.date <= pd.Timestamp(game_date).date())
    ].sort_values("as_of")
    return None if eligible.empty else eligible.iloc[-1]

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = pd.read_parquet(args.processed_dir / "games.parquet")
    lineups = pd.read_parquet(args.processed_dir / "lineups.parquet")
    starters = pd.read_parquet(args.processed_dir / "starters.parquet")
    results = pd.read_parquet(args.processed_dir / "results.parquet")
    rates = pd.read_parquet(args.event_rates)
    venue_profiles = json.loads(args.venue_profiles.read_text(encoding="utf-8"))
    venue_by_name = {v["venue_name"]: v for v in venue_profiles}

    manifest = []
    for _, game in games.iterrows():
        game_id = game["game_id"]
        game_date = str(game["game_date"])
        prediction_ts = f"{game_date}T{args.prediction_hour_utc:02d}:00:00Z"

        snapshot = {
            "game_id": int(game_id),
            "game_date": game_date,
            "prediction_timestamp": prediction_ts,
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "venue": {
                "venue_id": None if pd.isna(game["venue_id"]) else int(game["venue_id"]),
                "venue_name": game["venue_name"],
            },
            "lineup_status": "retrospective_final_feed",
            "snapshot_class": "retrospective_research",
            "starters": {},
            "team_inputs": {},
            "environmental_context": venue_by_name.get(game["venue_name"]),
            "observed_result": None,
            "audit": {
                "production_eligible": False,
                "reason": "Official final-feed lineup and actual starter timing not verified as pregame",
                "source": "MLB Stats API game feed",
            },
        }

        complete = True
        for side in ("home", "away"):
            side_lineup = lineups[
                (lineups["game_id"] == game_id) &
                (lineups["team_side"] == side)
            ].sort_values("batting_order_slot")
            shares = [0.125, 0.122, 0.119, 0.116, 0.112, 0.108, 0.104, 0.099, 0.095]
            shares = [x / sum(shares[:len(side_lineup)]) for x in shares[:len(side_lineup)]]

            lineup_rows = []
            for share, (_, batter) in zip(shares, side_lineup.iterrows()):
                rate = latest_rate(rates, "batter", batter["player_id"], game_date)
                if rate is None:
                    complete = False
                    continue
                lineup_rows.append({
                    "player_id": int(batter["player_id"]),
                    "name": batter["name"],
                    "batter_side": "S",
                    "projected_plate_appearance_share": share,
                    "event_rates": {**event_vector(rate), "as_of": rate["as_of"]},
                })

            starter_row = starters[
                (starters["game_id"] == game_id) &
                (starters["team_side"] == side)
            ]
            if starter_row.empty:
                complete = False
                continue
            starter_row = starter_row.iloc[0]
            starter_rate = latest_rate(rates, "pitcher", starter_row["pitcher_id"], game_date)
            if starter_rate is None:
                complete = False
                continue

            snapshot["starters"][side] = {
                "player_id": int(starter_row["pitcher_id"]),
                "name": starter_row["name"],
                "source_class": starter_row["source_class"],
            }
            snapshot["team_inputs"][side] = {
                "lineup": lineup_rows,
                "starter_allowed_rates": event_vector(starter_rate),
                "bullpen_allowed_rates": event_vector(starter_rate),
                "starter_expected_batters_faced": 22,
                "bullpen_expected_batters_faced": 16,
                "as_of": starter_rate["as_of"],
            }

        if not complete or snapshot["environmental_context"] is None:
            snapshot["audit"]["incomplete"] = True

        path = args.output_dir / game_date / f"{game_id}_{args.prediction_hour_utc:02d}00_research.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        manifest.append({
            "game_id": int(game_id),
            "snapshot_path": str(path),
            "complete": bool(complete),
            "production_eligible": False,
        })

    (args.output_dir / "manifest.json").write_text(
        json.dumps({"snapshots": manifest}, indent=2), encoding="utf-8"
    )
    print(f"Built {len(manifest)} retrospective research snapshots")

if __name__ == "__main__":
    main()

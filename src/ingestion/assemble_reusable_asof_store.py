#!/usr/bin/env python3
"""Assemble modular game/player/team tables from leakage-safe as-of rates.

The output is intentionally not tied to any one betting/model target. Final-feed
lineup and starter identities are preserved with explicit retrospective timing
flags; their underlying statistics are as-of safe, but the identities themselves
are not promoted to pregame-safe without independent archived timing evidence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--entity-asof", type=Path, required=True)
    p.add_argument("--team-asof", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def read(root: Path, name: str):
    p = root / f"{name}.parquet"
    return pd.read_parquet(p)


def norm_date(df, col="game_date"):
    x = df.copy(); x[col] = pd.to_datetime(x[col], errors="coerce").dt.normalize(); return x


def select_rate_cols(df, prefix="365d"):
    keep = [c for c in df.columns if c in {"entity_id","team_id","entity_type","team_role","as_of_date"} or c.startswith(prefix + "_")]
    return df[keep].copy()


def main():
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    games = norm_date(read(args.processed_dir,"games"))
    lineups = norm_date(read(args.processed_dir,"lineups"))
    starters = norm_date(read(args.processed_dir,"starters"))
    results = norm_date(read(args.processed_dir,"results"))
    innings = norm_date(read(args.processed_dir,"inning_results"))
    entity = pd.read_parquet(args.entity_asof)
    entity["as_of_date"] = pd.to_datetime(entity["as_of_date"], errors="coerce").dt.normalize()
    team = pd.read_parquet(args.team_asof)
    team["as_of_date"] = pd.to_datetime(team["as_of_date"], errors="coerce").dt.normalize()

    batter = select_rate_cols(entity[entity["entity_type"]=="batter"]).drop(columns=["entity_type"])
    pitcher = select_rate_cols(entity[entity["entity_type"]=="pitcher"]).drop(columns=["entity_type"])

    lineup_asof = lineups.merge(
        batter, left_on=["player_id","game_date"], right_on=["entity_id","as_of_date"], how="left"
    ).drop(columns=["entity_id","as_of_date"], errors="ignore")
    lineup_asof["identity_timing_class"] = "retrospective_final_feed_unverified_pregame"
    lineup_asof["statistics_timing_class"] = "asof_safe_strictly_prior_date"

    starter_asof = starters.merge(
        pitcher, left_on=["pitcher_id","game_date"], right_on=["entity_id","as_of_date"], how="left"
    ).drop(columns=["entity_id","as_of_date"], errors="ignore")
    starter_asof["identity_timing_class"] = "retrospective_actual_first_pitcher_unverified_pregame"
    starter_asof["statistics_timing_class"] = "asof_safe_strictly_prior_date"

    batting = select_rate_cols(team[team["team_role"]=="batting"]).drop(columns=["team_role"])
    pitching = select_rate_cols(team[team["team_role"]=="pitching_allowed"]).drop(columns=["team_role"])
    team_game_rows = []
    for side in ("home","away"):
        base = games[["game_id","game_date",f"{side}_team_id",f"{side}_team"]].rename(columns={f"{side}_team_id":"team_id",f"{side}_team":"team_name"}).copy()
        base["side"] = side
        base = base.merge(batting, left_on=["team_id","game_date"], right_on=["team_id","as_of_date"], how="left", suffixes=("","_bat"))
        base = base.drop(columns=["as_of_date"], errors="ignore")
        base = base.merge(pitching, left_on=["team_id","game_date"], right_on=["team_id","as_of_date"], how="left", suffixes=("_batting","_pitching_allowed"))
        base = base.drop(columns=["as_of_date"], errors="ignore")
        team_game_rows.append(base)
    game_team_asof = pd.concat(team_game_rows, ignore_index=True)
    game_team_asof["statistics_timing_class"] = "asof_safe_strictly_prior_date"

    game_index = games[[c for c in ["game_id","game_date","game_datetime","home_team_id","away_team_id","home_team","away_team","venue_id","venue_name","status"] if c in games.columns]].copy()
    outcomes = results.merge(innings, on=["game_id","game_date"], how="left", suffixes=("_game","_inning"))

    outputs = {
        "game_index": game_index,
        "lineup_asof": lineup_asof,
        "starter_asof": starter_asof,
        "game_team_asof": game_team_asof,
        "inning_outcomes": outcomes,
    }
    for name, df in outputs.items():
        df.to_parquet(args.output_dir / f"{name}.parquet", index=False)

    manifest = {
        "version":"1.1-research-modular",
        "built_at_utc":datetime.now(timezone.utc).isoformat(),
        "strict_stat_cutoff":"source_event_date < target_game_date",
        "same_day_prior_games_included":False,
        "market_data_used":False,
        "tables":{k:int(len(v)) for k,v in outputs.items()},
        "identity_governance":{
            "lineup":"retrospective final-feed; not pregame-safe unless independently timestamp-verified",
            "starter":"actual first pitcher from final feed; not pregame-safe unless independently timestamp-verified"
        },
        "designed_for_reuse":["inning scoring","half-inning scoring","full-game totals","F5","team totals","pitcher-hitter matchup studies","bullpen research","general baseball forecasting"]
    }
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Normalize MLB game-feed JSON into reusable modeling tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

FINAL_STATES = {"Final", "Game Over", "Completed Early"}
EVENT_MAP = {
    "single": "single", "double": "double", "triple": "triple",
    "home_run": "home_run", "walk": "walk", "intent_walk": "walk",
    "hit_by_pitch": "hit_by_pitch", "strikeout": "strikeout",
    "strikeout_double_play": "strikeout",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_event(event) -> str:
    return EVENT_MAP.get(str(event), "ball_in_play_out")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    games, plate_appearances, lineups, starters, results, inning_results = [], [], [], [], [], []

    for feed_path in sorted((args.raw_dir / "feeds").glob("*.json")):
        feed = load(feed_path)
        game_pk = feed.get("gamePk")
        gd = feed.get("gameData", {})
        ld = feed.get("liveData", {})
        dt = gd.get("datetime", {})
        teams = gd.get("teams", {})
        venue = gd.get("venue", {})
        status = gd.get("status", {})
        official_date = dt.get("officialDate")
        game_row = {
            "game_id": game_pk, "game_date": official_date,
            "game_datetime": dt.get("dateTime"), "status": status.get("detailedState"),
            "home_team_id": teams.get("home", {}).get("id"), "home_team": teams.get("home", {}).get("name"),
            "away_team_id": teams.get("away", {}).get("id"), "away_team": teams.get("away", {}).get("name"),
            "venue_id": venue.get("id"), "venue_name": venue.get("name"),
            "raw_feed_path": str(feed_path),
        }
        games.append(game_row)

        box_teams = ld.get("boxscore", {}).get("teams", {})
        for side in ("home", "away"):
            sb = box_teams.get(side, {})
            players = sb.get("players", {})
            team_id = game_row[f"{side}_team_id"]
            for slot, pid in enumerate(sb.get("battingOrder", []), start=1):
                p = players.get(f"ID{pid}", {})
                lineups.append({
                    "game_id": game_pk, "game_date": official_date, "team_side": side,
                    "team_id": team_id, "batting_order_slot": slot, "player_id": pid,
                    "name": p.get("person", {}).get("fullName"),
                    "source_class": "official_final_game_feed",
                    "pregame_availability_unverified": True,
                    "plate_appearances": p.get("batting", {}).get("plateAppearances"),
                })
            pitchers = sb.get("pitchers", [])
            if pitchers:
                pid = pitchers[0]
                p = players.get(f"ID{pid}", {})
                starters.append({
                    "game_id": game_pk, "game_date": official_date, "team_side": side,
                    "team_id": team_id, "pitcher_id": pid,
                    "name": p.get("person", {}).get("fullName"),
                    "source_class": "actual_first_pitcher_in_final_feed",
                    "pregame_availability_unverified": True,
                })

        for play_index, play in enumerate(ld.get("plays", {}).get("allPlays", [])):
            result = play.get("result", {})
            matchup = play.get("matchup", {})
            about = play.get("about", {})
            half = about.get("halfInning")
            batting_side = "away" if half == "top" else "home"
            pitching_side = "home" if half == "top" else "away"
            plate_appearances.append({
                "game_id": game_pk, "game_date": official_date, "play_index": play_index,
                "inning": about.get("inning"), "half_inning": half,
                "batting_team_id": game_row.get(f"{batting_side}_team_id"),
                "pitching_team_id": game_row.get(f"{pitching_side}_team_id"),
                "batter_id": matchup.get("batter", {}).get("id"),
                "batter_name": matchup.get("batter", {}).get("fullName"),
                "batter_side": matchup.get("batSide", {}).get("code"),
                "pitcher_id": matchup.get("pitcher", {}).get("id"),
                "pitcher_name": matchup.get("pitcher", {}).get("fullName"),
                "pitcher_hand": matchup.get("pitchHand", {}).get("code"),
                "raw_event": result.get("eventType"), "event": normalize_event(result.get("eventType")),
                "rbi": result.get("rbi"), "is_complete": about.get("isComplete"),
            })

        linescore = ld.get("linescore", {})
        for inn in linescore.get("innings", []):
            inning = inn.get("num")
            home = inn.get("home", {}) or {}
            away = inn.get("away", {}) or {}
            home_runs = home.get("runs")
            away_runs = away.get("runs")
            inning_results.append({
                "game_id": game_pk, "game_date": official_date, "inning": inning,
                "away_runs": away_runs, "home_runs": home_runs,
                "away_half_played": away_runs is not None,
                "home_half_played": home_runs is not None,
                "full_inning_runs": (away_runs or 0) + (home_runs or 0),
                "source_class": "official_final_linescore",
            })

        home_runs = linescore.get("teams", {}).get("home", {}).get("runs")
        away_runs = linescore.get("teams", {}).get("away", {}).get("runs")
        if status.get("detailedState") in FINAL_STATES and home_runs is not None and away_runs is not None:
            results.append({
                "game_id": game_pk, "game_date": official_date,
                "home_runs": home_runs, "away_runs": away_runs,
                "home_win": int(home_runs > away_runs), "run_margin": home_runs - away_runs,
                "total_runs": home_runs + away_runs,
            })

    tables = {
        "games": games, "plate_appearances": plate_appearances, "lineups": lineups,
        "starters": starters, "results": results, "inning_results": inning_results,
    }
    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        df.to_parquet(args.output_dir / f"{name}.parquet", index=False)
        df.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(name, len(df))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Archive pregame team standings context for temporal I2 research.

Each game-date row requests the previous calendar day's MLB standings.
The output preserves raw rank and games-back fields with explicit missing
values. No playoff outcome, future result, sportsbook input, or market price
is used. This is a candidate input, not a fitted playoff effect.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE = "https://statsapi.mlb.com/api/v1/standings"
FIELDS = ("game_date", "standings_asof_date", "season", "team_id",
          "division_id", "wins", "losses", "winning_pct", "games_back",
          "wild_card_games_back", "division_rank", "wild_card_rank",
          "league_rank", "clinched", "elimination_number", "magic_number")


def fetch_snapshot(season: int, asof: date, attempts: int = 3) -> dict:
    params = urlencode({"leagueId": "103,104", "season": season,
                        "standingsTypes": "regularSeason", "date": asof.isoformat()})
    request = Request(f"{BASE}?{params}", headers={"User-Agent": "MLB-I2-Temporal/1.0",
                                                   "Accept": "application/json"})
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def parse_snapshot(payload: dict, game_date: date) -> list[dict]:
    asof = game_date - timedelta(days=1)
    records = []
    seen = set()
    for division in payload.get("records", []):
        division_id = division.get("division", {}).get("id")
        for team in division.get("teamRecords", []):
            team_id = team.get("team", {}).get("id")
            if team_id is None or team_id in seen:
                raise ValueError("Missing or duplicated team in standings")
            seen.add(team_id)
            records.append({
                "game_date": game_date.isoformat(),
                "standings_asof_date": asof.isoformat(),
                "season": game_date.year,
                "team_id": int(team_id), "division_id": division_id,
                "wins": team.get("wins"), "losses": team.get("losses"),
                "winning_pct": team.get("winningPercentage"),
                "games_back": team.get("gamesBack"),
                "wild_card_games_back": team.get("wildCardGamesBack"),
                "division_rank": team.get("divisionRank"),
                "wild_card_rank": team.get("wildCardRank"),
                "league_rank": team.get("leagueRank"),
                "clinched": team.get("clinched"),
                "elimination_number": team.get("eliminationNumber"),
                "magic_number": team.get("magicNumber"),
            })
    if records and len(records) != 30:
        raise ValueError(f"Incomplete as-of standings: {len(records)} teams")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-daily", type=Path, required=True,
                        help="Prior-day league CSV defining regular-season dates")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0.1)
    args = parser.parse_args()
    with args.league_daily.open(newline="") as handle:
        dates = [date.fromisoformat(row["game_date"]) for row in csv.DictReader(handle)
                 if int(row["games_on_date"]) > 0]
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("Invalid league game dates")
    season = dates[0].year
    if any(day.year != season for day in dates):
        raise ValueError("Mixed seasons")
    rows = []
    missing_dates = []
    for day in dates:
        snapshot = parse_snapshot(fetch_snapshot(season, day - timedelta(days=1)), day)
        if not snapshot:
            missing_dates.append(day.isoformat())
        rows.extend(snapshot)
        time.sleep(max(0.0, args.sleep))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"season": season, "game_dates": len(dates),
                      "complete_standings_dates": len(dates) - len(missing_dates),
                      "missing_dates": missing_dates,
                      "rows": len(rows), "market_inputs_used": False,
                      "same_day_outcomes_used": False}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch historical MLB schedule and game-feed data from MLB Stats API.

This adapter downloads:
- schedule metadata
- venue
- probable/actual starters when available
- official batting orders from game feeds
- plate-appearance outcomes
- final scores

It stores raw JSON unchanged before producing normalized tables.

Important:
- Historical feeds describe what ultimately happened. For a true pregame model,
  batting orders and announced starters must be filtered by an archived
  prediction timestamp. This adapter therefore tags fields with source timing
  and does not automatically claim they were known pregame.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

BASE = "https://statsapi.mlb.com/api"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sleep", type=float, default=0.15)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def get_json(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> dict:
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()

def save_json(path: Path, payload: dict, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def main():
    args = parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "MLB-V6-Research/0.1",
        "Accept-Encoding": "gzip",
    })

    manifest = []
    for game_date in daterange(start, end):
        schedule_url = f"{BASE}/v1/schedule"
        schedule = get_json(session, schedule_url, {
            "sportId": 1,
            "date": game_date.isoformat(),
            "gameTypes": "R,F,D,L,W",
            "hydrate": "venue,probablePitcher,linescore",
        })
        schedule_path = args.output_dir / "schedule" / f"{game_date}.json"
        save_json(schedule_path, schedule, args.overwrite)

        for date_block in schedule.get("dates", []):
            for game in date_block.get("games", []):
                game_pk = game["gamePk"]
                status = game.get("status", {}).get("detailedState")
                feed_url = f"{BASE}/v1.1/game/{game_pk}/feed/live"
                feed_path = args.output_dir / "feeds" / f"{game_pk}.json"
                if args.overwrite or not feed_path.exists():
                    feed = get_json(session, feed_url)
                    save_json(feed_path, feed, True)
                    time.sleep(args.sleep)

                manifest.append({
                    "game_id": game_pk,
                    "game_date": game_date.isoformat(),
                    "status": status,
                    "schedule_path": str(schedule_path),
                    "feed_path": str(feed_path),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                })

    manifest_path = args.output_dir / "fetch_manifest.json"
    save_json(manifest_path, {"games": manifest}, True)
    print(f"Fetched or indexed {len(manifest)} games")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch historical MLB schedule and game-feed data from MLB Stats API.

Raw feeds are preserved unchanged. Historical final feeds describe what ultimately
happened; lineup/starter fields therefore remain retrospective unless independent
archived timing verifies they were available before the target prediction time.
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
    p.add_argument("--game-types", default="R", help="MLB Stats API gameTypes, comma-delimited if needed")
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
    session.headers.update({"User-Agent": "MLB-V6-Research/0.2", "Accept-Encoding": "gzip"})

    manifest = []
    for game_date in daterange(start, end):
        schedule = get_json(session, f"{BASE}/v1/schedule", {
            "sportId": 1,
            "date": game_date.isoformat(),
            "gameTypes": args.game_types,
            "hydrate": "venue,probablePitcher,linescore",
        })
        schedule_path = args.output_dir / "schedule" / f"{game_date}.json"
        save_json(schedule_path, schedule, args.overwrite)

        for date_block in schedule.get("dates", []):
            for game in date_block.get("games", []):
                game_pk = game["gamePk"]
                status = game.get("status", {}).get("detailedState")
                feed_path = args.output_dir / "feeds" / f"{game_pk}.json"
                if args.overwrite or not feed_path.exists():
                    feed = get_json(session, f"{BASE}/v1.1/game/{game_pk}/feed/live")
                    save_json(feed_path, feed, True)
                    time.sleep(args.sleep)
                manifest.append({
                    "game_id": game_pk,
                    "game_date": game_date.isoformat(),
                    "status": status,
                    "game_types": args.game_types,
                    "schedule_path": str(schedule_path),
                    "feed_path": str(feed_path),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                })

    save_json(args.output_dir / "fetch_manifest.json", {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "game_types": args.game_types,
        "games": manifest,
    }, True)
    print(f"Fetched or indexed {len(manifest)} games")


if __name__ == "__main__":
    main()

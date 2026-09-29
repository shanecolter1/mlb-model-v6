#!/usr/bin/env python3
"""Build prior-day MLB-wide event-rate features for temporal I2 research.

Reads completed regular-season MLB Stats API feeds, all innings and all
terminal plate appearances. A row dated D contains observations through D-1
only. It contains raw rates and exposure counts; joint shrinkage and feature
selection are deliberately deferred to the forecasting comparison.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from build_i2_vnext_replay_inputs_mlb import EVENT_KEYS, norm_event


def feed_games(raw_dir: Path) -> tuple[dict[date, Counter], int]:
    manifest = json.loads((raw_dir / "fetch_manifest.json").read_text())
    daily: dict[date, Counter] = defaultdict(Counter)
    seen = set()
    for rec in manifest["games"]:
        game_id = int(rec["game_id"])
        if game_id in seen:
            raise ValueError(f"Duplicate game {game_id}")
        seen.add(game_id)
        path = Path(rec["feed_path"])
        if not path.is_absolute() and not path.exists():
            path = raw_dir / path
        feed = json.loads(path.read_text())
        gd = feed["gameData"]
        if int(feed["gamePk"]) != game_id:
            raise ValueError(f"Feed/manifest game mismatch: {game_id}")
        game_type = gd.get("game", {}).get("type") or gd.get("game", {}).get("gameType")
        if str(game_type) != "R":
            raise ValueError(f"Non-regular-season feed: {game_id}")
        state = str(gd["status"].get("abstractGameState") or "").lower()
        if state != "final":
            raise ValueError(f"Non-final feed: {game_id}")
        game_date = date.fromisoformat(str(gd["datetime"]["officialDate"])[:10])
        if game_date.isoformat() != rec["game_date"]:
            raise ValueError(f"Date mismatch: {game_id}")
        plays = feed["liveData"]["plays"]["allPlays"]
        if not plays:
            raise ValueError(f"No plays: {game_id}")
        for play in plays:
            if not play.get("about", {}).get("isComplete"):
                continue
            event = norm_event(play.get("result", {}).get("eventType"))
            if event in EVENT_KEYS:
                daily[game_date][event] += 1
        daily[game_date]["games"] += 1
    return daily, len(seen)


def build_features(daily: dict[date, Counter], windows: tuple[int, ...]) -> pd.DataFrame:
    if not daily:
        raise ValueError("No games")
    start, end = min(daily), max(daily)
    rows = []
    current = start
    while current <= end:
        row = {"game_date": current.isoformat(),
               "games_on_date": daily[current]["games"]}
        for days in windows:
            previous = Counter()
            for lag in range(1, days + 1):
                previous.update(daily.get(current - timedelta(days=lag), Counter()))
            pa = sum(previous[event] for event in EVENT_KEYS)
            row[f"prior_{days}d_games"] = previous["games"]
            row[f"prior_{days}d_pa"] = pa
            for event in EVENT_KEYS:
                row[f"prior_{days}d_{event}_count"] = previous[event]
                row[f"prior_{days}d_{event}_rate"] = previous[event] / pa if pa else None
            bip = sum(previous[event] for event in (
                "single", "double", "triple", "home_run", "ball_in_play_out"))
            row[f"prior_{days}d_bip"] = bip
            row[f"prior_{days}d_hr_per_bip"] = previous["home_run"] / bip if bip else None
        rows.append(row)
        current += timedelta(days=1)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="14,28,56")
    args = parser.parse_args()
    windows = tuple(sorted({int(n) for n in args.windows.split(",")}))
    if not windows or any(n < 1 for n in windows):
        raise ValueError("Positive prior-day windows required")
    daily, games = feed_games(args.raw_dir)
    frame = build_features(daily, windows)
    if frame.games_on_date.sum() != games:
        raise ValueError("Game coverage mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps({"games": games, "dates": len(frame), "start": str(min(daily)),
                      "end": str(max(daily)), "windows": windows,
                      "market_inputs_used": False, "same_day_outcomes_used": False}))


if __name__ == "__main__":
    main()

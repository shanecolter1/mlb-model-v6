#!/usr/bin/env python3
"""Extract Savant's four-seam drag series and lagged pregame summaries.

The public dashboard is a retrospective snapshot. A calendar lag prevents
same-day outcome use, but does not establish historical publication timing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://baseballsavant.mlb.com/drag-dashboard"
MARKER = "const serverVals = "


def extract(html: str) -> list[dict]:
    if html.count(MARKER) != 1:
        raise ValueError("Savant serverVals missing or ambiguous")
    vals, _ = json.JSONDecoder().raw_decode(html.split(MARKER, 1)[1])
    rows = []
    seen = set()
    for item in vals["scatterData"]:
        day = date.fromisoformat(item["game_date"][:10])
        if day in seen:
            raise ValueError(f"Duplicate drag date: {day}")
        seen.add(day)
        if day.year != item["year"] or item["num_pitches"] <= 0:
            raise ValueError(f"Invalid drag observation: {day}")
        rows.append({"date": day.isoformat(), "year": day.year,
                     "mean_cd": item["mean_cd"],
                     "num_pitches": item["num_pitches"],
                     "num_games": item["num_games"]})
    return sorted(rows, key=lambda r: r["date"])


def asof(raw: list[dict], game_dates: list[date], lag: int) -> list[dict]:
    result = []
    for day in game_dates:
        end = day - timedelta(days=lag)
        row = {"game_date": day.isoformat(), "drag_asof_date": end.isoformat(),
               "drag_lag_days": lag}
        for window in (14, 28, 56):
            start = end - timedelta(days=window-1)
            obs = [r for r in raw if start <= date.fromisoformat(r["date"]) <= end
                   and r["year"] == day.year]
            n = sum(r["num_pitches"] for r in obs)
            row[f"drag_cd_{window}d"] = (sum(r["mean_cd"] * r["num_pitches"]
                                              for r in obs) / n) if n else ""
            row[f"drag_pitches_{window}d"] = n
            row[f"drag_days_{window}d"] = len(obs)
        result.append(row)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--html", type=Path, help="Saved public dashboard HTML")
    p.add_argument("--league-daily-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--lag-days", type=int, default=2)
    args = p.parse_args()
    if args.lag_days < 1:
        raise ValueError("Drag data must precede the game")
    html = (args.html.read_text() if args.html else
            urlopen(Request(URL, headers={"User-Agent": "Mozilla/5.0"}),
                    timeout=30).read().decode())
    raw = extract(html)
    write_csv(args.output_dir / "drag_daily_snapshot.csv", raw)
    summary = {"source": URL, "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
               "retrospective_snapshot": True, "publication_timing_verified": False,
               "lag_days": args.lag_days, "seasons": {}}
    for year in range(2022, 2027):
        path = args.league_daily_dir / f"league_{year}_asof.csv"
        with path.open(newline="") as handle:
            game_dates = [date.fromisoformat(r["game_date"]) for r in csv.DictReader(handle)
                          if int(r["games_on_date"]) > 0]
        rows = asof(raw, game_dates, args.lag_days)
        write_csv(args.output_dir / f"drag_{year}_asof.csv", rows)
        summary["seasons"][str(year)] = {
            "game_dates": len(rows), "nonempty_28d": sum(bool(r["drag_days_28d"])
                                                       for r in rows)}
    (args.output_dir / "DRAG_SOURCE_MANIFEST.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build normalized Baseball Savant venue profiles.

The script reads Baseball Savant's official Statcast Park Factors leaderboard,
normalizes 100-centered factors to multipliers, and writes an auditable JSON/CSV
snapshot. It is designed for an internet-enabled CI or local environment.

Important:
- The legacy generic park_score is NOT blended into these profiles.
- Weather remains separate.
- Composite fields are retained for diagnostics unless promoted by validation.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

BASE_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"

COLUMN_MAP = {
    "Park Factor": "overall",
    "wOBAcon": "wobacon",
    "xwOBAcon": "xwobacon",
    "BACON": "bacon",
    "xBACON": "xbacon",
    "HardHit": "hard_hit",
    "R": "run",
    "OBP": "obp",
    "H": "hit",
    "1B": "single",
    "2B": "double",
    "3B": "triple",
    "HR": "hr",
    "BB": "bb",
    "SO": "so",
    "PA": "plate_appearances",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--rolling", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    return p.parse_args()

def leaderboard_url(year: int, rolling: int, bat_side: str = "") -> str:
    return (
        f"{BASE_URL}?rolling={rolling}&stat=index_wOBA&type=year"
        f"&year={year}&condition=All&batSide={bat_side}"
    )

def read_leaderboard(url: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    candidates = [t for t in tables if {"Venue", "R", "HR", "PA"}.issubset(set(map(str, t.columns)))]
    if not candidates:
        raise RuntimeError(f"Could not locate park-factor table at {url}")
    return candidates[0].copy()

def number(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def multiplier(value: Any) -> float | None:
    n = number(value)
    return None if n is None else round(n / 100.0, 4)

def confidence_from_pa(pa: float | None, rolling: int) -> float:
    # Transparent initial rule for shadow mode, not a final empirical calibration.
    if pa is None:
        return 0.50
    expected = 16000 * max(1, rolling)
    return round(max(0.45, min(0.99, pa / expected)), 3)

def normalize(df: pd.DataFrame, year: int, rolling: int, side: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    now = datetime.now(timezone.utc).isoformat()
    for _, row in df.iterrows():
        venue = str(row.get("Venue", "")).strip()
        if not venue:
            continue
        pa = number(row.get("PA"))
        profile = {
            "venue_name": venue,
            "team": str(row.get("Team", "")).strip(),
            "window": str(row.get("Year", f"{year-rolling+1}-{year}")),
            "source": "Baseball Savant Statcast Park Factors",
            "source_url": leaderboard_url(year, rolling, side),
            "plate_appearances": int(pa) if pa is not None else None,
            "confidence": confidence_from_pa(pa, rolling),
            "status": "shadow",
            "multipliers": {},
            "audit": {
                "generic_park_score_disabled": True,
                "fallback_reason": None,
                "retrieved_at": now,
                "bat_side": side or "All",
            },
        }
        for source_col, target in COLUMN_MAP.items():
            if target == "plate_appearances":
                continue
            profile["multipliers"][target] = multiplier(row.get(source_col))
        out[venue] = profile
    return out

def merge_splits(all_profiles, left_profiles, right_profiles):
    for venue, profile in all_profiles.items():
        profile["handedness"] = {}
        for side, source in (("L", left_profiles), ("R", right_profiles)):
            split = source.get(venue, {}).get("multipliers", {})
            profile["handedness"][side] = {
                k: split.get(k) for k in ("run", "hr", "single", "double", "triple")
            }
    return all_profiles

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_df = read_leaderboard(leaderboard_url(args.year, args.rolling, ""))
    left_df = read_leaderboard(leaderboard_url(args.year, args.rolling, "L"))
    right_df = read_leaderboard(leaderboard_url(args.year, args.rolling, "R"))

    profiles = merge_splits(
        normalize(all_df, args.year, args.rolling, ""),
        normalize(left_df, args.year, args.rolling, "L"),
        normalize(right_df, args.year, args.rolling, "R"),
    )

    json_path = args.output_dir / f"savant_venue_profiles_{args.year}_{args.rolling}yr.json"
    json_path.write_text(json.dumps(list(profiles.values()), indent=2), encoding="utf-8")

    rows = []
    for p in profiles.values():
        row = {
            "venue_name": p["venue_name"],
            "team": p["team"],
            "window": p["window"],
            "plate_appearances": p["plate_appearances"],
            "confidence": p["confidence"],
        }
        row.update({f"multiplier_{k}": v for k, v in p["multipliers"].items()})
        for side in ("L", "R"):
            row.update({f"{side.lower()}_{k}": v for k, v in p["handedness"][side].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        args.output_dir / f"savant_venue_profiles_{args.year}_{args.rolling}yr.csv",
        index=False,
    )

if __name__ == "__main__":
    main()

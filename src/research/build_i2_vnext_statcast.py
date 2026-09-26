#!/usr/bin/env python3
"""Build the lean Statcast I2 PA dataset used by I2 vNext.

Baseball-only. No sportsbook or market inputs.
Queries Baseball Savant for inning 2 only in monthly chunks and retains
terminal PA rows. Also archives annual pitch-arsenal leaderboards.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

STATCAST_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
ARSENAL_URL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"

MODELED_EVENTS = [
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
]

OUT_EVENTS = {
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "triple_play", "fielders_choice", "fielders_choice_out", "sac_fly",
    "sac_bunt", "strikeout_double_play",
}
WALK_EVENTS = {"walk", "intent_walk", "intentional_walk"}
SINGLE_PROXY_EVENTS = {"field_error"}
WALK_PROXY_EVENTS = {"catcher_interf", "catcher_interference"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-03-01")
    p.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/i2_pa_statcast.csv"))
    p.add_argument("--arsenal-dir", type=Path, default=Path("data/derived/i2_vnext/arsenal"))
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache/i2_vnext/statcast"))
    p.add_argument("--sleep", type=float, default=0.25)
    return p.parse_args()


def month_chunks(start: date, end: date) -> Iterable[tuple[date, date]]:
    cur = start
    while cur <= end:
        nxt = date(cur.year + 1, 1, 1) if cur.month == 12 else date(cur.year, cur.month + 1, 1)
        chunk_end = min(end, nxt - timedelta(days=1))
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def request_csv(url: str, params: dict, retries: int = 4) -> pd.DataFrame:
    last = None
    headers = {"User-Agent": "MLB-I2-vNext/1.0", "Accept": "text/csv,*/*"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=90)
            r.raise_for_status()
            if not r.text.strip():
                return pd.DataFrame()
            return pd.read_csv(io.StringIO(r.text))
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed CSV request after {retries} attempts: {last}")


def statcast_params(start: date, end: date) -> dict[str, str]:
    return {
        "all": "true",
        "type": "details",
        "player_type": "pitcher",
        "hfGT": "R|",
        "hfInn": "2|",
        "game_date_gt": start.isoformat(),
        "game_date_lt": end.isoformat(),
        "group_by": "name",
        "sort_col": "pitches",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "min_abs": "0",
    }


def norm_event(raw: object) -> str | None:
    e = str(raw or "").strip().lower()
    if not e or e == "nan":
        return None
    if e in {"single", "double", "triple", "home_run", "hit_by_pitch", "strikeout"}:
        return e
    if e in WALK_EVENTS:
        return "walk"
    if e in OUT_EVENTS:
        return "strikeout" if e == "strikeout_double_play" else "ball_in_play_out"
    if e in SINGLE_PROXY_EVENTS:
        return "single"
    if e in WALK_PROXY_EVENTS:
        return "walk"
    return None


def terminal_pa_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = ["game_date", "game_pk", "at_bat_number", "batter", "pitcher", "events", "stand", "p_throws"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Statcast schema missing required columns: {missing}")

    x = df[df["events"].notna()].copy()
    x["event_class"] = x["events"].map(norm_event)
    x = x[x["event_class"].isin(MODELED_EVENTS)].copy()
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce")
    x = x[x["game_date"].notna()]
    x["season"] = x["game_date"].dt.year.astype(int)
    x["batter"] = pd.to_numeric(x["batter"], errors="coerce").astype("Int64")
    x["pitcher"] = pd.to_numeric(x["pitcher"], errors="coerce").astype("Int64")
    x = x[x["batter"].notna() & x["pitcher"].notna()]
    x["batter"] = x["batter"].astype(int)
    x["pitcher"] = x["pitcher"].astype(int)
    x["stand"] = x["stand"].fillna("?").astype(str)
    x["p_throws"] = x["p_throws"].fillna("?").astype(str)
    x["platoon"] = x["stand"] + "v" + x["p_throws"]
    keep = [
        "game_date", "season", "game_pk", "at_bat_number", "batter", "pitcher",
        "stand", "p_throws", "platoon", "events", "event_class", "pitch_type",
        "home_team", "away_team",
    ]
    keep = [c for c in keep if c in x.columns]
    return x[keep].drop_duplicates(["game_pk", "at_bat_number"], keep="last")


def build_pa_dataset(start: date, end: date, cache_dir: Path, sleep: float) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    for lo, hi in month_chunks(start, end):
        cache = cache_dir / f"i2_{lo.isoformat()}_{hi.isoformat()}.csv"
        if cache.exists():
            raw = pd.read_csv(cache, low_memory=False)
        else:
            raw = request_csv(STATCAST_URL, statcast_params(lo, hi))
            raw.to_csv(cache, index=False)
            time.sleep(sleep)
        part = terminal_pa_rows(raw)
        if not part.empty:
            parts.append(part)
    if not parts:
        raise RuntimeError("No I2 Statcast PA rows were built")
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["game_date", "game_pk", "at_bat_number"]).reset_index(drop=True)


def build_arsenal(year: int, role: str, out_dir: Path, sleep: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{role}_{year}.csv"
    if path.exists():
        return path
    df = request_csv(ARSENAL_URL, {
        "type": role,
        "pitchType": "",
        "year": str(year),
        "team": "",
        "min": "1",
        "csv": "true",
    })
    expected = {"player_id", "pitch_type", "pitches", "est_woba"}
    if not expected.issubset(df.columns):
        raise RuntimeError(f"Unexpected Savant arsenal schema for {role} {year}: {sorted(df.columns)}")
    df.to_csv(path, index=False)
    time.sleep(sleep)
    return path


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on/after --start")

    pa = build_pa_dataset(start, end, args.cache_dir, args.sleep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pa.to_csv(args.output, index=False)

    years = range(max(2022, start.year - 1), end.year + 1)
    arsenal_files = []
    for year in years:
        for role in ("batter", "pitcher"):
            arsenal_files.append(str(build_arsenal(year, role, args.arsenal_dir, args.sleep)))

    manifest = {
        "version": "i2-vnext-statcast-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": int(len(pa)),
        "games": int(pa["game_pk"].nunique()),
        "batters": int(pa["batter"].nunique()),
        "pitchers": int(pa["pitcher"].nunique()),
        "event_counts": {k: int(v) for k, v in pa["event_class"].value_counts().to_dict().items()},
        "market_inputs_used": False,
        "statcast_scope": "regular-season inning 2 terminal plate appearances only",
        "arsenal_historical_rule": "season Y PA uses season Y-1 arsenal profile during model fitting",
        "arsenal_live_rule": "use current-season YTD profile available at prediction cutoff",
        "proxy_event_mapping": {"field_error": "single", "catcher_interference": "walk"},
        "arsenal_files": arsenal_files,
    }
    args.output.with_name("dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

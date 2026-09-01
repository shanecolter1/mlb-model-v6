#!/usr/bin/env python3
"""Build leakage-safe PA-history counts for the all-inning M1 screen.

The source is limited to normalized 2021-2024 plate appearances.  For every
player-date on which a batter or pitcher appears, the output contains counts
from dates strictly before that date.  Same-day plate appearances are never
included, even for later games of a doubleheader.  Extra-inning PAs may inform
future-date player history, but the downstream targets remain I1-I9 only.

This store deliberately contains counts and league priors rather than a fixed
shrunken rate.  Shrinkage strengths are candidates to be selected by the
chronological validator; none is assumed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEASONS = [2021, 2022, 2023, 2024]
WINDOW_DAYS = {"30d": 30, "90d": 90, "365d": 365, "season": None}
EVENT_METRICS = {
    "k": ["strikeout"],
    "walk": ["walk"],
    "hbp": ["hit_by_pitch"],
    "baserunner": ["walk", "hit_by_pitch"],
    "hr": ["home_run"],
    "single": ["single"],
    "double": ["double"],
    "triple": ["triple"],
    "nonhr_hit": ["single", "double", "triple"],
    "nonhr_xbh": ["double", "triple"],
    "hit": ["single", "double", "triple", "home_run"],
    "xbh": ["double", "triple", "home_run"],
    "bip_out": ["ball_in_play_out"],
    "batted_ball": ["single", "double", "triple", "home_run", "ball_in_play_out"],
    "on_base": ["walk", "hit_by_pitch", "single", "double", "triple", "home_run"],
}
NORMALIZED_EVENTS = {
    "strikeout",
    "walk",
    "hit_by_pitch",
    "home_run",
    "single",
    "double",
    "triple",
    "ball_in_play_out",
}


def find_one(root: Path, filename: str) -> Path:
    hits = list(root.rglob(filename))
    if len(hits) != 1:
        raise RuntimeError(f"{filename} expected once under {root}; found {hits}")
    return hits[0]


def load_normalized(root: Path) -> pd.DataFrame:
    parts = []
    for season in SEASONS:
        season_root = root / f"normalized-mlb-{season}"
        if not season_root.exists():
            season_root = root / str(season)
        path = find_one(season_root, "plate_appearances.parquet")
        frame = pd.read_parquet(
            path,
            columns=["game_id", "game_date", "play_index", "batter_id", "pitcher_id", "event"],
        )
        frame["season"] = season
        parts.append(frame)
    pa = pd.concat(parts, ignore_index=True)
    pa["game_date"] = pd.to_datetime(pa.game_date, errors="raise").dt.normalize()
    pa["season"] = pd.to_numeric(pa.season, errors="raise").astype(int)
    if set(pa.season.unique()) != set(SEASONS) or (pa.season >= 2025).any():
        raise RuntimeError("development source seasons must be exactly 2021-2024")
    observed = set(pa.event.astype(str).unique())
    if not observed.issubset(NORMALIZED_EVENTS):
        raise RuntimeError(f"unexpected normalized PA events: {sorted(observed - NORMALIZED_EVENTS)}")
    if pa.duplicated(["game_id", "play_index"]).any():
        raise RuntimeError("normalized PA key is not unique")
    if pa[["batter_id", "pitcher_id"]].isna().any().any():
        raise RuntimeError("missing player identity in normalized PA source")
    return pa


def add_metric_indicators(pa: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()
    event = out.event.astype(str)
    out["pa"] = np.int8(1)
    for metric, events in EVENT_METRICS.items():
        out[metric] = event.isin(events).astype("int8")
    return out


def prior_window_sums(dates: np.ndarray, values: np.ndarray, days: int | None) -> np.ndarray:
    """Return per-row sums over dates before the current row's date."""
    if len(dates) == 0:
        return np.empty_like(values)
    prefix = np.vstack([np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(values, axis=0)])
    row = np.arange(len(dates), dtype=np.int64)
    if days is None:
        years = pd.DatetimeIndex(dates).year.to_numpy()
        starts = np.empty(len(dates), dtype=np.int64)
        first = 0
        for idx in range(len(dates)):
            if idx == 0 or years[idx] != years[idx - 1]:
                first = idx
            starts[idx] = first
    else:
        lower_dates = dates.astype("datetime64[D]") - np.timedelta64(days, "D")
        starts = np.searchsorted(dates.astype("datetime64[D]"), lower_dates, side="left")
    return prefix[row] - prefix[starts]


def daily_counts(pa: pd.DataFrame, entity_col: str | None) -> pd.DataFrame:
    count_cols = ["pa", *EVENT_METRICS]
    keys = ["game_date"] if entity_col is None else [entity_col, "game_date"]
    return (
        pa.groupby(keys, as_index=False, sort=True)[count_cols]
        .sum()
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )


def build_entity_asof(pa: pd.DataFrame, entity_col: str, label: str) -> pd.DataFrame:
    daily = daily_counts(pa, entity_col)
    value_cols = ["pa", *EVENT_METRICS]
    pieces = []
    for entity_id, group in daily.groupby(entity_col, sort=False):
        group = group.sort_values("game_date", kind="mergesort")
        dates = group.game_date.to_numpy(dtype="datetime64[ns]")
        values = group[value_cols].to_numpy(dtype=np.int64)
        result = pd.DataFrame({entity_col: entity_id, "game_date": group.game_date.to_numpy()})
        for window, days in WINDOW_DAYS.items():
            sums = prior_window_sums(dates, values, days)
            for j, metric in enumerate(value_cols):
                result[f"{label}_{window}_{metric}_count"] = sums[:, j].astype("int32")
        pieces.append(result)
    out = pd.concat(pieces, ignore_index=True)
    if out.duplicated([entity_col, "game_date"]).any():
        raise RuntimeError(f"duplicate {label} as-of key")
    return out


def build_league_asof(pa: pd.DataFrame) -> pd.DataFrame:
    daily = daily_counts(pa, None)
    value_cols = ["pa", *EVENT_METRICS]
    dates = daily.game_date.to_numpy(dtype="datetime64[ns]")
    values = daily[value_cols].to_numpy(dtype=np.int64)
    columns = {"game_date": daily.game_date.to_numpy()}
    for window, days in WINDOW_DAYS.items():
        sums = prior_window_sums(dates, values, days)
        pa_count = sums[:, 0].astype(float)
        columns[f"league_{window}_pa_count"] = sums[:, 0].astype("int32")
        for j, metric in enumerate(EVENT_METRICS, start=1):
            count = sums[:, j]
            columns[f"league_{window}_{metric}_count"] = count.astype("int32")
            columns[f"league_{window}_{metric}_rate"] = np.divide(
                count,
                pa_count,
                out=np.full(len(count), np.nan, dtype=float),
                where=pa_count > 0,
            )
    out = pd.DataFrame(columns)
    if out.duplicated("game_date").any():
        raise RuntimeError("duplicate league as-of key")
    return out


def audit_same_day_exclusion(pa: pd.DataFrame, asof: pd.DataFrame, entity_col: str, label: str) -> None:
    first_dates = pa.groupby(entity_col).game_date.min().rename("first_date").reset_index()
    first = asof.merge(first_dates, on=entity_col, how="inner")
    first = first[first.game_date == first.first_date]
    count_cols = [f"{label}_{window}_pa_count" for window in WINDOW_DAYS]
    if len(first) == 0 or (first[count_cols].to_numpy() != 0).any():
        raise RuntimeError(f"same-day exclusion audit failed for {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pa = add_metric_indicators(load_normalized(args.normalized_root))
    batter = build_entity_asof(pa, "batter_id", "batter")
    pitcher = build_entity_asof(pa, "pitcher_id", "pitcher")
    league = build_league_asof(pa)
    audit_same_day_exclusion(pa, batter, "batter_id", "batter")
    audit_same_day_exclusion(pa, pitcher, "pitcher_id", "pitcher")

    batter.to_parquet(args.output_dir / "m1_pa_history_batter_asof.parquet", index=False)
    pitcher.to_parquet(args.output_dir / "m1_pa_history_pitcher_asof.parquet", index=False)
    league.to_parquet(args.output_dir / "m1_pa_history_league_asof.parquet", index=False)
    manifest = {
        "status": "PASS",
        "architecture": "M1_strictly_prior_date_PA_history_count_store",
        "development_seasons": SEASONS,
        "source_min_date": str(pa.game_date.min().date()),
        "source_max_date": str(pa.game_date.max().date()),
        "source_pa_rows": int(len(pa)),
        "batter_date_rows": int(len(batter)),
        "pitcher_date_rows": int(len(pitcher)),
        "league_date_rows": int(len(league)),
        "windows": list(WINDOW_DAYS),
        "metrics": list(EVENT_METRICS),
        "player_history_counts_only": True,
        "league_prior_rates_materialized": True,
        "fixed_shrinkage_used": False,
        "same_day_history_included": False,
        "doubleheader_same_day_history_included": False,
        "history_includes_prior_date_extra_inning_PA": True,
        "downstream_target_innings": list(range(1, 10)),
        "holdout_season": 2025,
        "holdout_opened": False,
        "market_data_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

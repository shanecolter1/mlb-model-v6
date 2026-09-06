#!/usr/bin/env python3
"""Calibrate price-independent I2 Under/Over qualification thresholds.

Input must contain true chronological out-of-sample/frozen pregame predictions. The
prediction column is P(I2 Over 0.5). Sportsbook I2 prices are intentionally absent.

Outputs cumulative threshold sweeps for Under and Over, year-by-year grading, and a
compact recommendation table. This script does not fit the baseball model; it grades
already-frozen retrospective predictions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, names: Iterable[str]):
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def wilson_interval(wins: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = wins / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def detect_input(df: pd.DataFrame):
    pred_col = pick_col(df, ["prediction", "p_i2_over", "i2_over_probability", "over_probability", "prob_over"])
    runs_col = pick_col(df, ["i2_runs", "__i2_runs", "inning_2_runs", "inning2_runs", "runs_inning_2"])
    y_col = pick_col(df, ["__i2_over", "i2_over", "over_hit", "actual_over"])
    season_col = pick_col(df, ["test_season", "season", "__season", "year"])
    date_col = pick_col(df, ["__game_date", "game_date", "date"])
    game_col = pick_col(df, ["game_id", "game_pk", "gamepk"])

    if not pred_col:
        raise ValueError("Input must contain a frozen pregame P(I2 Over 0.5) prediction column.")
    if not y_col and not runs_col:
        raise ValueError("Input must contain either I2 Over outcome or I2 runs.")
    if not season_col and not date_col:
        raise ValueError("Input must contain season or game date for year-by-year grading.")
    return pred_col, runs_col, y_col, season_col, date_col, game_col


def prepare(df: pd.DataFrame):
    pred_col, runs_col, y_col, season_col, date_col, game_col = detect_input(df)
    out = df.copy()
    out["p_over"] = pd.to_numeric(out[pred_col], errors="coerce")
    if y_col:
        out["actual_over"] = pd.to_numeric(out[y_col], errors="coerce")
    else:
        runs = pd.to_numeric(out[runs_col], errors="coerce")
        out["actual_over"] = np.where(runs.notna(), (runs >= 1).astype(float), np.nan)
    out["p_under"] = 1.0 - out["p_over"]
    out["actual_under"] = 1.0 - out["actual_over"]
    if season_col:
        out["season"] = pd.to_numeric(out[season_col], errors="coerce")
    else:
        out["season"] = pd.to_datetime(out[date_col], errors="coerce").dt.year
    if date_col:
        out["game_date"] = pd.to_datetime(out[date_col], errors="coerce")
    else:
        out["game_date"] = pd.NaT
    if game_col:
        out["game_id"] = out[game_col].astype(str)
    else:
        out["game_id"] = np.arange(len(out)).astype(str)

    out = out[
        out["p_over"].between(0, 1, inclusive="both")
        & out["actual_over"].isin([0.0, 1.0])
        & out["season"].notna()
    ].copy()
    out["season"] = out["season"].astype(int)
    return out


def grade_subset(g: pd.DataFrame, side: str, threshold: float):
    pcol = f"p_{side}"
    ycol = f"actual_{side}"
    q = g[g[pcol] >= threshold].copy()
    n = len(q)
    wins = int(q[ycol].sum()) if n else 0
    losses = n - wins
    lo, hi = wilson_interval(wins, n)
    hit = wins / n if n else np.nan
    mean_p = float(q[pcol].mean()) if n else np.nan
    return {
        "side": side.upper(),
        "threshold": threshold,
        "n": n,
        "wins": wins,
        "losses": losses,
        "hit_rate": hit,
        "mean_model_probability": mean_p,
        "calibration_gap_pp": 100 * (hit - mean_p) if n else np.nan,
        "wilson95_low": lo,
        "wilson95_high": hi,
    }


def threshold_grid(start: float, stop: float, step: float):
    n = int(round((stop - start) / step))
    vals = [round(start + i * step, 10) for i in range(n + 1)]
    return [v for v in vals if 0 < v < 1]


def sweep(df: pd.DataFrame, side: str, thresholds):
    return pd.DataFrame([grade_subset(df, side, t) for t in thresholds])


def year_by_year(df: pd.DataFrame, side: str, thresholds):
    rows = []
    for season, g in df.groupby("season"):
        for t in thresholds:
            r = grade_subset(g, side, t)
            r["season"] = int(season)
            rows.append(r)
    return pd.DataFrame(rows)


def stable_candidate_table(over_sweep: pd.DataFrame, under_sweep: pd.DataFrame, min_n: int):
    """Rank thresholds by conservative observed reliability, not price or ROI.

    Primary sort is Wilson lower bound, with hit rate and sample size as tie-breakers.
    This is a diagnostic ranking only; the user can choose a looser gate if volume is
    more important than conservative reliability.
    """
    frames = []
    for s in [under_sweep, over_sweep]:
        q = s[s["n"] >= min_n].copy()
        if q.empty:
            continue
        q = q.sort_values(["wilson95_low", "hit_rate", "n"], ascending=[False, False, False])
        frames.append(q.head(10))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/CSV.GZ containing frozen OOS I2 predictions")
    ap.add_argument("--output-dir", default="data/derived/i2/threshold_calibration")
    ap.add_argument("--start", type=float, default=0.45)
    ap.add_argument("--stop", type=float, default=0.75)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--min-n", type=int, default=100)
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare(pd.read_csv(src))
    thresholds = threshold_grid(args.start, args.stop, args.step)

    under = sweep(df, "under", thresholds)
    over = sweep(df, "over", thresholds)
    under_year = year_by_year(df, "under", thresholds)
    over_year = year_by_year(df, "over", thresholds)
    candidates = stable_candidate_table(over, under, args.min_n)

    under.to_csv(out_dir / "under_threshold_sweep.csv", index=False)
    over.to_csv(out_dir / "over_threshold_sweep.csv", index=False)
    under_year.to_csv(out_dir / "under_threshold_by_season.csv", index=False)
    over_year.to_csv(out_dir / "over_threshold_by_season.csv", index=False)
    candidates.to_csv(out_dir / "candidate_thresholds.csv", index=False)
    df[["game_id", "game_date", "season", "p_over", "p_under", "actual_over", "actual_under"]].to_csv(
        out_dir / "graded_predictions.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "input": str(src),
        "n_predictions": int(len(df)),
        "seasons": sorted(int(s) for s in df["season"].unique()),
        "threshold_grid": {"start": args.start, "stop": args.stop, "step": args.step},
        "minimum_candidate_n": args.min_n,
        "price_used": False,
        "market_fields_used": [],
        "qualification_definition": "UNDER if P_under >= threshold; OVER if P_over >= threshold; price considered only after qualification and is absent here.",
        "selection_note": "candidate_thresholds ranks conservative observed reliability by Wilson lower bound; final gate should also inspect monotonicity and year-by-year persistence.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

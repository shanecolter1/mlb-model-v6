#!/usr/bin/env python3
"""Research-only I2 date split on every primary-home game, independent of contact type.

Input ZIPs are the season bundles from Phase 19C. The May 25 boundary is an
exploratory external carry-change date, not a fitted change point or a model
feature. All predictions are the archived market-isolated pregame replay.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def load_season(path: Path, year: int) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        replay = json.loads(archive.read(f"replay_{year}_precision_10000.json"))
        venue = json.loads(archive.read(f"home_venue_{year}.json"))
        inputs = json.loads(archive.read(f"replay_{year}_inputs.json"))
    if not (replay["season"] == venue["season"] == inputs["season"] == year):
        raise ValueError("Season mismatch")
    if any(x["market_inputs_used"] is not False for x in (replay, venue, inputs)):
        raise ValueError("Market-contaminated input")
    if replay["trials_per_game"] != 10000:
        raise ValueError("Replay precision mismatch")
    observations = {str(g["gid"]): g["observed"] for g in inputs["games"]}
    rows = []
    for pred in replay["predictions"]:
        gid = str(pred["gid"])
        if venue["games"].get(gid, {}).get("status") != "PRIMARY_HOME_VENUE":
            continue
        observed = observations[gid]
        rows.append({
            "season": year,
            "date": pd.Timestamp(str(pred["date"])).strftime("%Y-%m-%d"),
            "gid": gid,
            "under_y": int(pred["observed_under05"]),
            "under_p": float(pred["raw_under05"]),
            "top_y": int(observed["top2_runs"] > 0),
            "top_p": float(pred["top2_score_probability"]),
            "bottom_y": int(observed["bottom2_runs"] > 0),
            "bottom_p": float(pred["bottom2_score_probability"]),
        })
    data = pd.DataFrame(rows)
    if len(data) != data.gid.nunique() or data.empty:
        raise ValueError("Missing or duplicate games")
    if not np.array_equal(data.under_y.to_numpy(),
                          ((data.top_y == 0) & (data.bottom_y == 0)).astype(int)):
        raise ValueError("Half/full target mismatch")
    return data


def metrics(data: pd.DataFrame) -> dict:
    out = {"games": len(data), "dates": data.date.nunique()}
    for name in ("under", "top", "bottom"):
        y, p = data[f"{name}_y"].to_numpy(), data[f"{name}_p"].to_numpy()
        p = np.clip(p, 1e-12, 1 - 1e-12)
        out[name] = {
            "actual": float(y.mean()), "predicted": float(p.mean()),
            "actual_minus_predicted": float(np.mean(y - p)),
            "brier": float(np.mean((y - p) ** 2)),
            "logloss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
        }
    return out


def date_bootstrap(data: pd.DataFrame, cutoff: str, draws: int = 10000) -> dict:
    """Day-cluster CI for post-minus-pre under residual, resampling each side."""
    daily = data.assign(residual=data.under_y - data.under_p).groupby(
        ["date", "after"], as_index=False
    ).agg(total=("residual", "sum"), games=("residual", "size"))
    clusters = [daily.loc[daily.after == flag, ["total", "games"]].to_numpy()
                for flag in (False, True)]
    if any(len(group) == 0 for group in clusters):
        raise ValueError(f"No games on one side of {cutoff}")
    rng = np.random.default_rng(192026)
    effects = np.empty(draws)
    for i in range(draws):
        means = []
        for group in clusters:
            sample = group[rng.integers(len(group), size=len(group))].sum(axis=0)
            means.append(sample[0] / sample[1])
        effects[i] = means[1] - means[0]
    return {"post_minus_pre": float((data.loc[data.after, "under_y"] -
                                       data.loc[data.after, "under_p"]).mean() -
                                      (data.loc[~data.after, "under_y"] -
                                       data.loc[~data.after, "under_p"]).mean()),
            "ci95_day_cluster": [float(x) for x in np.quantile(effects, [.025, .975])]}


def consecutive_windows(data: pd.DataFrame, days: int = 28) -> list[dict]:
    """Fixed, non-overlapping game-date windows, including the last short window."""
    start = pd.Timestamp(data.date.min())
    dates = pd.to_datetime(data.date)
    windows = []
    for index, group in data.groupby(((dates - start).dt.days // days)):
        window_start = start + pd.Timedelta(days=int(index) * days)
        window_end = min(window_start + pd.Timedelta(days=days - 1),
                         pd.Timestamp(data.date.max()))
        windows.append({"start": window_start.strftime("%Y-%m-%d"),
                        "end": window_end.strftime("%Y-%m-%d"),
                        **metrics(group)})
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", nargs=2, action="append", metavar=("YEAR", "ZIP"),
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-mm-dd", default="05-25")
    args = parser.parse_args()
    years = {}
    for year_str, path in args.bundle:
        year = int(year_str)
        data = load_season(Path(path), year)
        cutoff = f"{year}-{args.cutoff_mm_dd}"
        data["after"] = data.date >= cutoff
        years[str(year)] = {
            "before": metrics(data.loc[~data.after]),
            "after": metrics(data.loc[data.after]),
            "shift": date_bootstrap(data, cutoff),
            "fixed_28_day_windows": consecutive_windows(data),
        }
    output = {"version": "i2-date-regime-v1", "market_inputs_used": False,
              "sample": "all primary-home games, no contact-type filter",
              "cutoff_mm_dd": args.cutoff_mm_dd,
              "interpretation": "exploratory association; date does not establish ball cause",
              "years": years}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

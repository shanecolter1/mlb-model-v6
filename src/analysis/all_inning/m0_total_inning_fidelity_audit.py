#!/usr/bin/env python3
"""Audit M0 fidelity for the all-inning engine on 2021-2024 development data only.

Purpose:
- verify canonical half-inning outcomes reconstruct the master full-inning outcomes;
- build the unsmoothed empirical opening-total x inning baseline used by M0;
- prohibit 2025 holdout access;
- retain only the isolated opening full-game total as market context.

This script performs no model tuning and introduces no shrinkage/pooling assumptions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path: Path, columns=None):
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-matrix", type=Path, required=True)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    h = read(a.half_matrix).copy()
    h["game_date"] = pd.to_datetime(h["game_date"], errors="coerce").dt.normalize()
    if (pd.to_numeric(h["season"], errors="coerce") >= 2025).any():
        raise RuntimeError("2025 holdout leaked into half matrix")
    if not set(pd.to_numeric(h["inning"], errors="coerce").dropna().astype(int).unique()).issubset(set(range(1, 10))):
        raise RuntimeError("half matrix contains inning outside 1..9")
    if h.duplicated(["game_id", "inning", "half"]).any():
        raise RuntimeError("half matrix key is non-unique")

    # Full-inning reconstruction is valid only when both halves were played.
    p = h[h["half_played"]].pivot_table(
        index=["game_id", "game_date", "season", "game_number", "away_team_code", "home_team_code", "dk_total_open_total", "inning"],
        columns="half", values="runs_half", aggfunc="first"
    ).reset_index()
    for c in ["top", "bottom"]:
        if c not in p.columns:
            p[c] = np.nan
    p["both_halves_played"] = p["top"].notna() & p["bottom"].notna()
    p["full_inning_runs_reconstructed"] = np.where(
        p["both_halves_played"], p["top"] + p["bottom"], np.nan
    )
    p["full_inning_scored_reconstructed"] = np.where(
        p["both_halves_played"], (p["full_inning_runs_reconstructed"] >= 1).astype(int), np.nan
    )

    master_cols = ["game_date", "away_team_code", "home_team_code", "game_number"] + [
        f"inning{i}_total_runs" for i in range(1, 10)
    ]
    m = read(a.master, columns=master_cols).copy()
    m["game_date"] = pd.to_datetime(m["game_date"], errors="coerce").dt.normalize()
    m = m[m["game_date"].dt.year <= 2024].copy()
    if (m["game_date"].dt.year >= 2025).any():
        raise RuntimeError("2025 holdout leaked from master")
    m["game_number"] = pd.to_numeric(m["game_number"], errors="coerce").astype("Int64")

    long = []
    for i in range(1, 10):
        q = m[["game_date", "away_team_code", "home_team_code", "game_number", f"inning{i}_total_runs"]].copy()
        q["inning"] = i
        q = q.rename(columns={f"inning{i}_total_runs": "master_full_inning_runs"})
        q["master_full_inning_runs"] = pd.to_numeric(q["master_full_inning_runs"], errors="coerce")
        long.append(q)
    ml = pd.concat(long, ignore_index=True)

    keys = ["game_date", "away_team_code", "home_team_code", "game_number", "inning"]
    c = p.merge(ml, on=keys, how="left", validate="many_to_one")
    comparable = c[c["both_halves_played"] & c["master_full_inning_runs"].notna()].copy()
    comparable["diff"] = comparable["full_inning_runs_reconstructed"] - comparable["master_full_inning_runs"]
    mismatch = comparable[comparable["diff"].abs() > 1e-12].copy()

    # Locked M0 implementation object: raw empirical distribution by exact opening total x inning.
    played = p[p["both_halves_played"]].copy()
    played["total"] = pd.to_numeric(played["dk_total_open_total"], errors="coerce")
    played = played[played["total"].notna()].copy()
    played["run_bucket"] = played["full_inning_runs_reconstructed"].clip(lower=0).astype(int).astype(str)
    played.loc[played["full_inning_runs_reconstructed"] >= 4, "run_bucket"] = "4+"

    rows = []
    for (total, inning), g in played.groupby(["total", "inning"], sort=True):
        n = len(g)
        counts = {k: int((g["run_bucket"] == k).sum()) for k in ["0", "1", "2", "3", "4+"]}
        row = {
            "opening_total": float(total),
            "inning": int(inning),
            "n": int(n),
            "p0": counts["0"] / n,
            "p1": counts["1"] / n,
            "p2": counts["2"] / n,
            "p3": counts["3"] / n,
            "p4plus": counts["4+"] / n,
            "p1plus": 1 - counts["0"] / n,
            "mean_runs": float(g["full_inning_runs_reconstructed"].mean()),
        }
        row["p2plus"] = row["p2"] + row["p3"] + row["p4plus"]
        row["p3plus"] = row["p3"] + row["p4plus"]
        rows.append(row)
    baseline = pd.DataFrame(rows)
    if not baseline.empty:
        baseline["probability_sum"] = baseline[["p0", "p1", "p2", "p3", "p4plus"]].sum(axis=1)

    baseline.to_csv(a.output_dir / "m0_total_x_inning_empirical_2021_2024.csv", index=False)
    comparable.to_parquet(a.output_dir / "m0_full_inning_fidelity_rows.parquet", index=False)

    summary_by_inning = []
    for i in range(1, 10):
        z = comparable[comparable["inning"] == i]
        summary_by_inning.append({
            "inning": i,
            "n_compared": int(len(z)),
            "mismatches": int((z["diff"].abs() > 1e-12).sum()),
            "max_abs_diff": float(z["diff"].abs().max()) if len(z) else None,
        })

    manifest = {
        "status": "PASS" if len(mismatch) == 0 else "FAIL",
        "architecture": "M0_raw_empirical_opening_total_x_inning",
        "development_seasons": [2021, 2022, 2023, 2024],
        "holdout_season": 2025,
        "holdout_opened": False,
        "smoothing_used": False,
        "shrinkage_used": False,
        "pooling_used": False,
        "market_data_retained": ["dk_total_open_total"],
        "derivative_market_data_used": False,
        "fidelity_rows_compared": int(len(comparable)),
        "fidelity_mismatches": int(len(mismatch)),
        "summary_by_inning": summary_by_inning,
        "baseline_cells": int(len(baseline)),
        "baseline_probability_sum_max_error": float((baseline["probability_sum"] - 1).abs().max()) if len(baseline) else None,
        "note": "M0 table is an implementation-fidelity development object only; no new smoothing/tuning was introduced.",
    }
    (a.output_dir / "m0_fidelity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if len(mismatch):
        print(mismatch[keys + ["full_inning_runs_reconstructed", "master_full_inning_runs", "diff"]].head(50).to_string(index=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

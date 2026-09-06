#!/usr/bin/env python3
"""Dual-view I2 probability-threshold analysis.

Price-blind. This script does not choose a sportsbook wager or calculate EV.
It reports two complementary views of the same chronological OOS predictions:

1. Pooled descriptive view: 2022-2025 combined, maximizing sample size for
   estimating realized performance at each probability threshold.
2. Robustness view: 2022-2024 development plus untouched 2025 validation,
   preserving the original sealed-period check for threshold selection risk.

Both Under and Over are evaluated independently. No sportsbook derivative price,
juice, moneyline, or run line is used.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEV_SEASONS = [2022, 2023, 2024]
VAL_SEASON = 2025
POOLED_SEASONS = [2022, 2023, 2024, 2025]


def wilson(w, n, z=1.959963984540054):
    if n <= 0:
        return float("nan"), float("nan")
    p = w / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - rad, ctr + rad


def american_odds_from_probability(p):
    """Fair American odds corresponding to a probability; no market price used."""
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return float("nan")
    if abs(p - 0.5) < 1e-12:
        return 100.0
    if p > 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def normalize_prediction_columns(x):
    x = x.copy()
    if "p_over" not in x.columns:
        for c in ["prediction", "context_prediction", "pred_over", "prob_over", "p_i2_over"]:
            if c in x.columns:
                x["p_over"] = pd.to_numeric(x[c], errors="coerce")
                break
    if "p_over" not in x.columns:
        raise SystemExit("Could not identify P(Over) column")
    x["p_over"] = pd.to_numeric(x["p_over"], errors="coerce")
    if "p_under" not in x.columns:
        x["p_under"] = 1.0 - x["p_over"]
    else:
        x["p_under"] = pd.to_numeric(x["p_under"], errors="coerce")
    if "actual_over" not in x.columns:
        raise SystemExit("Missing actual_over outcome column")
    x["actual_over"] = pd.to_numeric(x["actual_over"], errors="coerce")
    if "actual_under" not in x.columns:
        x["actual_under"] = 1.0 - x["actual_over"]
    else:
        x["actual_under"] = pd.to_numeric(x["actual_under"], errors="coerce")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    return x


def score(df, side, t):
    pcol = "p_under" if side == "UNDER" else "p_over"
    ycol = "actual_under" if side == "UNDER" else "actual_over"
    q = df[pd.to_numeric(df[pcol], errors="coerce") >= t].copy()
    n = len(q)
    w = int(pd.to_numeric(q[ycol], errors="coerce").fillna(0).sum())
    hit = w / n if n else float("nan")
    meanp = float(pd.to_numeric(q[pcol], errors="coerce").mean()) if n else float("nan")
    lo, hi = wilson(w, n)
    return dict(
        n=n,
        wins=w,
        losses=n - w,
        hit_rate=hit,
        fair_american_odds_from_hit_rate=american_odds_from_probability(hit),
        mean_model_probability=meanp,
        calibration_gap_pp=(hit - meanp) * 100 if n else float("nan"),
        wilson95_low=lo,
        wilson95_high=hi,
    )


def season_scores(df, seasons, side, threshold):
    per = []
    for season in seasons:
        per.append(score(df[df.season.eq(season)], side, threshold))
    valid_hits = [s["hit_rate"] for s in per if np.isfinite(s["hit_rate"])]
    sd = float(np.std(valid_hits, ddof=1)) if len(valid_hits) >= 2 else float("nan")
    return per, sd


def add_incremental_columns(tab):
    tab = tab.sort_values(["side", "threshold"]).reset_index(drop=True)
    for prefix in ["pooled", "dev", "val2025"]:
        tab[f"{prefix}_hit_rate_delta_pp_vs_prev_threshold"] = (
            tab.groupby("side")[f"{prefix}_hit_rate"].diff() * 100
        )
        tab[f"{prefix}_n_delta_vs_prev_threshold"] = tab.groupby("side")[f"{prefix}_n"].diff()
        tab[f"{prefix}_n_lost_vs_prev_threshold"] = -tab[f"{prefix}_n_delta_vs_prev_threshold"]
    return tab


def run_one(path, outdir, start, stop, step):
    x = normalize_prediction_columns(pd.read_csv(path))
    req = {"season", "p_over", "p_under", "actual_over", "actual_under"}
    miss = req - set(x.columns)
    if miss:
        raise SystemExit(f"Missing columns: {sorted(miss)}")

    pooled = x[x.season.isin(POOLED_SEASONS)].copy()
    dev = x[x.season.isin(DEV_SEASONS)].copy()
    val = x[x.season.eq(VAL_SEASON)].copy()

    rows = []
    dev_season_rows = []
    pooled_season_rows = []
    grid = np.arange(start, stop + step / 2, step)

    for side in ["UNDER", "OVER"]:
        for raw_t in grid:
            t = round(float(raw_t), 6)
            p = score(pooled, side, t)
            d = score(dev, side, t)
            v = score(val, side, t)

            dev_per, dev_sd = season_scores(dev, DEV_SEASONS, side, t)
            pooled_per, pooled_sd = season_scores(pooled, POOLED_SEASONS, side, t)

            for season, s in zip(DEV_SEASONS, dev_per):
                dev_season_rows.append({"side": side, "threshold": t, "season": season, **s})
            for season, s in zip(POOLED_SEASONS, pooled_per):
                pooled_season_rows.append({"side": side, "threshold": t, "season": season, **s})

            row = {
                "side": side,
                "threshold": t,
                **{f"pooled_{k}": vv for k, vv in p.items()},
                "pooled_season_hit_rate_sd": pooled_sd,
                **{f"dev_{k}": vv for k, vv in d.items()},
                "dev_season_hit_rate_sd": dev_sd,
                "dev_min_season_n": min(s["n"] for s in dev_per),
                **{f"val2025_{k}": vv for k, vv in v.items()},
            }
            for season, s in zip(POOLED_SEASONS, pooled_per):
                row[f"pooled_{season}_n"] = s["n"]
                row[f"pooled_{season}_hit_rate"] = s["hit_rate"]
            for season, s in zip(DEV_SEASONS, dev_per):
                row[f"dev_{season}_hit_rate"] = s["hit_rate"]
            rows.append(row)

    outdir.mkdir(parents=True, exist_ok=True)
    tab = add_incremental_columns(pd.DataFrame(rows))

    # Backward-compatible filename plus explicit dual-view filename.
    tab.to_csv(outdir / "threshold_holdout_table.csv", index=False)
    tab.to_csv(outdir / "threshold_dual_view_table.csv", index=False)
    pd.DataFrame(dev_season_rows).to_csv(outdir / "development_by_season.csv", index=False)
    pd.DataFrame(pooled_season_rows).to_csv(outdir / "pooled_by_season.csv", index=False)

    # Focus table requested for investigating a possible Under performance step-up.
    focus = tab[
        (tab.side.eq("UNDER"))
        & (tab.threshold >= 0.57 - 1e-12)
        & (tab.threshold <= 0.64 + 1e-12)
    ].copy()
    focus_cols = [
        "threshold",
        "pooled_n", "pooled_wins", "pooled_losses", "pooled_hit_rate",
        "pooled_fair_american_odds_from_hit_rate",
        "pooled_wilson95_low", "pooled_wilson95_high",
        "pooled_season_hit_rate_sd",
        "pooled_2022_hit_rate", "pooled_2023_hit_rate", "pooled_2024_hit_rate", "pooled_2025_hit_rate",
        "pooled_hit_rate_delta_pp_vs_prev_threshold", "pooled_n_lost_vs_prev_threshold",
        "dev_n", "dev_hit_rate", "dev_wilson95_low", "dev_wilson95_high",
        "val2025_n", "val2025_hit_rate", "val2025_wilson95_low", "val2025_wilson95_high",
    ]
    focus[focus_cols].to_csv(outdir / "under_focus_57_64.csv", index=False)

    # Original development ranking remains a robustness diagnostic only.
    eligible = tab[(tab.dev_n >= 100) & (tab.dev_min_season_n >= 25)].copy()
    eligible["rank_wilson"] = eligible.groupby("side")["dev_wilson95_low"].rank(
        method="min", ascending=False
    )
    eligible.sort_values(["side", "rank_wilson", "threshold"]).to_csv(
        outdir / "ranked_development_candidates.csv", index=False
    )

    # Pooled ranking is descriptive only; it must not be represented as sealed selection validation.
    pooled_eligible = tab[tab.pooled_n >= 100].copy()
    pooled_eligible["rank_pooled_wilson"] = pooled_eligible.groupby("side")[
        "pooled_wilson95_low"
    ].rank(method="min", ascending=False)
    pooled_eligible.sort_values(["side", "rank_pooled_wilson", "threshold"]).to_csv(
        outdir / "ranked_pooled_descriptive.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "price_used": False,
        "market_fields_used": [],
        "pooled_descriptive_seasons": POOLED_SEASONS,
        "development_seasons": DEV_SEASONS,
        "sealed_validation_season": VAL_SEASON,
        "pooled_predictions": int(len(pooled)),
        "development_predictions": int(len(dev)),
        "validation_predictions": int(len(val)),
        "threshold_grid": {"start": start, "stop": stop, "step": step},
        "under_focus_range": {"start": 0.57, "stop": 0.64},
        "notes": [
            "Pooled 2022-2025 view is descriptive and maximizes sample size; it is not a sealed threshold-selection test.",
            "2022-2024 development plus untouched 2025 validation is retained as the robustness view.",
            "Observed fair American odds are mathematical transforms of realized hit rate, not sportsbook prices.",
            "No sportsbook derivative price, juice, moneyline, or run line is used.",
        ],
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-input", required=True)
    ap.add_argument("--baseline-input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", type=float, default=0.40)
    ap.add_argument("--stop", type=float, default=0.75)
    ap.add_argument("--step", type=float, default=0.0025)
    a = ap.parse_args()
    root = Path(a.output_dir)
    run_one(a.strict_input, root / "strict_context", a.start, a.stop, a.step)
    run_one(a.baseline_input, root / "total_only", a.start, a.stop, a.step)
    print(json.dumps({"status": "PASS", "output_dir": str(root), "price_used": False}, indent=2))


if __name__ == "__main__":
    main()

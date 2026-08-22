#!/usr/bin/env python3
"""Build empirical betting-target probabilities and fair American odds by opening game total.

Required input: one row per MLB game containing a pregame opening full-game total and
inning-level runs for both teams (or full-inning total runs). Designed for the joined
2021-2025 historical master used by the I2 research workflow.

Outputs:
  * inning_any_run_by_total.csv
  * half_inning_runs_by_total.csv
  * f5_game_total_by_total.csv
  * f5_team_total_by_total.csv
  * scoreless_innings_distribution_by_total.csv
  * manifest.json

No independence/Poisson assumptions are used. Every probability is an empirical count.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pandas as pd


def fair_american(p: float):
    if pd.isna(p) or p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p > 0.5 else round(100 * (1 - p) / p)


def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def detect_total_col(df):
    c = pick_col(df, [
        "opening_total", "dk_opening_total", "draftkings_opening_total",
        "pregame_opening_total", "game_total_open", "total_open", "open_total"
    ])
    if not c:
        raise ValueError("Could not find opening total column")
    return c


def detect_inning_cols(df):
    away, home, full = {}, {}, {}
    for inn in range(1, 10):
        away[inn] = pick_col(df, [f"away_i{inn}", f"away_inning_{inn}", f"away_inning{inn}_runs", f"away_runs_i{inn}"])
        home[inn] = pick_col(df, [f"home_i{inn}", f"home_inning_{inn}", f"home_inning{inn}_runs", f"home_runs_i{inn}"])
        full[inn] = pick_col(df, [f"i{inn}_runs", f"inning_{inn}_runs", f"inning{inn}_runs", f"runs_inning_{inn}"])
    if all(away.values()) and all(home.values()):
        return "halves", away, home, full
    if all(full.values()):
        return "full", away, home, full
    missing = [i for i in range(1, 10) if not full[i] and not (away[i] and home[i])]
    raise ValueError(f"Missing inning run columns for innings: {missing}")


def add_prob_rows(rows, total, market, threshold, values, predicate):
    s = pd.Series(values).dropna()
    n = len(s)
    if not n:
        return
    wins = int(predicate(s).sum())
    p = wins / n
    rows.append({
        "opening_total": total, "market": market, "threshold": threshold,
        "n": n, "wins": wins, "probability": p, "fair_american": fair_american(p)
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="data/derived/i2/total_bucket_targets")
    ap.add_argument("--season-min", type=int, default=2021)
    ap.add_argument("--season-max", type=int, default=2025)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    season_col = pick_col(df, ["season", "year"])
    if season_col:
        df = df[(df[season_col] >= args.season_min) & (df[season_col] <= args.season_max)].copy()

    total_col = detect_total_col(df)
    mode, away_cols, home_cols, full_cols = detect_inning_cols(df)
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df = df[df[total_col].notna()].copy()

    # Normalize inning run columns.
    for inn in range(1, 10):
        if mode == "halves":
            df[f"away_i{inn}"] = pd.to_numeric(df[away_cols[inn]], errors="coerce")
            df[f"home_i{inn}"] = pd.to_numeric(df[home_cols[inn]], errors="coerce")
            df[f"i{inn}"] = df[f"away_i{inn}"] + df[f"home_i{inn}"]
        else:
            df[f"i{inn}"] = pd.to_numeric(df[full_cols[inn]], errors="coerce")

    totals = sorted(df[total_col].dropna().unique())

    # 1) Full-inning any-run probabilities, every inning.
    inning_rows = []
    for total in totals:
        g = df[df[total_col] == total]
        for inn in range(1, 10):
            s = g[f"i{inn}"].dropna()
            n = len(s)
            if not n:
                continue
            yes = int((s >= 1).sum())
            p = yes / n
            inning_rows.append({
                "opening_total": total, "inning": inn, "n": n,
                "p_yes_run": p, "p_no_run": 1-p,
                "fair_yes_run": fair_american(p), "fair_no_run": fair_american(1-p)
            })
    pd.DataFrame(inning_rows).to_csv(out / "inning_any_run_by_total.csv", index=False)

    # 2) Half-inning exact/cumulative distributions (requires separate away/home inning runs).
    half_rows = []
    if mode == "halves":
        for total in totals:
            g = df[df[total_col] == total]
            for inn in range(1, 10):
                vals = pd.concat([g[f"away_i{inn}"], g[f"home_i{inn}"]], ignore_index=True).dropna()
                n = len(vals)
                if not n:
                    continue
                for label, pred in [
                    ("0", lambda x: x == 0), ("1", lambda x: x == 1),
                    ("2", lambda x: x == 2), ("3", lambda x: x == 3),
                    ("4+", lambda x: x >= 4), ("1+", lambda x: x >= 1),
                    ("2+", lambda x: x >= 2), ("3+", lambda x: x >= 3)
                ]:
                    wins = int(pred(vals).sum()); p = wins / n
                    half_rows.append({
                        "opening_total": total, "inning": inn, "runs_target": label,
                        "n_half_innings": n, "hits": wins, "probability": p,
                        "fair_american": fair_american(p)
                    })
        pd.DataFrame(half_rows).to_csv(out / "half_inning_runs_by_total.csv", index=False)

    # 3) First-five game total distributions and O/U thresholds.
    f5_rows = []
    for total in totals:
        g = df[df[total_col] == total].copy()
        g["f5"] = g[[f"i{i}" for i in range(1,6)]].sum(axis=1, min_count=5)
        s = g["f5"].dropna(); n = len(s)
        if not n: continue
        max_run = int(s.max())
        for r in range(0, max_run + 1):
            hits = int((s == r).sum()); p = hits/n
            f5_rows.append({"opening_total": total,"market":"F5 exact runs","threshold":r,"side":"exact","n":n,"hits":hits,"probability":p,"fair_american":fair_american(p)})
        for x2 in range(1, 22, 2):
            line = x2/2
            for side, pred in [("over", s > line), ("under", s < line)]:
                hits = int(pred.sum()); p = hits/n
                f5_rows.append({"opening_total":total,"market":"F5 total","threshold":line,"side":side,"n":n,"hits":hits,"probability":p,"fair_american":fair_american(p)})
    pd.DataFrame(f5_rows).to_csv(out / "f5_game_total_by_total.csv", index=False)

    # 4) First-five team-total cumulative targets (requires separate half-inning scoring).
    team_rows = []
    if mode == "halves":
        for total in totals:
            g = df[df[total_col] == total].copy()
            away_f5 = g[[f"away_i{i}" for i in range(1,6)]].sum(axis=1, min_count=5)
            home_f5 = g[[f"home_i{i}" for i in range(1,6)]].sum(axis=1, min_count=5)
            vals = pd.concat([away_f5, home_f5], ignore_index=True).dropna(); n=len(vals)
            if not n: continue
            for k in range(1,8):
                for side, pred in [("over", vals >= k), ("under", vals < k)]:
                    hits=int(pred.sum()); p=hits/n
                    team_rows.append({"opening_total":total,"market":"team F5 runs","threshold":f"{k}+" if side=="over" else f"under {k}","side":side,"n_team_games":n,"hits":hits,"probability":p,"fair_american":fair_american(p)})
        pd.DataFrame(team_rows).to_csv(out / "f5_team_total_by_total.csv", index=False)

    # 5) Empirical scoreless-innings-per-game distribution (no independence assumption).
    dist_rows = []
    for total in totals:
        g=df[df[total_col]==total].copy()
        reached = g[[f"i{i}" for i in range(1,10)]].notna().sum(axis=1)
        scoreless = (g[[f"i{i}" for i in range(1,10)]] == 0).sum(axis=1)
        valid = reached >= 8
        s=scoreless[valid]; n=len(s)
        if not n: continue
        for k in sorted(s.unique()):
            hits=int((s==k).sum()); p=hits/n
            dist_rows.append({"opening_total":total,"scoreless_innings":int(k),"n_games":n,"games":hits,"probability":p,"fair_american":fair_american(p)})
    pd.DataFrame(dist_rows).to_csv(out / "scoreless_innings_distribution_by_total.csv", index=False)

    manifest = {
        "source": str(src),
        "season_window": [args.season_min,args.season_max],
        "rows_after_filter": int(len(df)),
        "opening_total_column": total_col,
        "inning_input_mode": mode,
        "opening_total_buckets": [float(x) for x in totals],
        "method": "empirical row-level counts; no Poisson or inning-independence assumptions",
        "outputs": sorted(p.name for p in out.glob("*.csv"))
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

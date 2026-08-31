#!/usr/bin/env python3
"""Build canonical 2021-2024 all-inning half-inning research matrix.

This is a DEVELOPMENT-ONLY target builder for the unified I1-I9 engine.
It intentionally excludes 2025 from all target extraction and outputs.

Inputs
------
--master
    Canonical historical joined master containing opening full-game total and
    inning1_total_runs..inning9_total_runs. Used for market-isolated M0 context
    and game-level validation only.
--reusable-root
    Root containing normalized reusable as-of artifact files. inning_outcomes.parquet
    is the preferred source for half-inning outcomes and game_index.parquet supplies
    game/team identity.

Output
------
One row per game x inning x half for innings 1..9, with explicit half_played.
No realized pitcher/batter identity is joined here; this file is the neutral target
spine onto which leakage-safe state distributions are attached later.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

DEV_MAX_SEASON = 2024
INNINGS = range(1, 10)


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"{name} expected exactly once under {root}; found {hits}")
    return hits[0]


def normalize_team_code(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.strip()


def master_dev(master_path: Path) -> pd.DataFrame:
    m = read(master_path).copy()
    if "game_date" not in m.columns:
        raise RuntimeError("master missing game_date")
    m["game_date"] = pd.to_datetime(m["game_date"], errors="coerce").dt.normalize()
    m["season"] = m["game_date"].dt.year
    # Critical holdout guard: target columns are not inspected until after 2025 is removed.
    m = m[m["season"].between(2021, DEV_MAX_SEASON)].copy()
    need = ["game_date", "away_team_code", "home_team_code", "dk_total_open_total"]
    miss = [c for c in need if c not in m.columns]
    if miss:
        raise RuntimeError(f"master missing required development columns {miss}")
    m["away_team_code"] = normalize_team_code(m["away_team_code"])
    m["home_team_code"] = normalize_team_code(m["home_team_code"])
    return m


def attach_game_number(df: pd.DataFrame) -> pd.DataFrame:
    """Retrosheet-compatible singleton/doubleheader disambiguation with no outcomes."""
    x = df.copy()
    counts = x.groupby(["game_date", "away_team_code", "home_team_code"])["game_date"].transform("size")
    x["game_number"] = 0
    dup = counts > 1
    if dup.any():
        sort_cols = ["game_date", "away_team_code", "home_team_code"]
        if "game_datetime" in x.columns:
            sort_cols.append("game_datetime")
        ordered = x.loc[dup].sort_values(sort_cols, kind="stable").copy()
        ordered["game_number"] = ordered.groupby(["game_date", "away_team_code", "home_team_code"]).cumcount() + 1
        x.loc[ordered.index, "game_number"] = ordered["game_number"].astype(int)
    return x


def infer_team_codes(game_index: pd.DataFrame) -> pd.DataFrame:
    x = game_index.copy()
    rename = {}
    for cand in ("away_team_code", "away_code", "away_abbr"):
        if cand in x.columns:
            rename[cand] = "away_team_code"; break
    for cand in ("home_team_code", "home_code", "home_abbr"):
        if cand in x.columns:
            rename[cand] = "home_team_code"; break
    x = x.rename(columns=rename)
    if "away_team_code" not in x.columns or "home_team_code" not in x.columns:
        raise RuntimeError("game_index requires away/home team codes for canonical master join")
    x["away_team_code"] = normalize_team_code(x["away_team_code"])
    x["home_team_code"] = normalize_team_code(x["home_team_code"])
    return x


def build_half_spine(master: pd.DataFrame, game_index: pd.DataFrame, inning_outcomes: pd.DataFrame) -> pd.DataFrame:
    g = game_index.copy()
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    g["season"] = g["game_date"].dt.year
    g = g[g["season"].between(2021, DEV_MAX_SEASON)].copy()
    g = infer_team_codes(g)
    g = attach_game_number(g)

    m = attach_game_number(master)
    join = ["game_date", "away_team_code", "home_team_code", "game_number"]
    keep_master = join + ["dk_total_open_total"]
    gm = g.merge(m[keep_master], on=join, how="inner", validate="one_to_one")

    io = inning_outcomes.copy()
    io["game_date"] = pd.to_datetime(io["game_date"], errors="coerce").dt.normalize()
    io["inning"] = pd.to_numeric(io["inning"], errors="coerce").astype("Int64")
    io = io[io["inning"].between(1, 9)].copy()
    io = io[io["game_id"].isin(gm["game_id"])].copy()

    required = ["game_id", "inning", "away_runs", "home_runs", "away_half_played", "home_half_played"]
    miss = [c for c in required if c not in io.columns]
    if miss:
        raise RuntimeError(f"inning_outcomes missing {miss}")

    cols = ["game_id", "game_date", "season", "away_team_id", "home_team_id", "away_team_code", "home_team_code", "game_number", "dk_total_open_total"]
    if "game_datetime" in gm.columns:
        cols.append("game_datetime")
    base = gm[cols].drop_duplicates("game_id")
    z = io.merge(base, on="game_id", how="inner", validate="many_to_one", suffixes=("", "_game"))

    rows = []
    for r in z.itertuples(index=False):
        inning = int(r.inning)
        common = {
            "game_id": r.game_id,
            "game_date": getattr(r, "game_date_game", getattr(r, "game_date", pd.NaT)),
            "season": int(getattr(r, "season_game", getattr(r, "season", pd.Timestamp(getattr(r, "game_date")).year))),
            "inning": inning,
            "dk_total_open_total": getattr(r, "dk_total_open_total"),
            "away_team_id": getattr(r, "away_team_id"),
            "home_team_id": getattr(r, "home_team_id"),
            "away_team_code": getattr(r, "away_team_code"),
            "home_team_code": getattr(r, "home_team_code"),
            "game_number": int(getattr(r, "game_number")),
        }
        away_played = bool(getattr(r, "away_half_played"))
        home_played = bool(getattr(r, "home_half_played"))
        away_runs = pd.to_numeric(pd.Series([getattr(r, "away_runs")]), errors="coerce").iloc[0]
        home_runs = pd.to_numeric(pd.Series([getattr(r, "home_runs")]), errors="coerce").iloc[0]
        rows.append({**common,
                     "half": "top",
                     "batting_team_id": getattr(r, "away_team_id"),
                     "pitching_team_id": getattr(r, "home_team_id"),
                     "half_played": away_played,
                     "half_runs": float(away_runs) if away_played and pd.notna(away_runs) else np.nan})
        rows.append({**common,
                     "half": "bottom",
                     "batting_team_id": getattr(r, "home_team_id"),
                     "pitching_team_id": getattr(r, "away_team_id"),
                     "half_played": home_played,
                     "half_runs": float(home_runs) if home_played and pd.notna(home_runs) else np.nan})

    out = pd.DataFrame(rows)
    out["half_scored"] = np.where(out["half_played"] & out["half_runs"].notna(), (out["half_runs"] >= 1).astype(float), np.nan)
    out["target_class"] = np.where(out["half_played"], "PLAYED_TARGET", "UNPLAYED_HALF")
    out["holdout_target_loaded"] = False
    return out.sort_values(["game_date", "game_id", "inning", "half"], kind="stable").reset_index(drop=True)


def audit(out: pd.DataFrame) -> dict:
    dup = int(out.duplicated(["game_id", "inning", "half"]).sum())
    by_inning = {}
    for inn in INNINGS:
        g = out[out["inning"] == inn]
        by_inning[str(inn)] = {
            "rows": int(len(g)),
            "played_halves": int(g["half_played"].sum()),
            "unplayed_halves": int((~g["half_played"]).sum()),
            "scoring_halves": int(g["half_scored"].fillna(0).sum()),
        }
    b9 = out[(out["inning"] == 9) & (out["half"] == "bottom")]
    return {
        "status": "PASS" if dup == 0 and len(out) else "FAIL",
        "development_seasons": sorted(int(x) for x in out["season"].dropna().unique()),
        "max_season": int(out["season"].max()) if len(out) else None,
        "rows": int(len(out)),
        "games": int(out["game_id"].nunique()),
        "duplicate_game_inning_half_rows": dup,
        "holdout_target_loaded": bool(out["holdout_target_loaded"].any()) if len(out) else False,
        "bottom9": {
            "rows": int(len(b9)),
            "played": int(b9["half_played"].sum()),
            "unplayed": int((~b9["half_played"]).sum()),
        },
        "by_inning": by_inning,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--reusable-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    master = master_dev(args.master)
    game_index = read(find_one(args.reusable_root, "game_index.parquet"))
    inning_outcomes = read(find_one(args.reusable_root, "inning_outcomes.parquet"))
    out = build_half_spine(master, game_index, inning_outcomes)
    if len(out) == 0:
        raise RuntimeError("all-inning development matrix is empty")
    if out["season"].max() > DEV_MAX_SEASON:
        raise RuntimeError("2025 holdout target leaked into development matrix")
    if out.duplicated(["game_id", "inning", "half"]).any():
        raise RuntimeError("non-unique game x inning x half rows")
    out.to_parquet(args.output, index=False)
    manifest = audit(out)
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

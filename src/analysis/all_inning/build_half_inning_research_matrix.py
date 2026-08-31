#!/usr/bin/env python3
"""Build the canonical I1-I9 half-inning development matrix.

Governing unit: game x inning x half. This builder intentionally stops at the
state/target layer: it does not fit a run model, does not add arbitrary
shrinkage, and does not inspect sportsbook inning markets.

2025 is a protected final holdout. Development output is hard-limited to
2021-2024. The historical master contributes only the isolated opening full-game
total and join keys. Half-inning outcomes come from the reusable normalized MLB
feed artifact, where played-vs-not-played is explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEAM_ID_TO_CODE = {
    108:"LAA",109:"ARI",110:"BAL",111:"BOS",112:"CHC",113:"CIN",114:"CLE",
    115:"COL",116:"DET",117:"HOU",118:"KC",119:"LAD",120:"WSH",121:"NYM",
    133:"OAK",134:"PIT",135:"SD",136:"SEA",137:"SF",138:"STL",139:"TB",
    140:"TEX",141:"TOR",142:"MIN",143:"PHI",144:"ATL",145:"CHW",146:"MIA",
    147:"NYY",158:"MIL",
}


def read(path: Path, columns=None):
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"{name} expected exactly once under {root}; found {hits}")
    return hits[0]


def norm_code(x):
    if pd.isna(x):
        return None
    s = str(x).upper().strip()
    aliases = {
        "ANA":"LAA","CHA":"CHW","CHN":"CHC","LAN":"LAD","NYA":"NYY",
        "NYN":"NYM","SDN":"SD","SFN":"SF","SLN":"STL","TBA":"TB",
        "KCA":"KC","WAS":"WSH",
    }
    return aliases.get(s, s)


def add_game_number(gi: pd.DataFrame) -> pd.DataFrame:
    """Recover Retrosheet game_number using schedule order only, never outcome."""
    x = gi.copy()
    keys = ["game_date", "away_team_code", "home_team_code"]
    if "game_number" in x.columns:
        x["game_number"] = pd.to_numeric(x["game_number"], errors="coerce").astype("Int64")
        return x
    if "game_datetime" not in x.columns:
        sizes = x.groupby(keys, dropna=False)["game_id"].transform("size")
        if (sizes > 1).any():
            raise RuntimeError("doubleheader present but game_datetime/game_number unavailable")
        x["game_number"] = 0
        return x
    x["_dt"] = pd.to_datetime(x["game_datetime"], errors="coerce", utc=True)
    x = x.sort_values(keys + ["_dt", "game_id"], kind="mergesort").copy()
    sizes = x.groupby(keys, dropna=False)["game_id"].transform("size")
    seq = x.groupby(keys, dropna=False).cumcount() + 1
    x["game_number"] = np.where(sizes.eq(1), 0, seq).astype(int)
    return x.drop(columns=["_dt"])


def resolve_inning_run_columns(outcomes: pd.DataFrame) -> tuple[str, str]:
    """Resolve normalized inning run columns after results/innings merge suffixing."""
    candidates = [
        ("away_runs_inning", "home_runs_inning"),
        ("away_runs", "home_runs"),
    ]
    for away_col, home_col in candidates:
        if away_col in outcomes.columns and home_col in outcomes.columns:
            return away_col, home_col
    raise RuntimeError(
        "inning_outcomes lacks inning run columns; available columns="
        + ",".join(map(str, outcomes.columns))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--reusable-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-development-season", type=int, default=2024)
    a = ap.parse_args()
    if a.max_development_season >= 2025:
        raise RuntimeError("2025 holdout protection: max development season must be <= 2024")
    a.output.parent.mkdir(parents=True, exist_ok=True)

    master_cols = [
        "game_date", "away_team_code", "home_team_code", "game_number",
        "dk_total_open_total",
    ]
    m = read(a.master, columns=master_cols).copy()
    m["game_date"] = pd.to_datetime(m["game_date"], errors="coerce").dt.normalize()
    m["season"] = m["game_date"].dt.year.astype("Int64")
    m = m[m["season"] <= a.max_development_season].copy()
    m["away_team_code"] = m["away_team_code"].map(norm_code)
    m["home_team_code"] = m["home_team_code"].map(norm_code)
    m["game_number"] = pd.to_numeric(m["game_number"], errors="coerce").astype("Int64")
    m["dk_total_open_total"] = pd.to_numeric(m["dk_total_open_total"], errors="coerce")
    m = m[m["dk_total_open_total"].notna()].copy()

    gi = read(find_one(a.reusable_root, "game_index.parquet")).copy()
    gi["game_date"] = pd.to_datetime(gi["game_date"], errors="coerce").dt.normalize()
    gi["season"] = gi["game_date"].dt.year.astype("Int64")
    gi = gi[gi["season"] <= a.max_development_season].copy()
    gi["away_team_code"] = pd.to_numeric(gi["away_team_id"], errors="coerce").map(TEAM_ID_TO_CODE)
    gi["home_team_code"] = pd.to_numeric(gi["home_team_id"], errors="coerce").map(TEAM_ID_TO_CODE)
    gi = add_game_number(gi)

    join_keys = ["game_date", "away_team_code", "home_team_code", "game_number"]
    idx_cols = join_keys + ["game_id", "away_team_id", "home_team_id"]
    idx = gi[idx_cols].copy()
    if idx.duplicated(join_keys).any() or m.duplicated(join_keys).any():
        raise RuntimeError("non-unique master/game-index join keys")
    games = m.merge(idx, on=join_keys, how="inner", validate="one_to_one")
    if len(games) != len(m):
        miss = m.merge(idx[join_keys], on=join_keys, how="left", indicator=True)
        miss = miss[miss["_merge"] == "left_only"][join_keys]
        raise RuntimeError(
            f"master/game-index join incomplete: {len(games)} of {len(m)}; "
            f"first missing={miss.head(20).to_dict('records')}"
        )

    outcomes = read(find_one(a.reusable_root, "inning_outcomes.parquet")).copy()
    outcomes["game_date"] = pd.to_datetime(outcomes["game_date"], errors="coerce").dt.normalize()
    outcomes["season"] = outcomes["game_date"].dt.year.astype("Int64")
    outcomes = outcomes[
        (outcomes["season"] <= a.max_development_season) &
        pd.to_numeric(outcomes["inning"], errors="coerce").between(1, 9)
    ].copy()
    outcomes["inning"] = pd.to_numeric(outcomes["inning"], errors="coerce").astype(int)

    away_run_col, home_run_col = resolve_inning_run_columns(outcomes)
    required_outcomes = [
        "game_id", "inning", away_run_col, home_run_col,
        "away_half_played", "home_half_played",
    ]
    missing = [c for c in required_outcomes if c not in outcomes.columns]
    if missing:
        raise RuntimeError(f"inning_outcomes missing {missing}")
    outcomes = outcomes[required_outcomes].rename(
        columns={away_run_col: "away_runs_half", home_run_col: "home_runs_half"}
    ).copy()
    if outcomes.duplicated(["game_id", "inning"]).any():
        raise RuntimeError("inning_outcomes non-unique by game_id/inning")

    base = games.merge(outcomes, on="game_id", how="inner", validate="one_to_many")
    rows = []
    for half in ("top", "bottom"):
        r = base.copy()
        if half == "top":
            r["batting_team_id"] = r["away_team_id"]
            r["pitching_team_id"] = r["home_team_id"]
            r["runs_half"] = pd.to_numeric(r["away_runs_half"], errors="coerce")
            r["half_played"] = r["away_half_played"].fillna(False).astype(bool)
        else:
            r["batting_team_id"] = r["home_team_id"]
            r["pitching_team_id"] = r["away_team_id"]
            r["runs_half"] = pd.to_numeric(r["home_runs_half"], errors="coerce")
            r["half_played"] = r["home_half_played"].fillna(False).astype(bool)
        r["half"] = half
        r["scored_half"] = np.where(
            r["half_played"], (r["runs_half"].fillna(0) >= 1).astype(int), np.nan
        )
        # A not-played half is never encoded as zero runs.
        r.loc[~r["half_played"], "runs_half"] = np.nan
        rows.append(r)

    out = pd.concat(rows, ignore_index=True)
    keep = [
        "game_id", "game_date", "season", "game_number", "away_team_code",
        "home_team_code", "inning", "half", "batting_team_id", "pitching_team_id",
        "dk_total_open_total", "half_played", "runs_half", "scored_half",
    ]
    out = out[keep].sort_values(["game_date", "game_id", "inning", "half"], kind="mergesort")
    if (out["season"] >= 2025).any():
        raise RuntimeError("2025 holdout leaked into development matrix")
    if out.duplicated(["game_id", "inning", "half"]).any():
        raise RuntimeError("canonical half-inning key is not unique")

    out.to_parquet(a.output, index=False)
    played = out[out["half_played"]].copy()
    manifest = {
        "status": "PASS",
        "architecture": "canonical_game_x_inning_x_half_development_matrix",
        "development_seasons": sorted(int(x) for x in out["season"].dropna().unique()),
        "holdout_season": 2025,
        "holdout_opened": False,
        "max_development_season": a.max_development_season,
        "games_with_opening_total": int(out["game_id"].nunique()),
        "matrix_rows_including_unplayed_halves": int(len(out)),
        "played_half_inning_rows": int(len(played)),
        "played_rows_by_inning": {str(i): int((played["inning"] == i).sum()) for i in range(1, 10)},
        "unplayed_rows_by_inning": {str(i): int(((out["inning"] == i) & ~out["half_played"]).sum()) for i in range(1, 10)},
        "opening_total_nonnull_rate": float(out["dk_total_open_total"].notna().mean()),
        "market_data_retained": ["dk_total_open_total"],
        "market_derivative_features_retained": False,
        "not_played_is_not_zero": True,
        "canonical_key": ["game_id", "inning", "half"],
        "inning_run_columns_used": {"away": away_run_col, "home": home_run_col},
    }
    a.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

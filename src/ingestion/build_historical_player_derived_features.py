#!/usr/bin/env python3
"""Build reusable leakage-safe derived player features from normalized MLB history.

Outputs:
- batter platoon as-of rates by opposing pitcher hand
- pitcher platoon as-of rates by batter hand
- starter workload / rest / retention summaries

All predictor rows are based on observations strictly before the target game date.
Same-day earlier games are excluded by aggregating to date before lagging.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS = ["single","double","triple","home_run","walk","hit_by_pitch","strikeout","ball_in_play_out"]
DERIVED = ["hit","xbh","onbase","contact"]
WINDOWS = (90, 365)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plate-appearances", type=Path, required=True)
    p.add_argument("--games", type=Path, required=True)
    p.add_argument("--starters", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shrink-strength", type=float, default=50.0)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def prep_pa(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce").dt.normalize()
    for e in EVENTS:
        x[f"ev_{e}"] = (x["event"].astype(str) == e).astype("int16")
    x["ev_hit"] = x[["ev_single","ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_xbh"] = x[["ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_onbase"] = x[["ev_single","ev_double","ev_triple","ev_home_run","ev_walk","ev_hit_by_pitch"]].sum(axis=1)
    x["ev_contact"] = 1 - x["ev_strikeout"]
    metrics = [f"ev_{e}" for e in EVENTS + DERIVED]
    return x, metrics


def build_platoon(pa: pd.DataFrame, metrics: list[str], entity: str, split_col: str, strength: float) -> pd.DataFrame:
    entity_col = "batter_id" if entity == "batter" else "pitcher_id"
    x = pa.dropna(subset=[entity_col, split_col, "game_date"]).copy()
    x[split_col] = x[split_col].astype(str).str.upper()
    x = x[x[split_col].isin(["L","R"])].copy()
    daily = x.groupby([entity_col, split_col, "game_date"])[metrics].sum().reset_index()
    daily = daily.rename(columns={entity_col:"entity_id", split_col:"split_hand"})
    daily["opportunities"] = daily[[f"ev_{e}" for e in EVENTS]].sum(axis=1)

    # League prior by split hand, also strictly prior-date.
    ld = x.groupby([split_col,"game_date"])[metrics].sum().reset_index().rename(columns={split_col:"split_hand"})
    ld["opportunities"] = ld[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    priors = []
    for hand, g in ld.groupby("split_hand"):
        g = g.sort_values("game_date").copy()
        vals = g[metrics + ["opportunities"]].shift(1).fillna(0).cumsum()
        r = vals[metrics].div(vals["opportunities"].replace(0, np.nan), axis=0).fillna(0)
        r.columns = [f"league_{c}" for c in metrics]
        r["game_date"] = g["game_date"].values
        r["split_hand"] = hand
        priors.append(r)
    prior = pd.concat(priors, ignore_index=True) if priors else pd.DataFrame()

    out_parts = []
    for (eid, hand), g in daily.groupby(["entity_id","split_hand"], sort=False):
        g = g.sort_values("game_date").copy()
        g["season"] = g["game_date"].dt.year
        base = g[["entity_id","split_hand","game_date"]].copy()
        cols = metrics + ["opportunities"]
        for c in cols:
            base[f"season_{c}"] = g.groupby("season", sort=False)[c].transform(lambda s: s.shift(1).fillna(0).cumsum())
        for days in WINDOWS:
            r = g.set_index("game_date")[cols].rolling(f"{days}D", closed="left").sum().fillna(0).reset_index()
            for c in cols:
                base[f"{days}d_{c}"] = r[c].values
        base["entity_type"] = entity
        out_parts.append(base)
    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()
    if out.empty:
        return out
    out = out.merge(prior, on=["split_hand","game_date"], how="left")
    for prefix in ["season"] + [f"{d}d" for d in WINDOWS]:
        opp = pd.to_numeric(out[f"{prefix}_opportunities"], errors="coerce").fillna(0.0)
        out[f"{prefix}_reliability"] = opp / (opp + strength)
        for c in metrics:
            cnt = pd.to_numeric(out[f"{prefix}_{c}"], errors="coerce").fillna(0.0)
            lp = pd.to_numeric(out[f"league_{c}"], errors="coerce").fillna(0.0)
            out[f"{prefix}_{c}_rate_shrunk"] = (cnt + strength * lp) / (opp + strength)
    out = out.drop(columns=[c for c in out.columns if c.startswith("league_")])
    return out.rename(columns={"game_date":"as_of_date"})


def build_starter_workload(pa: pd.DataFrame, games: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    g = games.copy(); g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    s = starters.copy(); s["game_date"] = pd.to_datetime(s["game_date"], errors="coerce").dt.normalize()
    p = pa.copy()

    # Per game pitcher workload derived from actual PAs faced and maximum inning reached.
    pg = p.groupby(["game_id","pitcher_id","game_date"]).agg(
        batters_faced=("batter_id","size"),
        max_inning=("inning","max"),
        pitches_proxy=("play_index","size"),
    ).reset_index()
    first_pitcher = s[["game_id","game_date","team_side","team_id","pitcher_id"]].dropna(subset=["pitcher_id"]).copy()
    st = first_pitcher.merge(pg, on=["game_id","pitcher_id","game_date"], how="left")
    st["batters_faced"] = pd.to_numeric(st["batters_faced"], errors="coerce").fillna(0)
    st["max_inning"] = pd.to_numeric(st["max_inning"], errors="coerce")
    for inn in range(2, 8):
        st[f"reached_i{inn}"] = (st["max_inning"] >= inn).astype(int)

    rows = []
    for pid, x in st.groupby("pitcher_id", sort=False):
        x = x.sort_values("game_date").copy()
        hist_dates = []
        for i, row in x.reset_index(drop=True).iterrows():
            d = row["game_date"]
            h = x[x["game_date"] < d].copy()
            h30 = h[h["game_date"] >= d - pd.Timedelta(days=30)]
            h90 = h[h["game_date"] >= d - pd.Timedelta(days=90)]
            prev_date = h["game_date"].max() if not h.empty else pd.NaT
            rec = {
                "pitcher_id": pid,
                "as_of_date": d,
                "days_rest": (d - prev_date).days if pd.notna(prev_date) else np.nan,
                "prior_starts": int(len(h)),
                "starts_30d": int(len(h30)),
                "starts_90d": int(len(h90)),
                "bf_per_start_30d": float(h30["batters_faced"].mean()) if len(h30) else np.nan,
                "bf_per_start_90d": float(h90["batters_faced"].mean()) if len(h90) else np.nan,
                "bf_sd_90d": float(h90["batters_faced"].std(ddof=0)) if len(h90) else np.nan,
            }
            for inn in range(2,8):
                rec[f"starter_retention_i{inn}_90d"] = float(h90[f"reached_i{inn}"].mean()) if len(h90) else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    a = parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    pa, metrics = prep_pa(read(a.plate_appearances))
    games = read(a.games); starters = read(a.starters)
    batter = build_platoon(pa, metrics, "batter", "pitcher_hand", a.shrink_strength)
    pitcher = build_platoon(pa, metrics, "pitcher", "batter_side", a.shrink_strength)
    workload = build_starter_workload(pa, games, starters)
    batter.to_parquet(a.output_dir / "batter_platoon_asof.parquet", index=False)
    pitcher.to_parquet(a.output_dir / "pitcher_platoon_asof.parquet", index=False)
    workload.to_parquet(a.output_dir / "starter_workload_asof.parquet", index=False)
    print({"batter_platoon_rows":len(batter),"pitcher_platoon_rows":len(pitcher),"starter_workload_rows":len(workload)})

if __name__ == "__main__":
    main()

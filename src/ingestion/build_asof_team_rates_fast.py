#!/usr/bin/env python3
"""Build leakage-safe team batting and pitching-allowed event rates.

Outputs are generic team-strength/context features for reuse beyond inning models.
Only observations from dates strictly before the target date are used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS = ["single","double","triple","home_run","walk","hit_by_pitch","strikeout","ball_in_play_out"]
DERIVED = ["hit","xbh","onbase","contact"]
WINDOWS = (30, 90, 365)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plate-appearances", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shrink-strength", type=float, default=200.0)
    return p.parse_args()


def read(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def prep(df):
    x = df.copy()
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce").dt.normalize()
    x = x[x["game_date"].notna()].copy()
    for e in EVENTS:
        x[f"ev_{e}"] = (x["event"].astype(str) == e).astype("int16")
    x["ev_hit"] = x[["ev_single","ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_xbh"] = x[["ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_onbase"] = x[["ev_single","ev_double","ev_triple","ev_home_run","ev_walk","ev_hit_by_pitch"]].sum(axis=1)
    x["ev_contact"] = 1 - x["ev_strikeout"]
    return x, [f"ev_{e}" for e in EVENTS] + [f"ev_{e}" for e in DERIVED]


def league_prior(pa, metrics):
    d = pa.groupby("game_date")[metrics].sum().sort_index()
    d["opp"] = d[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    prior = d.shift(1).fillna(0).cumsum()
    r = prior[metrics].div(prior["opp"].replace(0, np.nan), axis=0).fillna(0)
    r.columns = [f"league_{c}" for c in metrics]
    return r


def daily(pa, team_col, metrics):
    d = pa.dropna(subset=[team_col]).groupby([team_col,"game_date"])[metrics].sum().reset_index()
    d = d.rename(columns={team_col:"team_id"}).sort_values(["team_id","game_date"])
    d["opportunities"] = d[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    return d


def counts(d, metrics):
    cols = metrics + ["opportunities"]
    x = d.copy(); x["season"] = x["game_date"].dt.year
    additions = {}
    for c in cols:
        additions[f"season_{c}"] = x.groupby(["team_id","season"], sort=False)[c].transform(lambda s: s.shift(1).fillna(0).cumsum())
    out = pd.concat([x[["team_id","game_date"]].reset_index(drop=True), pd.DataFrame(additions)], axis=1)
    for days in WINDOWS:
        parts = []
        for tid, g in d.groupby("team_id", sort=False):
            r = g.sort_values("game_date").set_index("game_date")[cols].rolling(f"{days}D", closed="left").sum().fillna(0)
            r = r.add_prefix(f"{days}d_").reset_index(); r.insert(0,"team_id",tid); parts.append(r)
        out = out.merge(pd.concat(parts, ignore_index=True), on=["team_id","game_date"], how="left")
    return out


def rates(frame, prefix, metrics, league, strength):
    x = frame.merge(league.reset_index(), on="game_date", how="left")
    opp = pd.to_numeric(x[f"{prefix}_opportunities"], errors="coerce").fillna(0)
    a = {f"{prefix}_reliability": opp/(opp+strength)}
    for c in metrics:
        n = pd.to_numeric(x[f"{prefix}_{c}"], errors="coerce").fillna(0)
        lp = pd.to_numeric(x[f"league_{c}"], errors="coerce").fillna(0)
        a[f"{prefix}_{c}_rate_raw"] = n/opp.replace(0,np.nan)
        a[f"{prefix}_{c}_rate_shrunk"] = (n + strength*lp)/(opp+strength)
    return pd.concat([x.drop(columns=[f"league_{c}" for c in metrics]), pd.DataFrame(a,index=x.index)], axis=1).copy()


def build(pa, role, team_col, metrics, league, strength):
    out = counts(daily(pa, team_col, metrics), metrics)
    for prefix in ["season","30d","90d","365d"]:
        out = rates(out, prefix, metrics, league, strength)
    out.insert(1,"team_role",role)
    return out.rename(columns={"game_date":"as_of_date"})


def main():
    args = parse_args()
    pa, metrics = prep(read(args.plate_appearances))
    league = league_prior(pa, metrics)
    offense = build(pa,"batting","batting_team_id",metrics,league,args.shrink_strength)
    defense = build(pa,"pitching_allowed","pitching_team_id",metrics,league,args.shrink_strength)
    out = pd.concat([offense,defense],ignore_index=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(args.output,index=False)
    print({"team_snapshots":len(out),"output":str(args.output)})


if __name__ == "__main__":
    main()

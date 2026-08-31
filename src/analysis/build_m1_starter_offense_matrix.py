#!/usr/bin/env python3
"""Build the game-level M1 starter + opposing-offense research matrix.

This script intentionally contains no market modeling. It only joins the isolated
pregame full-game total/outcome ledger to leakage-safe baseball features.

Required inputs
---------------
master CSV/CSV.GZ:
    game_date, away_team_code, home_team_code, dk_total_open_total,
    inning2_total_runs
reusable feature-store directory:
    game_index.parquet, starter_asof.parquet, game_team_asof.parquet

Governance
----------
* Statistics come from the reusable feature store and are strictly prior-date.
* The opening total is used only as the already-approved M0 conditioning variable.
* No moneyline, run-line, inning-market price, current total, or market derivative
  is retained in the output.
* Final-feed starter identities remain explicitly retrospective/unverified pregame;
  this matrix is research Tier B until independent historical identity timing is
  supplied.
* Two half-inning matchup measurements are averaged to one full-I2 game feature.
  For standardized regression this is equivalent to summing the two halves up to
  a constant scale, so it does not impose a relative half-inning weight.
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
    147:"NYY",158:"MIL"
}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet": return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def pick(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns: return c
    raise KeyError(f"None of the required columns exist: {candidates}")


def rate_cols(df: pd.DataFrame):
    return {
        "k": pick(df,["365d_ev_strikeout_rate_shrunk","365d_strikeout_rate_shrunk","season_ev_strikeout_rate_shrunk"]),
        "bb": pick(df,["365d_ev_walk_rate_shrunk","365d_walk_rate_shrunk","season_ev_walk_rate_shrunk"]),
        "hr": pick(df,["365d_ev_home_run_rate_shrunk","365d_home_run_rate_shrunk","season_ev_home_run_rate_shrunk"]),
        "hit": pick(df,["365d_ev_hit_rate_shrunk","365d_hit_rate_shrunk","season_ev_hit_rate_shrunk"]),
    }


def norm_code(x):
    if pd.isna(x): return None
    s=str(x).upper().strip()
    aliases={"ANA":"LAA","CHA":"CHW","CHN":"CHC","LAN":"LAD","NYA":"NYY","NYN":"NYM","OAK":"OAK","SDN":"SD","SFN":"SF","SLN":"STL","TBA":"TB","KCA":"KC","WAS":"WSH"}
    return aliases.get(s,s)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--feature-store", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a=ap.parse_args(); a.output.parent.mkdir(parents=True, exist_ok=True)

    m=read_table(a.master)
    needed=["game_date","away_team_code","home_team_code","dk_total_open_total","inning2_total_runs"]
    miss=[c for c in needed if c not in m.columns]
    if miss: raise RuntimeError(f"Historical master missing required fields: {miss}")
    m=m[needed].copy()
    m["game_date"]=pd.to_datetime(m.game_date,errors="coerce").dt.normalize()
    m["away_team_code"]=m.away_team_code.map(norm_code)
    m["home_team_code"]=m.home_team_code.map(norm_code)
    m["dk_total_open_total"]=pd.to_numeric(m.dk_total_open_total,errors="coerce")
    m["inning2_total_runs"]=pd.to_numeric(m.inning2_total_runs,errors="coerce")
    m=m[m.dk_total_open_total.notna() & m.inning2_total_runs.notna()].copy()

    gi=read_table(a.feature_store/"game_index.parquet")
    st=read_table(a.feature_store/"starter_asof.parquet")
    tg=read_table(a.feature_store/"game_team_asof.parquet")
    for d in (gi,st,tg):
        d["game_date"]=pd.to_datetime(d.game_date,errors="coerce").dt.normalize()

    gi["away_team_code"]=pd.to_numeric(gi.away_team_id,errors="coerce").map(TEAM_ID_TO_CODE)
    gi["home_team_code"]=pd.to_numeric(gi.home_team_id,errors="coerce").map(TEAM_ID_TO_CODE)
    idx=gi[["game_id","game_date","away_team_id","home_team_id","away_team_code","home_team_code"]].copy()
    joined=m.merge(idx,on=["game_date","away_team_code","home_team_code"],how="inner",validate="one_to_one")

    # Starter features: one actual first pitcher for each team side. Statistics are as-of safe.
    sr=rate_cols(st)
    starter_keep=["game_id","team_id","team_side","identity_timing_class","statistics_timing_class",*sr.values()]
    s=st[starter_keep].copy()
    s=s.rename(columns={sr["k"]:"starter_k",sr["bb"]:"starter_bb",sr["hr"]:"starter_hr",sr["hit"]:"starter_hit"})

    # Team batting history is the opposing-offense signal for M1. This deliberately avoids
    # batting-order-path features, which belong to M3.
    tr=rate_cols(tg)
    t=tg[["game_id","team_id","side","statistics_timing_class",*tr.values()]].copy()
    t=t.rename(columns={tr["k"]:"off_k",tr["bb"]:"off_bb",tr["hr"]:"off_hr",tr["hit"]:"off_hit"})

    halves=[]
    for batting_side, pitching_side in [("away","home"),("home","away")]:
        off=t[t.side==batting_side].copy()
        pit=s[s.team_side==pitching_side].copy()
        h=joined[["game_id","game_date","away_team_code","home_team_code","dk_total_open_total","inning2_total_runs"]].merge(
            off,on="game_id",how="left",suffixes=("","_off"))
        h=h.merge(pit,on="game_id",how="left",suffixes=("_off","_pit"))
        h["half"]=batting_side
        halves.append(h)
    h=pd.concat(halves,ignore_index=True)

    numeric=["starter_k","starter_bb","starter_hr","starter_hit","off_k","off_bb","off_hr","off_hit"]
    for c in numeric: h[c]=pd.to_numeric(h[c],errors="coerce")
    h["starter_nonhr_hit"]=h.starter_hit-h.starter_hr
    h["off_nonhr_hit"]=h.off_hit-h.off_hr
    h["contact_interaction_half"]=(1-h.starter_k)*(1-h.off_k)
    h["power_interaction_half"]=h.starter_hr*h.off_hr
    h["baserunner_interaction_half"]=h.starter_bb*h.off_bb

    agg=(h.groupby(["game_id","game_date","away_team_code","home_team_code","dk_total_open_total","inning2_total_runs"],as_index=False)
         .agg(starter_k_rate=("starter_k","mean"), starter_bb_rate=("starter_bb","mean"),
              starter_hr_rate=("starter_hr","mean"), starter_nonhr_hit_rate=("starter_nonhr_hit","mean"),
              opponent_k_rate=("off_k","mean"), opponent_bb_rate=("off_bb","mean"),
              opponent_hr_rate=("off_hr","mean"), opponent_nonhr_hit_rate=("off_nonhr_hit","mean"),
              contact_interaction=("contact_interaction_half","mean"), power_interaction=("power_interaction_half","mean"),
              baserunner_interaction=("baserunner_interaction_half","mean")))
    agg["season"]=agg.game_date.dt.year.astype(int)
    agg["starter_identity_timing_class"]="retrospective_actual_first_pitcher_unverified_pregame"
    agg["statistics_timing_class"]="asof_safe_strictly_prior_date"
    agg["market_columns_retained"]="dk_total_open_total_only"
    # Platoon is intentionally absent until the separate leakage-safe split artifact is joined.
    agg.to_parquet(a.output,index=False)

    manifest={
        "rows":int(len(agg)), "seasons":sorted(agg.season.unique().tolist()),
        "opening_total_nonnull":int(agg.dk_total_open_total.notna().sum()),
        "i2_outcome_nonnull":int(agg.inning2_total_runs.notna().sum()),
        "m1_features":[c for c in ["starter_k_rate","starter_bb_rate","starter_hr_rate","starter_nonhr_hit_rate","opponent_k_rate","opponent_bb_rate","opponent_hr_rate","opponent_nonhr_hit_rate","contact_interaction","power_interaction","baserunner_interaction"] if c in agg.columns],
        "platoon_advantage_share":"not_joined_in_base M1 matrix; requires separate split artifact",
        "future_information_in_statistics":False,
        "starter_identity_pregame_verified":False,
        "market_data_retained":["dk_total_open_total"],
        "market_derivative_features_retained":False
    }
    mp=a.output.with_suffix(".manifest.json"); mp.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()

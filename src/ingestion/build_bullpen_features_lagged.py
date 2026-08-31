#!/usr/bin/env python3
"""Build leakage-safe bullpen features with strictly prior-date league priors.

This supersedes the shrunk bullpen rate columns emitted by
build_historical_order_bullpen_features.py. Raw rolling rates are always emitted.
Shrunk rates are explicitly candidate features until development-only validation
selects a shrinkage strength; 2025 is not used for tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS = ["single","double","triple","home_run","walk","hit_by_pitch","strikeout","ball_in_play_out"]
DERIVED = ["hit","xbh","onbase","contact"]
QUALITY_WINDOWS = (30,90,365)
WORKLOAD_WINDOWS = (1,2,3,7,14)


def read(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def prep(pa: pd.DataFrame, starters: pd.DataFrame) -> tuple[pd.DataFrame,list[str]]:
    x=pa.copy()
    x["game_date"]=pd.to_datetime(x["game_date"],errors="coerce").dt.normalize()
    for e in EVENTS:
        x[f"ev_{e}"]=(x["event"].astype(str)==e).astype("int16")
    x["ev_hit"]=x[["ev_single","ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_xbh"]=x[["ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_onbase"]=x[["ev_single","ev_double","ev_triple","ev_home_run","ev_walk","ev_hit_by_pitch"]].sum(axis=1)
    x["ev_contact"]=1-x["ev_strikeout"]
    st=starters[["game_id","team_id","pitcher_id"]].dropna(subset=["pitcher_id"]).copy()
    st=st.rename(columns={"team_id":"pitching_team_id","pitcher_id":"starter_id"})
    x=x.merge(st,on=["game_id","pitching_team_id"],how="left")
    x["is_relief"]=x["starter_id"].notna() & (x["pitcher_id"]!=x["starter_id"])
    return x,[f"ev_{e}" for e in EVENTS+DERIVED]


def league_prior_by_date(relief: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    d=(relief.groupby("game_date")[metrics].sum().sort_index())
    d["opportunities"]=d[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    prior_counts=d[metrics].cumsum().shift(1).fillna(0.0)
    prior_opp=d["opportunities"].cumsum().shift(1).fillna(0.0)
    out=pd.DataFrame(index=d.index)
    out["league_prior_opportunities"]=prior_opp
    for m in metrics:
        out[f"league_prior_{m}_rate"]=np.where(prior_opp>0,prior_counts[m]/prior_opp,np.nan)
    return out.reset_index()


def add_rate_columns(out: pd.DataFrame, prefix: str, metrics: list[str], strength: float) -> pd.DataFrame:
    opp=pd.to_numeric(out[f"{prefix}_opportunities"],errors="coerce").fillna(0.0)
    out[f"{prefix}_raw_support"]=opp
    for m in metrics:
        cnt=pd.to_numeric(out[f"{prefix}_{m}"],errors="coerce").fillna(0.0)
        out[f"{prefix}_{m}_rate_raw"]=np.where(opp>0,cnt/opp,np.nan)
        prior=pd.to_numeric(out[f"league_prior_{m}_rate"],errors="coerce")
        if strength>0:
            out[f"{prefix}_{m}_rate_candidate_shrunk"]=(cnt+strength*prior)/(opp+strength)
        else:
            out[f"{prefix}_{m}_rate_candidate_shrunk"]=out[f"{prefix}_{m}_rate_raw"]
    out[f"{prefix}_candidate_reliability"]=np.where(opp+strength>0,opp/(opp+strength),0.0)
    return out


def build_entity(relief: pd.DataFrame, metrics: list[str], entity_col: str, strength: float, team: bool) -> pd.DataFrame:
    r=relief.dropna(subset=[entity_col,"game_date"]).copy()
    daily=r.groupby([entity_col,"game_date"])[metrics].sum().reset_index()
    daily["opportunities"]=daily[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    if team:
        used=(r.groupby([entity_col,"game_date"])["pitcher_id"].nunique().rename("relievers_used").reset_index())
        daily=daily.merge(used,on=[entity_col,"game_date"],how="left")
    league=league_prior_by_date(relief,metrics)
    cols=metrics+["opportunities"]
    parts=[]
    for ent,g in daily.groupby(entity_col,sort=False):
        g=g.sort_values("game_date").copy()
        base=g[[entity_col,"game_date"]].copy()
        for days in QUALITY_WINDOWS:
            z=g.set_index("game_date")[cols].rolling(f"{days}D",closed="left").sum().fillna(0).reset_index()
            for c in cols:
                base[f"{days}d_{c}"]=z[c].values
        for days in WORKLOAD_WINDOWS:
            wcols=["opportunities"]+(["relievers_used"] if team else [])
            z=g.set_index("game_date")[wcols].rolling(f"{days}D",closed="left").sum().fillna(0).reset_index()
            base[("bullpen_bf_" if team else "relief_bf_")+f"{days}d"]=z["opportunities"].values
            if team:
                base[f"reliever_uses_{days}d"]=z["relievers_used"].values
        if not team:
            base["last_relief_date"]=g["game_date"].shift(1)
            base["days_since_relief"]=(g["game_date"]-base["last_relief_date"]).dt.days
        parts.append(base)
    out=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return out
    out=out.merge(league,on="game_date",how="left",validate="many_to_one")
    for days in QUALITY_WINDOWS:
        out=add_rate_columns(out,f"{days}d",metrics,strength)
    out=out.rename(columns={"game_date":"as_of_date"})
    out["candidate_shrink_strength"]=float(strength)
    out["league_prior_cutoff"]="strictly before as_of_date"
    out["source_class"]="relief_history_and_league_prior_strictly_prior_date"
    if team:
        out=out.rename(columns={entity_col:"team_id"})
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plate-appearances",type=Path,required=True)
    ap.add_argument("--starters",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--pitcher-candidate-strength",type=float,default=75.0)
    ap.add_argument("--team-candidate-strength",type=float,default=150.0)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa,metrics=prep(read(a.plate_appearances),read(a.starters))
    relief=pa[pa["is_relief"]].copy()
    pitcher=build_entity(relief,metrics,"pitcher_id",a.pitcher_candidate_strength,False)
    team=build_entity(relief,metrics,"pitching_team_id",a.team_candidate_strength,True)
    pitcher.to_parquet(a.output_dir/"bullpen_pitcher_asof.parquet",index=False)
    team.to_parquet(a.output_dir/"bullpen_team_asof.parquet",index=False)
    manifest={
      "future_aggregate_leakage":False,
      "league_prior":"cumulative league relief outcomes strictly before as_of_date",
      "same_day_prior_games_included":False,
      "raw_rates_emitted":True,
      "candidate_shrinkage_production_approved":False,
      "pitcher_candidate_strength":a.pitcher_candidate_strength,
      "team_candidate_strength":a.team_candidate_strength,
      "tuning_rule":"select only on 2021-2024 chronological development; 2025 untouched",
      "pitcher_rows":int(len(pitcher)),"team_rows":int(len(team))
    }
    (a.output_dir/"bullpen_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(manifest)

if __name__=="__main__": main()

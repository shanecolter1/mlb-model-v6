#!/usr/bin/env python3
"""
Build pitch-count/state-dependent half-inning run probabilities from Statcast.

Data source:
  Baseball Savant via pybaseball.statcast, regular season 2021-2025.

This is a research artifact. It uses baseball data only and contains no market inputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

LEGAL_COUNTS={(b,s) for b in range(4) for s in range(3)}
EVENT_MAP={
    "strikeout":"strikeout",
    "strikeout_double_play":"strikeout",
    "walk":"walk",
    "intent_walk":"walk",
    "hit_by_pitch":"hbp",
    "single":"single",
    "double":"double",
    "triple":"triple",
    "home_run":"home_run",
    "field_out":"bip_out",
    "force_out":"bip_out",
    "grounded_into_double_play":"bip_out",
    "fielders_choice_out":"bip_out",
    "sac_fly":"bip_out",
    "sac_bunt":"bip_out",
    "double_play":"bip_out",
    "triple_play":"bip_out",
}

def base_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["on_1b"].notna().astype(int)
        + 2*df["on_2b"].notna().astype(int)
        + 4*df["on_3b"].notna().astype(int)
    )

def download_statcast(start: str, end: str, cache_dir: Path, chunk_days: int=7) -> pd.DataFrame:
    from pybaseball import statcast
    cache_dir.mkdir(parents=True,exist_ok=True)
    start_dt=pd.Timestamp(start); end_dt=pd.Timestamp(end)
    pieces=[]
    cur=start_dt
    while cur<=end_dt:
        stop=min(cur+pd.Timedelta(days=chunk_days-1),end_dt)
        fp=cache_dir/f"statcast_{cur:%Y%m%d}_{stop:%Y%m%d}.parquet"
        if fp.exists():
            part=pd.read_parquet(fp)
        else:
            part=statcast(start_dt=cur.strftime("%Y-%m-%d"),end_dt=stop.strftime("%Y-%m-%d"))
            part.to_parquet(fp,index=False)
        pieces.append(part)
        cur=stop+pd.Timedelta(days=1)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()

def clean_pitch_states(df: pd.DataFrame) -> pd.DataFrame:
    required=[
        "game_pk","game_date","inning","inning_topbot","at_bat_number","pitch_number",
        "balls","strikes","outs_when_up","on_1b","on_2b","on_3b",
        "bat_score","fld_score","post_bat_score","events","batter","pitcher","game_type"
    ]
    miss=[c for c in required if c not in df.columns]
    if miss: raise ValueError(f"Missing Statcast columns: {miss}")
    x=df[required].copy()
    x=x[x["game_type"].eq("R")].copy()
    for c in ["balls","strikes","outs_when_up","inning","at_bat_number","pitch_number","bat_score","post_bat_score"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x[x[["balls","strikes","outs_when_up"]].notna().all(axis=1)]
    x=x[x.apply(lambda r:(int(r.balls),int(r.strikes)) in LEGAL_COUNTS and int(r.outs_when_up)<3,axis=1)]
    x["base_mask"]=base_mask(x)
    x["half_id"]=(
        x["game_pk"].astype("Int64").astype(str)+"|"+
        x["inning"].astype("Int64").astype(str)+"|"+
        x["inning_topbot"].astype(str)
    )

    # Score for batting team at this pitch. Statcast bat_score is the score before the pitch.
    # Final batting-team half score is max post_bat_score observed in the half minus score entering half.
    half=x.groupby("half_id",sort=False)
    x["half_start_score"]=half["bat_score"].transform("min")
    x["half_final_score"]=half["post_bat_score"].transform("max")
    x["runs_scored_so_far"]=x["bat_score"]-x["half_start_score"]
    x["additional_runs"]=x["half_final_score"]-x["bat_score"]
    x["additional_runs"]=x["additional_runs"].clip(lower=0)
    x["count"]=x["balls"].astype(int).astype(str)+"-"+x["strikes"].astype(int).astype(str)
    return x

def summarize_states(x: pd.DataFrame) -> pd.DataFrame:
    keys=["outs_when_up","base_mask","balls","strikes","count"]
    def one(g):
        r=g["additional_runs"].astype(int)
        return pd.Series({
            "N":len(g),
            "half_innings":g["half_id"].nunique(),
            "p0":(r==0).mean(),
            "p_exact1":(r==1).mean(),
            "p_exact2":(r==2).mean(),
            "p_exact3":(r==3).mean(),
            "p4plus":(r>=4).mean(),
            "p1plus":(r>=1).mean(),
            "p2plus":(r>=2).mean(),
            "p3plus":(r>=3).mean(),
            "expected_additional_runs":r.mean(),
        })
    return x.groupby(keys,dropna=False).apply(one,include_groups=False).reset_index()

def summarize_count(x: pd.DataFrame) -> pd.DataFrame:
    def one(g):
        r=g["additional_runs"].astype(int)
        return pd.Series({
            "N":len(g),"half_innings":g["half_id"].nunique(),
            "p0":(r==0).mean(),"p1plus":(r>=1).mean(),
            "p2plus":(r>=2).mean(),"p3plus":(r>=3).mean(),"p4plus":(r>=4).mean(),
            "expected_additional_runs":r.mean(),
        })
    return x.groupby(["balls","strikes","count"]).apply(one,include_groups=False).reset_index()

def summarize_pa_outcomes(x: pd.DataFrame) -> pd.DataFrame:
    # Final pitch row identifies PA terminal event; pre-pitch count on final pitch is its terminal-entry count.
    final=x.sort_values(["game_pk","at_bat_number","pitch_number"]).groupby(
        ["game_pk","at_bat_number"],as_index=False
    ).tail(1).copy()
    final["event_class"]=final["events"].map(EVENT_MAP).fillna("other")
    tab=(final.groupby(["balls","strikes","count","event_class"]).size()
         .rename("n").reset_index())
    tab["N_count"]=tab.groupby(["balls","strikes","count"])["n"].transform("sum")
    tab["probability"]=tab["n"]/tab["N_count"]
    return tab.sort_values(["balls","strikes","event_class"])

def beta_shrink(raw: pd.DataFrame, count_summary: pd.DataFrame, prior_strength: float=250.0) -> pd.DataFrame:
    """
    Empirical-Bayes shrinkage of cumulative probabilities toward the count-only rate.
    This is a transparent first challenger, not the final production smoother.
    """
    out=raw.merge(
        count_summary[["balls","strikes","p0","p1plus","p2plus","p3plus","p4plus"]],
        on=["balls","strikes"],suffixes=("","_count")
    )
    for col in ["p0","p1plus","p2plus","p3plus","p4plus"]:
        k=out[col]*out["N"]
        prior=out[f"{col}_count"]
        out[f"{col}_smoothed"]=(k+prior_strength*prior)/(out["N"]+prior_strength)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2021-04-01")
    ap.add_argument("--end",default="2025-09-30")
    ap.add_argument("--cache",default="data/raw/statcast_pitch_state")
    ap.add_argument("--out",default="data/derived/count_state")
    ap.add_argument("--input-parquet",default=None,
                    help="Optional pre-materialized Statcast parquet; skips downloading.")
    ap.add_argument("--prior-strength",type=float,default=250.0)
    args=ap.parse_args()

    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if args.input_parquet:
        raw=pd.read_parquet(args.input_parquet)
    else:
        raw=download_statcast(args.start,args.end,Path(args.cache))

    states=clean_pitch_states(raw)
    state_raw=summarize_states(states)
    count=summarize_count(states)
    pa=summarize_pa_outcomes(states)
    smooth=beta_shrink(state_raw,count,args.prior_strength)

    state_raw.to_csv(out/"count_state_raw.csv",index=False)
    smooth.to_csv(out/"count_state_smoothed.csv",index=False)
    count.to_csv(out/"count_only_summary.csv",index=False)
    pa.to_csv(out/"pa_outcome_by_count.csv",index=False)

    metadata={
        "source":"Baseball Savant Statcast via pybaseball",
        "market_inputs_used":False,
        "start":args.start,"end":args.end,
        "pitch_state_rows":int(len(states)),
        "half_innings":int(states["half_id"].nunique()),
        "games":int(states["game_pk"].nunique()),
        "legal_counts":sorted({c for c in states["count"].dropna().unique()}),
        "state_dimensions":["outs_when_up","base_mask","balls","strikes"],
        "outcome":"additional runs until half-inning end",
        "smoothing":{"method":"beta shrink cumulative probabilities toward count-only baseline",
                     "prior_strength":args.prior_strength},
        "status":"research_challenger"
    }
    (out/"state_model_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__=="__main__":
    main()

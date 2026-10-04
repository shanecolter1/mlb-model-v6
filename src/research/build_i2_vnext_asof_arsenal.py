#!/usr/bin/env python3
"""Build cutoff-safe current-season arsenal sufficient statistics.

Research only. No market data. This module reconstructs the two ingredients
used by the governed arsenal matchup feature from raw Statcast:
  * pitcher pitch-type usage (all innings overall; I2 by batter side)
  * batter terminal-PA expected-wOBA response by terminal pitch type

It can also reconstruct a completed annual snapshot and compare it with the
archived Savant batter/pitcher arsenal leaderboards before any outcome scoring.
"""
from __future__ import annotations
import argparse, io, json, time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests

STATCAST_URL='https://baseballsavant.mlb.com/statcast_search/csv'

# Statcast estimated_woba_using_speedangle is not available for every terminal
# event; estimated_woba_using_speedangle is preferred, woba_value is fallback.
XWOBA_COLS=("estimated_woba_using_speedangle","woba_value")

def month_chunks(start:date,end:date):
    cur=start
    while cur<=end:
        nxt=date(cur.year+1,1,1) if cur.month==12 else date(cur.year,cur.month+1,1)
        hi=min(end,nxt-timedelta(days=1)); yield cur,hi; cur=hi+timedelta(days=1)

def fetch_month(cache:Path,lo:date,hi:date):
    cache.mkdir(parents=True,exist_ok=True)
    p=cache/f"all_{lo.isoformat()}_{hi.isoformat()}.csv"
    if p.exists(): return p
    params={"all":"true","type":"details","player_type":"pitcher","hfGT":"R|","game_date_gt":lo.isoformat(),"game_date_lt":hi.isoformat(),"group_by":"name","sort_col":"pitches","sort_order":"desc","min_pitches":"0","min_results":"0","min_abs":"0"}
    headers={"User-Agent":"MLB-I2-vNext/1.0","Accept":"text/csv,*/*"}
    last=None
    for attempt in range(4):
        try:
            r=requests.get(STATCAST_URL,params=params,headers=headers,timeout=120); r.raise_for_status()
            if not r.text.strip(): raise RuntimeError("empty Statcast response")
            pd.read_csv(io.StringIO(r.text),nrows=2)
            p.write_text(r.text); return p
        except Exception as e:
            last=e; time.sleep(2*(attempt+1))
    raise RuntimeError(f"Statcast fetch failed {lo}..{hi}: {last}")

def ensure_raw_cache(cache:Path,start:str,cutoff:str):
    lo=date.fromisoformat(start); end=date.fromisoformat(cutoff)-timedelta(days=1)
    for a,b in month_chunks(lo,end): fetch_month(cache,a,b)

def read_raw(cache:Path, start:str, cutoff:str)->pd.DataFrame:
    parts=[]
    lo=pd.Timestamp(start); hi=pd.Timestamp(cutoff)
    files=sorted(cache.glob("all_*.csv"))
    if not files: files=sorted(cache.glob("*.csv"))
    for p in files:
        try:
            x=pd.read_csv(p,low_memory=False)
        except Exception:
            continue
        if "game_date" not in x: continue
        d=pd.to_datetime(x["game_date"],errors="coerce")
        keep=(d>=lo)&(d<hi)
        if keep.any():
            y=x.loc[keep].copy(); y["game_date"]=d.loc[keep]; parts.append(y)
    if not parts: raise RuntimeError(f"No raw Statcast rows in [{start},{cutoff})")
    return pd.concat(parts,ignore_index=True)

def xwoba_series(x):
    out=pd.Series(np.nan,index=x.index,dtype=float)
    for c in XWOBA_COLS:
        if c in x:
            v=pd.to_numeric(x[c],errors="coerce")
            out=out.where(out.notna(),v)
    return out

def sufficient_stats(raw:pd.DataFrame):
    req={"pitcher","batter","pitch_type","stand","inning"}
    miss=req-set(raw.columns)
    if miss: raise RuntimeError(f"Raw Statcast missing {sorted(miss)}")
    x=raw.copy()
    x["pitcher"]=pd.to_numeric(x["pitcher"],errors="coerce")
    x["batter"]=pd.to_numeric(x["batter"],errors="coerce")
    x=x[x["pitcher"].notna()&x["batter"].notna()&x["pitch_type"].notna()].copy()
    x["pitcher"]=x["pitcher"].astype(int); x["batter"]=x["batter"].astype(int)
    overall=(x.groupby(["pitcher","pitch_type"],observed=True).size().rename("pitches").reset_index())
    i2=x[pd.to_numeric(x["inning"],errors="coerce")==2].copy()
    side=(i2[i2["stand"].isin(["L","R"])].groupby(["pitcher","stand","pitch_type"],observed=True).size().rename("pitches").reset_index())
    term=x[x.get("events",pd.Series(index=x.index,dtype=object)).notna()].copy()
    term["est_woba"]=xwoba_series(term)
    term=term[term["est_woba"].notna()]
    batter=(term.groupby(["batter","pitch_type"],observed=True)["est_woba"].agg(["sum","count"]).reset_index().rename(columns={"sum":"est_woba_sum","count":"pa"}))
    league=(term.groupby("pitch_type",observed=True)["est_woba"].agg(["sum","count"]).reset_index().rename(columns={"sum":"est_woba_sum","count":"pa"}))
    return overall,side,batter,league

def compare_annual(recon:pd.DataFrame, archived:Path, role:str):
    a=pd.read_csv(archived,low_memory=False)
    if role=="pitcher":
        r=recon.rename(columns={"pitcher":"player_id"})
        if not {"player_id","pitch_type","pitches"}.issubset(a): raise RuntimeError("Archived pitcher schema mismatch")
        m=r.merge(a[["player_id","pitch_type","pitches"]],on=["player_id","pitch_type"],suffixes=("_raw","_savant"))
        if m.empty: raise RuntimeError("No pitcher reconstruction overlap")
        m["share_raw"]=m["pitches_raw"]/m.groupby("player_id")["pitches_raw"].transform("sum")
        m["share_savant"]=m["pitches_savant"]/m.groupby("player_id")["pitches_savant"].transform("sum")
        return {"overlap_rows":int(len(m)),"weighted_mae_pitch_share":float(np.average(abs(m.share_raw-m.share_savant),weights=m.pitches_savant))}
    r=recon.rename(columns={"batter":"player_id"})
    r["est_woba_raw"]=r["est_woba_sum"]/r["pa"]
    if not {"player_id","pitch_type","est_woba"}.issubset(a): raise RuntimeError("Archived batter schema mismatch")
    m=r.merge(a[["player_id","pitch_type","est_woba"]],on=["player_id","pitch_type"])
    m["est_woba"]=pd.to_numeric(m["est_woba"],errors="coerce"); m=m[m["est_woba"].notna()]
    if m.empty: raise RuntimeError("No batter reconstruction overlap")
    return {"overlap_rows":int(len(m)),"weighted_mae_est_woba":float(np.average(abs(m.est_woba_raw-m.est_woba),weights=m.pa)),"mean_signed_error":float(np.average(m.est_woba_raw-m.est_woba,weights=m.pa))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache-dir",type=Path,required=True)
    p.add_argument("--start",required=True)
    p.add_argument("--cutoff",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--fetch-if-missing",action="store_true")
    p.add_argument("--archived-pitcher",type=Path)
    p.add_argument("--archived-batter",type=Path)
    a=p.parse_args()
    if a.fetch_if_missing: ensure_raw_cache(a.cache_dir,a.start,a.cutoff)
    raw=read_raw(a.cache_dir,a.start,a.cutoff)
    overall,side,batter,league=sufficient_stats(raw)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    overall.to_csv(a.output_dir/"pitcher_usage_overall.csv",index=False)
    side.to_csv(a.output_dir/"pitcher_usage_side_i2.csv",index=False)
    batter.to_csv(a.output_dir/"batter_pitch_response.csv",index=False)
    league.to_csv(a.output_dir/"league_pitch_response.csv",index=False)
    gate={"start":a.start,"cutoff_exclusive":a.cutoff,"raw_rows":int(len(raw)),"market_inputs_used":False}
    if a.archived_pitcher: gate["pitcher_reconstruction"]=compare_annual(overall,a.archived_pitcher,"pitcher")
    if a.archived_batter: gate["batter_reconstruction"]=compare_annual(batter,a.archived_batter,"batter")
    (a.output_dir/"manifest.json").write_text(json.dumps(gate,indent=2)+"\n")
    print(json.dumps(gate,indent=2))

if __name__=="__main__": main()

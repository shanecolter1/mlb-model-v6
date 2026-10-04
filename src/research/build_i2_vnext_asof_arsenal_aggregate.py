#!/usr/bin/env python3
"""Cutoff-safe arsenal reconstruction from Savant aggregate search."""
import argparse, io, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

URL="https://baseballsavant.mlb.com/statcast_search/csv"
PITCH_TYPES=["FF","SI","FC","CH","FS","FO","SC","CU","KC","CS","SL","ST","SV","KN","EP","FA"]

def request_one(role,pt,start,end,stand=None,inning=None):
    params={"all":"true","player_type":role,"hfGT":"R|","hfSea":f"{start[:4]}|","hfPT":f"{pt}|","game_date_gt":start,"game_date_lt":end,"group_by":"name","sort_col":"pitches","sort_order":"desc","min_pitches":"0","min_results":"0","min_pas":"0","chk_stats_pa":"on","chk_stats_xwoba":"on"}
    if stand in {"L","R"}:
        params["batter_stands"]=stand
    if inning is not None:
        params["hfInn"]=f"{int(inning)}|"
    headers={"User-Agent":"MLB-I2-vNext/1.0","Accept":"text/csv,*/*"}
    last=None
    for attempt in range(4):
        try:
            r=requests.get(URL,params=params,headers=headers,timeout=120); r.raise_for_status()
            x=pd.read_csv(io.StringIO(r.text))
            if len(x.columns)>1:return x
        except Exception as e:
            last=e; time.sleep(2*(attempt+1))
    raise RuntimeError(f"bad Savant response {role} {pt}: {last}")

def normalize(x,pt):
    cols={str(c).lower():c for c in x.columns}
    def pick(*names): return next((cols[n] for n in names if n in cols),None)
    idc=pick("player_id","playerid","id"); pc=pick("pitches","total_pitches"); pac=pick("pa","pas"); xc=pick("xwoba","est_woba","estimated_woba")
    if not all([idc,pc,xc]): raise RuntimeError(f"Unexpected aggregate schema: {x.columns.tolist()}")
    z=pd.DataFrame({"player_id":pd.to_numeric(x[idc],errors="coerce"),"pitch_type":pt,"pitches":pd.to_numeric(x[pc],errors="coerce"),"pa":pd.to_numeric(x[pac],errors="coerce") if pac else np.nan,"est_woba":pd.to_numeric(x[xc],errors="coerce")})
    return z[z.player_id.notna() & z.pitches.notna()].copy()

def build(role,start,end):
    out=[]
    for pt in PITCH_TYPES:
        z=normalize(request_one(role,pt,start,end),pt); out.append(z); print(role,pt,len(z),flush=True); time.sleep(.1)
    return pd.concat(out,ignore_index=True)

def build_side_usage(start,end):
    out=[]
    for stand in ("L","R"):
        for pt in PITCH_TYPES:
            z=normalize(request_one("pitcher",pt,start,end,stand=stand,inning=2),pt)
            q=z[["player_id","pitch_type","pitches"]].rename(columns={"player_id":"pitcher"}).copy()
            q["stand"]=stand; out.append(q[["pitcher","stand","pitch_type","pitches"]])
            print("side",stand,pt,len(q),flush=True); time.sleep(.1)
    return pd.concat(out,ignore_index=True)

def compare(recon,archived):
    a=pd.read_csv(archived,low_memory=False)
    m=recon.merge(a[["player_id","pitch_type","pitches","est_woba"]],on=["player_id","pitch_type"],suffixes=("_raw","_savant"))
    m=m[m.pitches_savant>0].copy()
    if m.empty: raise RuntimeError("No reconstruction overlap")
    m["share_raw"]=m.pitches_raw/m.groupby("player_id").pitches_raw.transform("sum")
    m["share_savant"]=m.pitches_savant/m.groupby("player_id").pitches_savant.transform("sum")
    z=m.dropna(subset=["est_woba_raw","est_woba_savant"]).copy()
    w=np.maximum(pd.to_numeric(z.pa,errors="coerce").fillna(1),1)
    return {"overlap_rows":int(len(m)),"xwoba_overlap_rows":int(len(z)),"weighted_mae_pitch_share":float(np.average(abs(m.share_raw-m.share_savant),weights=m.pitches_savant)),"weighted_mae_est_woba":float(np.average(abs(z.est_woba_raw-z.est_woba_savant),weights=w)),"mean_signed_est_woba":float(np.average(z.est_woba_raw-z.est_woba_savant,weights=w))}

def compare_side(recon,archived):
    a=pd.read_csv(archived,low_memory=False)
    m=recon.merge(a[["pitcher","stand","pitch_type","pitches"]],on=["pitcher","stand","pitch_type"],suffixes=("_raw","_savant"))
    m=m[m.pitches_savant>0].copy()
    if m.empty: raise RuntimeError("No side-usage reconstruction overlap")
    m["share_raw"]=m.pitches_raw/m.groupby(["pitcher","stand"]).pitches_raw.transform("sum")
    m["share_savant"]=m.pitches_savant/m.groupby(["pitcher","stand"]).pitches_savant.transform("sum")
    return {"overlap_rows":int(len(m)),"weighted_mae_pitch_share":float(np.average(abs(m.share_raw-m.share_savant),weights=m.pitches_savant))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--start",required=True);p.add_argument("--end",required=True);p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--archived-batter",type=Path);p.add_argument("--archived-pitcher",type=Path);p.add_argument("--archived-side",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    b=build("batter",a.start,a.end); q=build("pitcher",a.start,a.end); side=build_side_usage(a.start,a.end)
    b.to_csv(a.output_dir/"batter.csv",index=False);q.to_csv(a.output_dir/"pitcher.csv",index=False);side.to_csv(a.output_dir/"pitcher_usage_side_i2.csv",index=False)
    out={"start":a.start,"end_inclusive":a.end,"market_inputs_used":False,"method":"Statcast Search aggregate by player and pitch type; side usage additionally filters inning 2 and actual batter side"}
    if a.archived_batter:out["batter"]=compare(b,a.archived_batter)
    if a.archived_pitcher:out["pitcher"]=compare(q,a.archived_pitcher)
    if a.archived_side:out["side_usage"]=compare_side(side,a.archived_side)
    (a.output_dir/"manifest.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()

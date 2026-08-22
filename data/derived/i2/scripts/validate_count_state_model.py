#!/usr/bin/env python3
"""Walk-forward validation scaffold for count-state challenger."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def brier(p,y): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def logloss(p,y):
    p=np.clip(np.asarray(p),1e-9,1-1e-9); y=np.asarray(y)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--states",required=True,help="Pitch-state observation parquet with game_date/additional_runs")
    ap.add_argument("--out",default="data/derived/count_state/validation_report.json")
    args=ap.parse_args()
    x=pd.read_parquet(args.states)
    x["year"]=pd.to_datetime(x.game_date).dt.year
    folds=[([2021,2022],2023),([2021,2022,2023],2024),([2021,2022,2023,2024],2025)]
    rows=[]
    for train_years,test_year in folds:
        tr=x[x.year.isin(train_years)]
        te=x[x.year.eq(test_year)].copy()
        keys=["outs_when_up","base_mask","balls","strikes"]
        fit=(tr.assign(y=(tr.additional_runs>=1).astype(int))
             .groupby(keys).y.agg(["mean","size"]).reset_index()
             .rename(columns={"mean":"p1plus","size":"N"}))
        count=(tr.assign(y=(tr.additional_runs>=1).astype(int))
               .groupby(["balls","strikes"]).y.mean().rename("count_prior").reset_index())
        fit=fit.merge(count,on=["balls","strikes"])
        strength=250
        fit["p1plus_smoothed"]=(fit.p1plus*fit.N+fit.count_prior*strength)/(fit.N+strength)
        te=te.merge(fit[keys+["p1plus_smoothed"]],on=keys,how="left").merge(
            count,on=["balls","strikes"],how="left")
        te["p"]=te.p1plus_smoothed.fillna(te.count_prior)
        y=(te.additional_runs>=1).astype(int)
        rows.append({
            "train_years":train_years,"test_year":test_year,"N":int(len(te)),
            "brier_1plus":brier(te.p,y),"logloss_1plus":logloss(te.p,y)
        })
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"folds":rows},indent=2))
    print(json.dumps(rows,indent=2))
if __name__=="__main__": main()

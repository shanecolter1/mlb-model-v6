#!/usr/bin/env python3
"""Chronologically test handedness/platoon features beyond validated M1 core event rates.

Development only: 2021-2024. Candidate selection uses 2022-2023 folds only;
2024 is confirmation. 2025 is never loaded. This is PA-event feature-family
screening, not production promotion and not an inning-market model.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS=["k","baserunner","hr","nonhr_hit"]
WINDOWS=["season","30d","90d","365d"]
SPECS=["same_hand","handedness","handedness_rate_interactions"]
EPS=1e-12

def sigmoid(z):
    z=np.clip(z,-35,35); return 1/(1+np.exp(-z))
def logloss(y,p):
    p=np.clip(p,EPS,1-EPS); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def brier(y,p): return float(np.mean((p-y)**2))
def fit_logistic(X,y,max_iter=40,tol=1e-7):
    beta=np.zeros(X.shape[1],float)
    for _ in range(max_iter):
        p=sigmoid(X@beta); w=np.maximum(p*(1-p),1e-7)
        grad=X.T@(p-y); H=X.T@(X*w[:,None])
        try: step=np.linalg.solve(H,grad)
        except np.linalg.LinAlgError: step=np.linalg.pinv(H)@grad
        beta2=beta-step
        if np.max(np.abs(beta2-beta))<tol:
            beta=beta2; break
        beta=beta2
    return beta

def standardize(train,test,cols):
    tr=train.copy(); te=test.copy()
    for c in cols:
        mu=float(tr[c].mean()); sd=float(tr[c].std(ddof=0))
        if not np.isfinite(sd) or sd<1e-10: sd=1.0
        tr[c]=(tr[c]-mu)/sd; te[c]=(te[c]-mu)/sd
    return tr,te

def design(df,event,window,spec,challenger):
    b=df[f"batter_{window}_{event}_rate"].to_numpy(float)
    p=df[f"pitcher_{window}_{event}_rate"].to_numpy(float)
    inn=df.inning.astype(int).to_numpy()
    cols=[np.ones(len(df))]; names=["intercept"]
    for i in range(2,10): cols.append((inn==i).astype(float)); names.append(f"inning_{i}")
    cols += [b,p,b*p]; names += ["batter","pitcher","core_interaction"]
    if challenger:
        bl=(df.batter_side.astype(str).str.upper()=="L").astype(float).to_numpy()
        pl=(df.pitcher_hand.astype(str).str.upper()=="L").astype(float).to_numpy()
        same=((df.batter_side.astype(str).str.upper()==df.pitcher_hand.astype(str).str.upper()) & df.batter_side.astype(str).str.upper().isin(["L","R"]) & df.pitcher_hand.astype(str).str.upper().isin(["L","R"])).astype(float).to_numpy()
        if spec=="same_hand": cols += [same]; names += ["same_hand"]
        elif spec=="handedness": cols += [bl,pl,bl*pl]; names += ["batter_left","pitcher_left","left_interaction"]
        elif spec=="handedness_rate_interactions":
            cols += [bl,pl,bl*pl,b*bl,b*pl,p*bl,p*pl]
            names += ["batter_left","pitcher_left","left_interaction","b_x_bL","b_x_pL","p_x_bL","p_x_pL"]
        else: raise ValueError(spec)
    return np.column_stack(cols),names

def eval_fold(x,event,window,spec,year):
    ycol=f"y_{event}"; bcol=f"batter_{window}_{event}_rate"; pcol=f"pitcher_{window}_{event}_rate"
    needed=["season","inning","batter_side","pitcher_hand",ycol,bcol,pcol]
    tr0=x[x.season<year][needed].dropna().copy(); te0=x[x.season==year][needed].dropna().copy()
    tr0=tr0[tr0.batter_side.astype(str).str.upper().isin(["L","R"]) & tr0.pitcher_hand.astype(str).str.upper().isin(["L","R"])]
    te0=te0[te0.batter_side.astype(str).str.upper().isin(["L","R"]) & te0.pitcher_hand.astype(str).str.upper().isin(["L","R"])]
    if len(tr0)<1000 or len(te0)<1000: raise RuntimeError(f"insufficient cases {event} {window} {spec} {year}")
    tr,te=standardize(tr0,te0,[bcol,pcol]); ytr=tr[ycol].to_numpy(float); yte=te[ycol].to_numpy(float)
    X0,_=design(tr,event,window,spec,False); T0,_=design(te,event,window,spec,False)
    X1,names=design(tr,event,window,spec,True); T1,_=design(te,event,window,spec,True)
    b0=fit_logistic(X0,ytr); b1=fit_logistic(X1,ytr)
    p0=sigmoid(T0@b0); p1=sigmoid(T1@b1)
    return {
      "event":event,"window":window,"spec":spec,"test_year":year,"n_train":int(len(tr)),"n_test":int(len(te)),
      "core_logloss":logloss(yte,p0),"challenger_logloss":logloss(yte,p1),"incremental_logloss_improvement":logloss(yte,p0)-logloss(yte,p1),
      "core_brier":brier(yte,p0),"challenger_brier":brier(yte,p1),"incremental_brier_improvement":brier(yte,p0)-brier(yte,p1),
      "challenger_terms":"|".join(names[len(X0[0]):]) if X1.shape[1]>X0.shape[1] else ""
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--matrix",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.matrix); x["season"]=pd.to_numeric(x.season,errors="raise").astype(int); x["inning"]=pd.to_numeric(x.inning,errors="raise").astype(int)
    if set(x.season.unique())!={2021,2022,2023,2024}: raise RuntimeError("development seasons must be exactly 2021-2024")
    if (x.season>=2025).any(): raise RuntimeError("2025 holdout leakage")
    if set(x.inning.unique())!=set(range(1,10)): raise RuntimeError("I1-I9 coverage incomplete")
    if x.market_data_used.astype(bool).any(): raise RuntimeError("market data found")
    rows=[]
    for event in EVENTS:
        for window in WINDOWS:
            for spec in SPECS:
                for year in [2022,2023,2024]: rows.append(eval_fold(x,event,window,spec,year))
    folds=pd.DataFrame(rows)
    dev=(folds[folds.test_year.isin([2022,2023])].groupby(["event","window","spec"],as_index=False).agg(
        dev_mean_logloss_improvement=("incremental_logloss_improvement","mean"),dev_worst_logloss_improvement=("incremental_logloss_improvement","min"),
        dev_mean_brier_improvement=("incremental_brier_improvement","mean"),dev_worst_brier_improvement=("incremental_brier_improvement","min")))
    # select on development only; 2024 is not used to choose window/spec
    selected=(dev.sort_values(["event","dev_mean_logloss_improvement","dev_mean_brier_improvement"],ascending=[True,False,False]).groupby("event",as_index=False).head(1))
    conf=folds[folds.test_year==2024].merge(selected[["event","window","spec"]],on=["event","window","spec"],how="inner")
    conf["confirmed_logloss_positive"]=conf.incremental_logloss_improvement>0
    conf["confirmed_brier_positive"]=conf.incremental_brier_improvement>0
    folds.to_csv(a.output_dir/"m1_platoon_incremental_folds.csv",index=False)
    dev.to_csv(a.output_dir/"m1_platoon_incremental_development_summary.csv",index=False)
    selected.to_csv(a.output_dir/"m1_platoon_selected_by_event.csv",index=False)
    conf.to_csv(a.output_dir/"m1_platoon_confirmation_2024.csv",index=False)
    manifest={
      "status":"PASS","architecture":"M1_platoon_incremental_feature_family_screen","development_seasons":[2021,2022,2023,2024],
      "selection_folds":[2022,2023],"confirmation_year":2024,"holdout_season":2025,"holdout_opened":False,"market_data_used":False,
      "core":"inning effects + batter rate + pitcher rate + batter*pitcher interaction for the same PA event",
      "candidate_family":"handedness/platoon indicators and interactions","windows":WINDOWS,"specs":SPECS,
      "automatic_production_promotion":False,"confirmation":conf.to_dict("records")
    }
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()

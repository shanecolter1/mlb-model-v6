#!/usr/bin/env python3
"""Chronologically validate PA-level matchup skill dimensions for I1-I9.

Development only: 2021-2024. 2025 is never loaded.
Purpose: determine whether strictly prior-date raw batter/pitcher event rates add
out-of-sample predictive information for the corresponding PA event. This is a
skill-signal validation layer, not yet the inning run challenger.

For each event dimension and lookback window, compare an inning-aware baseline
against batter-only, pitcher-only, additive, and additive+interaction logistic
models. These low-dimensional signal-discovery fits are unpenalized; no
regularization strength is promoted or tuned here. No production promotion is
made by this script.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS=["season","30d","90d","365d"]
EVENTS=["k","baserunner","hr","nonhr_hit"]
SPECS=["batter_only","pitcher_only","additive","additive_interaction"]
TEST_YEARS=[2022,2023,2024]
EPS=1e-12

def sigmoid(z):
    z=np.clip(z,-35,35); return 1/(1+np.exp(-z))
def logloss(y,p):
    p=np.clip(p,EPS,1-EPS); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def brier(y,p): return float(np.mean((p-y)**2))

def fit_logistic(X,y,max_iter=35,tol=1e-7):
    beta=np.zeros(X.shape[1],dtype=float)
    for _ in range(max_iter):
        p=sigmoid(X@beta); w=np.maximum(p*(1-p),1e-7)
        grad=X.T@(p-y); H=X.T@(X*w[:,None])
        try: step=np.linalg.solve(H,grad)
        except np.linalg.LinAlgError: step=np.linalg.pinv(H)@grad
        beta_new=beta-step
        if np.max(np.abs(beta_new-beta))<tol:
            beta=beta_new; break
        beta=beta_new
    return beta

def standardize(train,test,cols):
    tr=train.copy(); te=test.copy()
    for c in cols:
        mu=float(tr[c].mean()); sd=float(tr[c].std(ddof=0))
        if not np.isfinite(sd) or sd<1e-10: sd=1.0
        tr[c]=(tr[c]-mu)/sd; te[c]=(te[c]-mu)/sd
    return tr,te

def design(df,spec,bcol,pcol):
    n=len(df); cols=[np.ones(n)]; names=["intercept"]
    inn=pd.to_numeric(df.inning,errors="coerce").astype(int).to_numpy()
    for i in range(2,10): cols.append((inn==i).astype(float)); names.append(f"inning_{i}")
    b=df[bcol].to_numpy(float); p=df[pcol].to_numpy(float)
    if spec in ("batter_only","additive","additive_interaction"): cols.append(b); names.append("batter")
    if spec in ("pitcher_only","additive","additive_interaction"): cols.append(p); names.append("pitcher")
    if spec=="additive_interaction": cols.append(b*p); names.append("interaction")
    return np.column_stack(cols),names

def baseline_probs(train,test,ycol):
    global_rate=float(train[ycol].mean()); by=train.groupby("inning")[ycol].mean().to_dict()
    return np.array([float(by.get(int(i),global_rate)) for i in test.inning],dtype=float)

def run_candidate(x,event,window,spec):
    bcol=f"batter_{window}_{event}_rate"; pcol=f"pitcher_{window}_{event}_rate"; ycol=f"y_{event}"
    folds=[]; coefs=[]
    for year in TEST_YEARS:
        cols=["season","inning",ycol,bcol,pcol]
        tr0=x[x.season<year][cols].dropna().copy(); te0=x[x.season==year][cols].dropna().copy()
        if len(tr0)<1000 or len(te0)<1000: raise RuntimeError(f"insufficient complete cases {event} {window} {year}")
        tr,te=standardize(tr0,te0,[bcol,pcol])
        Xtr,names=design(tr,spec,bcol,pcol); Xte,_=design(te,spec,bcol,pcol)
        ytr=tr[ycol].to_numpy(float); yte=te[ycol].to_numpy(float)
        beta=fit_logistic(Xtr,ytr); pred=sigmoid(Xte@beta); p0=baseline_probs(tr0,te0,ycol)
        row={"event":event,"window":window,"spec":spec,"test_year":year,"n_train":int(len(tr)),"n_test":int(len(te)),
             "baseline_logloss":logloss(yte,p0),"model_logloss":logloss(yte,pred),"baseline_brier":brier(yte,p0),"model_brier":brier(yte,pred)}
        row["logloss_improvement"]=row["baseline_logloss"]-row["model_logloss"]
        row["brier_improvement"]=row["baseline_brier"]-row["model_brier"]; folds.append(row)
        for name,val in zip(names,beta):
            if name in {"batter","pitcher","interaction"}: coefs.append({"event":event,"window":window,"spec":spec,"test_year":year,"coefficient":name,"value":float(val)})
    return folds,coefs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--matrix",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.matrix)
    if set(pd.to_numeric(x.season,errors="coerce").dropna().astype(int).unique())!={2021,2022,2023,2024}: raise RuntimeError("development seasons must be 2021-2024")
    if (pd.to_numeric(x.season,errors="coerce")>=2025).any(): raise RuntimeError("2025 holdout leakage")
    if set(pd.to_numeric(x.inning,errors="coerce").dropna().astype(int).unique())!=set(range(1,10)): raise RuntimeError("I1-I9 coverage incomplete")
    if x.market_data_used.astype(bool).any(): raise RuntimeError("market data found")

    fold_rows=[]; coef_rows=[]
    for event in EVENTS:
        for window in WINDOWS:
            for spec in SPECS:
                fr,cr=run_candidate(x,event,window,spec); fold_rows.extend(fr); coef_rows.extend(cr)
    folds=pd.DataFrame(fold_rows); coefs=pd.DataFrame(coef_rows)
    summary=(folds.groupby(["event","window","spec"],as_index=False).agg(
        mean_logloss_improvement=("logloss_improvement","mean"),worst_year_logloss_improvement=("logloss_improvement","min"),
        mean_brier_improvement=("brier_improvement","mean"),worst_year_brier_improvement=("brier_improvement","min")))
    summary["all_years_logloss_positive"]=summary.worst_year_logloss_improvement>0
    summary["all_years_brier_positive"]=summary.worst_year_brier_improvement>0
    best=(summary.sort_values(["event","mean_logloss_improvement"],ascending=[True,False]).groupby("event",as_index=False).head(1).sort_values("event"))
    coverage=[]
    for event in EVENTS:
        for window in WINDOWS:
            bc=f"batter_{window}_{event}_rate"; pc=f"pitcher_{window}_{event}_rate"
            for inning,g in x.groupby("inning"):
                coverage.append({"event":event,"window":window,"inning":int(inning),"n":int(len(g)),"complete_case_rate":float(g[[bc,pc]].notna().all(axis=1).mean())})
    pd.DataFrame(coverage).to_csv(a.output_dir/"m1_pa_skill_coverage_by_inning.csv",index=False)
    folds.to_csv(a.output_dir/"m1_pa_skill_fold_results.csv",index=False); summary.to_csv(a.output_dir/"m1_pa_skill_candidate_summary.csv",index=False)
    best.to_csv(a.output_dir/"m1_pa_skill_best_by_event.csv",index=False); coefs.to_csv(a.output_dir/"m1_pa_skill_coefficients.csv",index=False)
    manifest={"status":"PASS","architecture":"M1_PA_skill_chronological_development_validation","development_seasons":[2021,2022,2023,2024],
      "test_folds":TEST_YEARS,"holdout_season":2025,"holdout_opened":False,"events":EVENTS,"windows":WINDOWS,"specs":SPECS,
      "regularization_used":False,"regularization_tuned":False,"baseline":"training-only empirical PA event rate by inning",
      "statistics_timing":"strictly_prior_date","participant_identity_use":"retrospective matchup oracle for skill validation only",
      "market_data_used":False,"automatic_production_promotion":False,"best_development_candidate_by_event":best.to_dict("records"),
      "note":"PA-level signal validation only. No inning-run residual promotion versus M0 and no 2025 access."}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()

#!/usr/bin/env python3
"""Diagnose the integrated M4 residual with inning-specific coefficients.

The first integrated challenger used one common coefficient vector across I1-I9
and failed against M0. The governing architecture allows a common feature family
with inning-specific weights when empirically supported. This development audit
therefore keeps the exact same integrated features and M0 probabilities, changes
no state construction, and tests whether separate inning coefficient vectors are
required.

Ridge strength is selected independently for each inning using 2023 only. The
selected specification is then frozen and evaluated on 2024. 2025 is never
loaded. No new feature, window, market variable, or smoothing assumption enters.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RIDGES=[0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0]
EPS=1e-8


def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))

def sigmoid(z):
    z=np.clip(np.asarray(z,float),-35,35)
    return 1/(1+np.exp(-z))

def logloss(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())

def brier(y,p):
    return float(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))

def fit_offset_ridge(X,y,offset,ridge):
    beta=np.zeros(X.shape[1],float); eye=np.eye(X.shape[1])
    for _ in range(60):
        p=sigmoid(offset+X@beta); w=np.maximum(p*(1-p),1e-7)
        grad=X.T@(y-p)-ridge*beta
        H=(X.T*w)@X+ridge*eye+1e-9*eye
        step=np.linalg.solve(H,grad); b2=beta+step
        if np.max(np.abs(step))<1e-8:
            beta=b2; break
        beta=b2
    return beta

def prepare(train,test,features):
    needed=features+["m0_p_any"]
    tr=train.dropna(subset=needed).copy(); te=test.dropna(subset=needed).copy()
    mu=tr[features].mean().to_numpy(float).copy()
    sd=tr[features].std(ddof=0).to_numpy(float).copy(); sd[sd<1e-9]=1.0
    Xtr=(tr[features].to_numpy(float)-mu)/sd
    Xte=(te[features].to_numpy(float)-mu)/sd
    return tr,te,Xtr,Xte,mu,sd

def evaluate_one(train,test,features,ridge):
    tr,te,Xtr,Xte,mu,sd=prepare(train,test,features)
    ytr=tr.any_run.to_numpy(float); yte=te.any_run.to_numpy(float)
    otr=logit(tr.m0_p_any); ote=logit(te.m0_p_any)
    beta=fit_offset_ridge(Xtr,ytr,otr,ridge)
    p0=sigmoid(ote); p=sigmoid(ote+Xte@beta)
    return {
        "n_train":len(tr),"n_test":len(te),"ridge":ridge,
        "m0_logloss":logloss(yte,p0),"matchup_logloss":logloss(yte,p),
        "logloss_improvement":logloss(yte,p0)-logloss(yte,p),
        "m0_brier":brier(yte,p0),"matchup_brier":brier(yte,p),
        "brier_improvement":brier(yte,p0)-brier(yte,p),
        "beta":beta,"mu":mu,"sd":sd,"y":yte,"p0":p0,"p":p,
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--integrated-full',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.integrated_full)
    x['season']=pd.to_numeric(x.season).astype(int); x['inning']=pd.to_numeric(x.inning).astype(int)
    if set(x.season.unique())!={2022,2023,2024}: raise RuntimeError(f'unexpected seasons {sorted(x.season.unique())}')
    if (x.season>=2025).any(): raise RuntimeError('2025 leakage')
    features=[c for c in x.columns if c.startswith('expected_')]

    selection=[]; confirm=[]; coeff=[]; all_y=[]; all_p0=[]; all_p=[]
    for inn in range(1,10):
        train22=x[(x.season==2022)&(x.inning==inn)]
        tune23=x[(x.season==2023)&(x.inning==inn)]
        candidates=[]
        for ridge in RIDGES:
            r=evaluate_one(train22,tune23,features,ridge)
            candidates.append({"inning":inn,"selection_year":2023,"ridge":ridge,
                               "n_train":r['n_train'],"n_test":r['n_test'],
                               "logloss_improvement":r['logloss_improvement'],"brier_improvement":r['brier_improvement']})
        c=pd.DataFrame(candidates)
        # Primary selection = log loss; Brier is secondary tie-break.
        best=c.sort_values(['logloss_improvement','brier_improvement'],ascending=False).iloc[0]
        selection.extend(c.to_dict('records'))
        ridge=float(best.ridge)

        train232=x[(x.season<=2023)&(x.season>=2022)&(x.inning==inn)]
        test24=x[(x.season==2024)&(x.inning==inn)]
        r=evaluate_one(train232,test24,features,ridge)
        confirm.append({"inning":inn,"selected_ridge_2023":ridge,"n_train_2024":r['n_train'],"n_test_2024":r['n_test'],
                        "m0_logloss_2024":r['m0_logloss'],"matchup_logloss_2024":r['matchup_logloss'],
                        "logloss_improvement_2024":r['logloss_improvement'],"m0_brier_2024":r['m0_brier'],
                        "matchup_brier_2024":r['matchup_brier'],"brier_improvement_2024":r['brier_improvement'],
                        "confirm_logloss_positive":r['logloss_improvement']>0,"confirm_brier_positive":r['brier_improvement']>0})
        for f,b in zip(features,r['beta']): coeff.append({"inning":inn,"selected_ridge":ridge,"feature":f,"standardized_coefficient_2022_2023_fit":float(b)})
        all_y.append(r['y']); all_p0.append(r['p0']); all_p.append(r['p'])

    sel=pd.DataFrame(selection); con=pd.DataFrame(confirm); co=pd.DataFrame(coeff)
    y=np.concatenate(all_y); p0=np.concatenate(all_p0); p=np.concatenate(all_p)
    aggregate={
        "n_test_2024":int(len(y)),"m0_logloss_2024":logloss(y,p0),"matchup_logloss_2024":logloss(y,p),
        "logloss_improvement_2024":logloss(y,p0)-logloss(y,p),"m0_brier_2024":brier(y,p0),
        "matchup_brier_2024":brier(y,p),"brier_improvement_2024":brier(y,p0)-brier(y,p),
        "innings_logloss_positive_2024":int(con.confirm_logloss_positive.sum()),
        "innings_brier_positive_2024":int(con.confirm_brier_positive.sum()),
    }
    sel.to_csv(a.output_dir/'m4_inning_specific_ridge_selection_2023.csv',index=False)
    con.to_csv(a.output_dir/'m4_inning_specific_confirmation_2024.csv',index=False)
    co.to_csv(a.output_dir/'m4_inning_specific_coefficients_2024_fit.csv',index=False)
    manifest={
        "status":"PASS","architecture":"M4_integrated_inning_specific_residual_diagnostic",
        "source_integrated_architecture":"same M4 integrated features; no state reconstruction changes",
        "development_seasons_loaded":[2022,2023,2024],"ridge_selection_year":2023,"confirmation_year":2024,
        "holdout_season":2025,"holdout_opened":False,"feature_family":features,"ridge_candidates":RIDGES,
        "coefficient_structure":"separate coefficient vector by inning; common feature family",
        "selection_rule":"2023 max logloss improvement, Brier secondary tie-break; frozen for 2024",
        "market_data_used":False,"only_market_context":"M0 opening-total x inning probability already present in integrated matrix",
        "confirmation_by_inning":con.to_dict('records'),"aggregate_2024":aggregate,
        "automatic_production_promotion":False,
        "note":"Diagnostic of coefficient pooling only. Positive 2024 confirmation supports inning-specific weighting; mixed results require further partial-pooling/feature-construction work before discrete run modeling."
    }
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

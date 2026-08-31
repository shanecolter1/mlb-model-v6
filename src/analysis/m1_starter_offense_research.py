#!/usr/bin/env python3
"""M1: starter + opposing offense residual research.

Governance:
- M0 is the already-validated empirical pregame-total x inning relationship.
- For chronological development, the same raw exact-total estimator is rebuilt
  using TRAINING SEASONS ONLY. No beta-binomial smoothing is introduced.
- 2021-2024 are development. 2025 is never used for feature/hyperparameter
  selection and is evaluated only after the winning development specification
  is frozen.
- M1 contains starter/offense matchup information only. No batting-order,
  bullpen, park, weather, umpire, or betting-price variables may enter.
- Every ridge strength / feature family must earn inclusion by improving
  chronological development log loss versus M0; Brier and calibration are
  secondary diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EPS=1e-9
RIDGE_GRID=(0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0)


def logistic(x):
    return 1/(1+np.exp(-np.clip(np.asarray(x,float),-40,40)))

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))

def log_loss(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))

def calibration(y,p):
    x=logit(p); X=np.column_stack([np.ones(len(x)),x]); beta=np.zeros(2)
    for _ in range(60):
        mu=logistic(X@beta); w=np.clip(mu*(1-mu),1e-6,None)
        z=X@beta+(np.asarray(y)-mu)/w
        A=X.T@(w[:,None]*X)+np.eye(2)*1e-8
        nb=np.linalg.solve(A,X.T@(w*z))
        if np.max(np.abs(nb-beta))<1e-9: beta=nb; break
        beta=nb
    return float(beta[0]),float(beta[1])


def fit_raw_total_prior(train,total_col,y_col):
    broad=float(train[y_col].mean())
    g=train.groupby(total_col)[y_col].agg(['sum','count'])
    priors={float(i):(float(r['sum']/r['count']),int(r['count'])) for i,r in g.iterrows() if r['count']>0}
    return broad,priors

def baseline(df,total_col,broad,priors):
    keys=np.array(sorted(priors),float)
    ps=[]; ns=[]; used=[]
    for t in pd.to_numeric(df[total_col],errors='coerce'):
        if float(t) in priors: k=float(t)
        else: k=float(keys[np.argmin(np.abs(keys-float(t)))])
        p,n=priors[k]; ps.append(p); ns.append(n); used.append(k)
    return np.asarray(ps),np.asarray(ns),used


def standardize(train,test,features):
    A=train[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    B=test[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    mean=np.nanmean(A,axis=0); sd=np.nanstd(A,axis=0)
    sd=np.where((~np.isfinite(sd))|(sd<1e-9),1.0,sd)
    return np.nan_to_num((A-mean)/sd),np.nan_to_num((B-mean)/sd)

def fit_offset_ridge(X,y,offset,lam):
    beta=np.zeros(X.shape[1]); R=np.eye(X.shape[1])*float(lam)
    for _ in range(100):
        mu=logistic(offset+X@beta); w=np.clip(mu*(1-mu),1e-6,None)
        grad=X.T@(y-mu)-R@beta
        H=X.T@(w[:,None]*X)+R+np.eye(X.shape[1])*1e-10
        step=np.linalg.solve(H,grad); nb=beta+step
        if np.max(np.abs(nb-beta))<1e-8: beta=nb; break
        beta=nb
    return beta


def dev_folds(df):
    # 2025 deliberately excluded from all selection.
    for test_year in (2022,2023,2024):
        tr=df[(df.season>=2021)&(df.season<test_year)].copy()
        te=df[df.season==test_year].copy()
        if len(tr) and len(te): yield test_year,tr,te


def evaluate_spec(df,features,total_col,y_col,lam):
    rows=[]; coefs=[]
    for year,tr,te in dev_folds(df):
        broad,priors=fit_raw_total_prior(tr,total_col,y_col)
        p0_tr,_,_=baseline(tr,total_col,broad,priors)
        p0_te,_,_=baseline(te,total_col,broad,priors)
        Xtr,Xte=standardize(tr,te,features)
        beta=fit_offset_ridge(Xtr,tr[y_col].to_numpy(float),logit(p0_tr),lam)
        p1=logistic(logit(p0_te)+Xte@beta)
        y=te[y_col].to_numpy(float); ci,cs=calibration(y,p1)
        rows.append({'test_season':year,'n':len(te),'ridge_lambda':lam,
                     'm0_log_loss':log_loss(y,p0_te),'m1_log_loss':log_loss(y,p1),
                     'log_loss_improvement':log_loss(y,p0_te)-log_loss(y,p1),
                     'm0_brier':brier(y,p0_te),'m1_brier':brier(y,p1),
                     'brier_improvement':brier(y,p0_te)-brier(y,p1),
                     'calibration_intercept':ci,'calibration_slope':cs})
        for f,b in zip(features,beta): coefs.append({'test_season':year,'ridge_lambda':lam,'feature':f,'standardized_beta':float(b)})
    return pd.DataFrame(rows),pd.DataFrame(coefs)


def final_holdout(df,features,total_col,y_col,lam):
    tr=df[(df.season>=2021)&(df.season<=2024)].copy(); te=df[df.season==2025].copy()
    broad,priors=fit_raw_total_prior(tr,total_col,y_col)
    p0tr,_,_=baseline(tr,total_col,broad,priors); p0,prior_n,bucket=baseline(te,total_col,broad,priors)
    Xtr,Xte=standardize(tr,te,features)
    beta=fit_offset_ridge(Xtr,tr[y_col].to_numpy(float),logit(p0tr),lam)
    p1=logistic(logit(p0)+Xte@beta); y=te[y_col].to_numpy(float); ci,cs=calibration(y,p1)
    metrics={'test_season':2025,'n':int(len(te)),'ridge_lambda':float(lam),
             'm0_log_loss':log_loss(y,p0),'m1_log_loss':log_loss(y,p1),
             'log_loss_improvement':log_loss(y,p0)-log_loss(y,p1),
             'm0_brier':brier(y,p0),'m1_brier':brier(y,p1),
             'brier_improvement':brier(y,p0)-brier(y,p1),
             'calibration_intercept':ci,'calibration_slope':cs}
    pred=te[['game_date','season',total_col,y_col]].copy(); pred['m0_probability']=p0; pred['m1_probability']=p1
    pred['matchup_delta']=p1-p0; pred['prior_n']=prior_n; pred['prior_bucket_used']=bucket
    coef=pd.DataFrame({'feature':features,'standardized_beta':beta})
    return metrics,pred,coef


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True,help='One row/game leakage-safe M1 matrix')
    ap.add_argument('--registry',default='config/m1_starter_offense_feature_registry.csv')
    ap.add_argument('--output-dir',default='data/derived/i2/m1_starter_offense')
    ap.add_argument('--total-col',default='dk_total_open_total')
    ap.add_argument('--i2-runs-col',default='inning2_total_runs')
    ap.add_argument('--run-2025-holdout',action='store_true',help='Only after development specification is frozen')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(a.input) if str(a.input).endswith('.parquet') else pd.read_csv(a.input)
    df['game_date']=pd.to_datetime(df['game_date'],errors='coerce'); df['season']=pd.to_numeric(df['season'],errors='coerce')
    df[a.total_col]=pd.to_numeric(df[a.total_col],errors='coerce'); df[a.i2_runs_col]=pd.to_numeric(df[a.i2_runs_col],errors='coerce')
    df=df[df[a.total_col].notna()&df[a.i2_runs_col].notna()&df.season.notna()].copy(); df['season']=df.season.astype(int)
    df['i2_over_05']=(df[a.i2_runs_col]>=1).astype(float)
    reg=pd.read_csv(a.registry); features=[f for f in reg.loc[reg.status=='candidate','feature'] if f in df.columns]
    missing=[f for f in reg.loc[reg.status=='candidate','feature'] if f not in df.columns]
    if not features: raise RuntimeError('No registered M1 features are present in input matrix')

    grids=[]; coef_all=[]
    for lam in RIDGE_GRID:
        r,c=evaluate_spec(df,features,a.total_col,'i2_over_05',lam); grids.append(r); coef_all.append(c)
    grid=pd.concat(grids,ignore_index=True); coefs=pd.concat(coef_all,ignore_index=True)
    summary=(grid.groupby('ridge_lambda',as_index=False).agg(mean_log_loss_improvement=('log_loss_improvement','mean'),
             worst_year_log_loss_improvement=('log_loss_improvement','min'),mean_brier_improvement=('brier_improvement','mean')))
    # Primary selection: best mean dev log loss among specifications that do not lose in any dev season.
    eligible=summary[summary.worst_year_log_loss_improvement>=0].copy()
    selected=float((eligible if len(eligible) else summary).sort_values(['mean_log_loss_improvement','mean_brier_improvement'],ascending=False).iloc[0].ridge_lambda)
    grid.to_csv(out/'development_folds.csv',index=False); summary.to_csv(out/'ridge_grid_summary.csv',index=False); coefs.to_csv(out/'development_coefficients.csv',index=False)
    manifest={'status':'DEVELOPMENT_COMPLETE','m0':'raw exact pregame-total x I2 empirical estimator; training seasons only; no smoothing',
              'development_seasons':[2021,2022,2023,2024],'selection_test_seasons':[2022,2023,2024],
              'holdout_season':2025,'holdout_opened':bool(a.run_2025_holdout),'features':features,'missing_registered_features':missing,
              'selected_ridge_lambda':selected,'ridge_grid':list(RIDGE_GRID),'market_derivative_data_used':False}
    if a.run_2025_holdout:
        metrics,pred,coef=final_holdout(df,features,a.total_col,'i2_over_05',selected)
        pd.DataFrame([metrics]).to_csv(out/'final_2025_holdout_metrics.csv',index=False); pred.to_parquet(out/'final_2025_holdout_predictions.parquet',index=False); coef.to_csv(out/'final_2025_coefficients.csv',index=False)
        manifest['final_2025_metrics']=metrics; manifest['status']='FINAL_HOLDOUT_COMPLETE'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""M2 development-only validation: starter retention + bullpen state.

M1 remains fixed at the empirically selected development architecture:
- raw exact pregame-total x I2 M0 baseline
- M1 starter/offense residual with ridge lambda 50

M2 is fit only as an additional residual offset on top of M1. Candidate feature
families/windows and M2 ridge strength are selected only from chronological
2021-2024 development folds. 2025 is never evaluated here.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

EPS=1e-9
M1_LAMBDA=50.0
M2_RIDGE_GRID=(0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0)
M1_FEATURES=[
 'starter_k_rate','starter_bb_rate','starter_hr_rate','starter_nonhr_hit_rate',
 'opponent_k_rate','opponent_bb_rate','opponent_hr_rate','opponent_nonhr_hit_rate',
 'contact_interaction','power_interaction','baserunner_interaction'
]
RETENTION_WINDOWS=(None,30,90,365)
QUALITY_WINDOWS=(None,30,90,365)
WORKLOAD_SETS={
 'none':[],
 'short':['bullpen_bf_1d','bullpen_bf_2d','bullpen_bf_3d'],
 'all':['bullpen_bf_1d','bullpen_bf_2d','bullpen_bf_3d','bullpen_bf_7d','bullpen_bf_14d'],
}


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
    g=train.groupby(total_col)[y_col].agg(['sum','count'])
    return {float(i):(float(r['sum']/r['count']),int(r['count'])) for i,r in g.iterrows() if r['count']>0}

def baseline(df,total_col,priors):
    keys=np.array(sorted(priors),float); ps=[]
    for t in pd.to_numeric(df[total_col],errors='coerce'):
        k=float(t) if float(t) in priors else float(keys[np.argmin(np.abs(keys-float(t)))])
        ps.append(priors[k][0])
    return np.asarray(ps)

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

def spec_features(retention_window,quality_window,workload_name):
    f=[]
    if retention_window is not None:
        f += [f'starter_i2_reached_rate_{retention_window}d',f'starter_i2_share_{retention_window}d']
    if quality_window is not None:
        f += [f'bullpen_k_rate_{quality_window}d',f'bullpen_bb_rate_{quality_window}d',
              f'bullpen_hr_rate_{quality_window}d',f'bullpen_nonhr_hit_rate_{quality_window}d']
    f += WORKLOAD_SETS[workload_name]
    return f

def development_folds(df):
    for test_year in (2022,2023,2024):
        tr=df[(df.season>=2021)&(df.season<test_year)].copy()
        te=df[df.season==test_year].copy()
        if len(tr) and len(te): yield test_year,tr,te

def fit_m1(tr,te,total_col,y_col):
    priors=fit_raw_total_prior(tr,total_col,y_col)
    p0tr=baseline(tr,total_col,priors); p0te=baseline(te,total_col,priors)
    Xtr,Xte=standardize(tr,te,M1_FEATURES)
    b1=fit_offset_ridge(Xtr,tr[y_col].to_numpy(float),logit(p0tr),M1_LAMBDA)
    p1tr=logistic(logit(p0tr)+Xtr@b1); p1te=logistic(logit(p0te)+Xte@b1)
    return p0te,p1tr,p1te,b1

def evaluate_spec(df,features,total_col,y_col,lam,spec_id,retention_window,quality_window,workload_name):
    rows=[]; coefs=[]
    for year,tr,te in development_folds(df):
        p0,p1tr,p1,b1=fit_m1(tr,te,total_col,y_col)
        Xtr,Xte=standardize(tr,te,features)
        b2=fit_offset_ridge(Xtr,tr[y_col].to_numpy(float),logit(p1tr),lam)
        p2=logistic(logit(p1)+Xte@b2)
        y=te[y_col].to_numpy(float); ci,cs=calibration(y,p2)
        rows.append({
          'spec_id':spec_id,'retention_window':retention_window if retention_window is not None else 'none',
          'quality_window':quality_window if quality_window is not None else 'none','workload_set':workload_name,
          'test_season':year,'n':len(te),'m2_ridge_lambda':lam,
          'm0_log_loss':log_loss(y,p0),'m1_log_loss':log_loss(y,p1),'m2_log_loss':log_loss(y,p2),
          'm1_vs_m0_log_loss_improvement':log_loss(y,p0)-log_loss(y,p1),
          'm2_vs_m1_log_loss_improvement':log_loss(y,p1)-log_loss(y,p2),
          'm0_brier':brier(y,p0),'m1_brier':brier(y,p1),'m2_brier':brier(y,p2),
          'm2_vs_m1_brier_improvement':brier(y,p1)-brier(y,p2),
          'calibration_intercept':ci,'calibration_slope':cs,
          'feature_count':len(features)
        })
        for f,b in zip(features,b2):
            coefs.append({'spec_id':spec_id,'test_season':year,'m2_ridge_lambda':lam,'feature':f,'standardized_beta':float(b)})
    return pd.DataFrame(rows),pd.DataFrame(coefs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--output-dir',default='data/derived/i2/m2_starter_bullpen')
    ap.add_argument('--total-col',default='dk_total_open_total')
    ap.add_argument('--i2-runs-col',default='inning2_total_runs')
    a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(a.input) if str(a.input).endswith('.parquet') else pd.read_csv(a.input)
    df['game_date']=pd.to_datetime(df.game_date,errors='coerce'); df['season']=pd.to_numeric(df.season,errors='coerce')
    df[a.total_col]=pd.to_numeric(df[a.total_col],errors='coerce'); df[a.i2_runs_col]=pd.to_numeric(df[a.i2_runs_col],errors='coerce')
    df=df[df.season.notna()&df[a.total_col].notna()&df[a.i2_runs_col].notna()].copy(); df['season']=df.season.astype(int)
    df['i2_over_05']=(df[a.i2_runs_col]>=1).astype(float)
    if any(f not in df.columns for f in M1_FEATURES): raise RuntimeError('M1 features missing from M2 matrix')

    specs=[]
    for rw,qw,wl in itertools.product(RETENTION_WINDOWS,QUALITY_WINDOWS,WORKLOAD_SETS):
        f=spec_features(rw,qw,wl)
        if not f: continue
        missing=[c for c in f if c not in df.columns]
        if missing: raise RuntimeError(f'M2 candidate columns missing: {missing}')
        sid=f"ret_{rw or 'none'}__qual_{qw or 'none'}__work_{wl}"
        specs.append((sid,rw,qw,wl,f))

    folds=[]; coef=[]
    for sid,rw,qw,wl,f in specs:
        for lam in M2_RIDGE_GRID:
            r,c=evaluate_spec(df,f,a.total_col,'i2_over_05',lam,sid,rw,qw,wl)
            folds.append(r); coef.append(c)
    folds=pd.concat(folds,ignore_index=True); coef=pd.concat(coef,ignore_index=True)
    summary=(folds.groupby(['spec_id','retention_window','quality_window','workload_set','m2_ridge_lambda','feature_count'],as_index=False)
             .agg(mean_m2_vs_m1_log_loss_improvement=('m2_vs_m1_log_loss_improvement','mean'),
                  worst_year_m2_vs_m1_log_loss_improvement=('m2_vs_m1_log_loss_improvement','min'),
                  mean_m2_vs_m1_brier_improvement=('m2_vs_m1_brier_improvement','mean')))
    summary['no_negative_log_loss_year']=(summary.worst_year_m2_vs_m1_log_loss_improvement>=0)
    summary['positive_mean_log_loss']=(summary.mean_m2_vs_m1_log_loss_improvement>0)
    eligible=summary[summary.no_negative_log_loss_year & summary.positive_mean_log_loss].copy()
    ranked=(eligible if len(eligible) else summary).sort_values(
        ['mean_m2_vs_m1_log_loss_improvement','mean_m2_vs_m1_brier_improvement'],ascending=False)
    best=ranked.iloc[0].to_dict()
    promoted=bool(best['mean_m2_vs_m1_log_loss_improvement']>0 and best['worst_year_m2_vs_m1_log_loss_improvement']>=0)

    folds.to_csv(out/'development_folds.csv',index=False)
    summary.to_csv(out/'spec_grid_summary.csv',index=False)
    coef.to_csv(out/'development_coefficients.csv',index=False)
    manifest={
      'status':'M2_PROMOTED_DEVELOPMENT' if promoted else 'M2_REJECTED_DEVELOPMENT',
      'comparison':'M2 incremental residual versus fixed M1 champion',
      'm0':'raw exact pregame-total x I2 empirical estimator; training seasons only; no smoothing',
      'm1_ridge_lambda_fixed':M1_LAMBDA,
      'development_seasons':[2021,2022,2023,2024],
      'selection_test_seasons':[2022,2023,2024],
      'holdout_season':2025,'holdout_opened':False,
      'candidate_retention_windows_days':[30,90,365],
      'candidate_quality_windows_days':[30,90,365],
      'candidate_workload_sets':WORKLOAD_SETS,
      'm2_ridge_grid':list(M2_RIDGE_GRID),
      'candidate_spec_count':len(specs),
      'best_development_spec':best,
      'promotion_eligible':promoted,
      'market_derivative_data_used':False
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

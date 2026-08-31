#!/usr/bin/env python3
"""M3 development-only validation: batting-order path on fixed M1 champion."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

EPS=1e-9
M1_LAMBDA=50.0
RIDGE=(0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0)
M1=[
 'starter_k_rate','starter_bb_rate','starter_hr_rate','starter_nonhr_hit_rate',
 'opponent_k_rate','opponent_bb_rate','opponent_hr_rate','opponent_nonhr_hit_rate',
 'contact_interaction','power_interaction','baserunner_interaction']
PATH=[f'order_p_start_{i}' for i in range(1,9)]
SEQ=[f'order_seq_{m}_rate' for m in ('k','bb','hr','hit')]
CONTRAST=[f'order_seq_{m}_contrast' for m in ('k','bb','hr','hit')]
SPECS={
 'path_probs':PATH,
 'sequence_quality':SEQ,
 'sequence_contrast':CONTRAST,
 'sequence_quality_plus_path':SEQ+PATH,
 'sequence_contrast_plus_path':CONTRAST+PATH,
}

def logistic(x): return 1/(1+np.exp(-np.clip(np.asarray(x,float),-40,40)))
def logit(p):
 p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def ll(y,p):
 p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def br(y,p): return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))
def priors(tr):
 g=tr.groupby('dk_total_open_total').i2_over_05.agg(['sum','count']); return {float(i):float(r['sum']/r['count']) for i,r in g.iterrows() if r['count']}
def base(df,p):
 keys=np.array(sorted(p)); out=[]
 for t in pd.to_numeric(df.dk_total_open_total,errors='coerce'):
  k=float(t) if float(t) in p else float(keys[np.argmin(abs(keys-float(t)))])
  out.append(p[k])
 return np.asarray(out)
def std(tr,te,f):
 A=tr[f].apply(pd.to_numeric,errors='coerce').to_numpy(float); B=te[f].apply(pd.to_numeric,errors='coerce').to_numpy(float)
 mu=np.nanmean(A,0); sd=np.nanstd(A,0); sd=np.where((~np.isfinite(sd))|(sd<1e-9),1,sd)
 return np.nan_to_num((A-mu)/sd),np.nan_to_num((B-mu)/sd)
def fit(X,y,off,lam):
 b=np.zeros(X.shape[1]); R=np.eye(X.shape[1])*lam
 for _ in range(100):
  m=logistic(off+X@b); w=np.clip(m*(1-m),1e-6,None); grad=X.T@(y-m)-R@b; H=X.T@(w[:,None]*X)+R+np.eye(X.shape[1])*1e-10
  nb=b+np.linalg.solve(H,grad)
  if np.max(abs(nb-b))<1e-8: b=nb; break
  b=nb
 return b
def cal(y,p):
 x=logit(p); X=np.c_[np.ones(len(x)),x]; b=np.zeros(2)
 for _ in range(60):
  m=logistic(X@b); w=np.clip(m*(1-m),1e-6,None); z=X@b+(y-m)/w; nb=np.linalg.solve(X.T@(w[:,None]*X)+np.eye(2)*1e-8,X.T@(w*z))
  if np.max(abs(nb-b))<1e-9: b=nb; break
  b=nb
 return float(b[0]),float(b[1])
def fold(df,year,spec,features,lam):
 tr=df[(df.season>=2021)&(df.season<year)].copy(); te=df[df.season==year].copy(); p=priors(tr); p0tr=base(tr,p); p0te=base(te,p)
 X1tr,X1te=std(tr,te,M1); b1=fit(X1tr,tr.i2_over_05.to_numpy(float),logit(p0tr),M1_LAMBDA); p1tr=logistic(logit(p0tr)+X1tr@b1); p1te=logistic(logit(p0te)+X1te@b1)
 X3tr,X3te=std(tr,te,features); b3=fit(X3tr,tr.i2_over_05.to_numpy(float),logit(p1tr),lam); p3=logistic(logit(p1te)+X3te@b3); y=te.i2_over_05.to_numpy(float); ci,cs=cal(y,p3)
 row={'spec':spec,'test_season':year,'n':len(te),'ridge_lambda':lam,'feature_count':len(features),'m0_log_loss':ll(y,p0te),'m1_log_loss':ll(y,p1te),'m3_log_loss':ll(y,p3),'m3_vs_m1_log_loss_improvement':ll(y,p1te)-ll(y,p3),'m1_brier':br(y,p1te),'m3_brier':br(y,p3),'m3_vs_m1_brier_improvement':br(y,p1te)-br(y,p3),'calibration_intercept':ci,'calibration_slope':cs}
 co=[{'spec':spec,'test_season':year,'ridge_lambda':lam,'feature':f,'standardized_beta':float(v)} for f,v in zip(features,b3)]
 return row,co

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 d=pd.read_parquet(a.input) if a.input.endswith('.parquet') else pd.read_csv(a.input); d['season']=pd.to_numeric(d.season,errors='coerce').astype('Int64'); d['dk_total_open_total']=pd.to_numeric(d.dk_total_open_total,errors='coerce'); d['inning2_total_runs']=pd.to_numeric(d.inning2_total_runs,errors='coerce'); d=d[d.season.notna()&d.dk_total_open_total.notna()&d.inning2_total_runs.notna()].copy(); d['season']=d.season.astype(int); d['i2_over_05']=(d.inning2_total_runs>=1).astype(float)
 missing=[f for f in M1+sum(SPECS.values(),[]) if f not in d.columns]
 if missing: raise RuntimeError(f'missing columns {sorted(set(missing))}')
 rows=[]; co=[]
 for s,f in SPECS.items():
  for lam in RIDGE:
   for y in (2022,2023,2024):
    r,c=fold(d,y,s,f,lam); rows.append(r); co.extend(c)
 folds=pd.DataFrame(rows); coef=pd.DataFrame(co)
 summ=(folds.groupby(['spec','ridge_lambda','feature_count'],as_index=False).agg(mean_log_loss_improvement=('m3_vs_m1_log_loss_improvement','mean'),worst_year_log_loss_improvement=('m3_vs_m1_log_loss_improvement','min'),mean_brier_improvement=('m3_vs_m1_brier_improvement','mean')))
 summ['no_negative_log_loss_year']=summ.worst_year_log_loss_improvement>=0; summ['positive_mean_log_loss']=summ.mean_log_loss_improvement>0
 elig=summ[summ.no_negative_log_loss_year & summ.positive_mean_log_loss]; rank=(elig if len(elig) else summ).sort_values(['mean_log_loss_improvement','mean_brier_improvement'],ascending=False); best=rank.iloc[0].to_dict(); promoted=bool(best['mean_log_loss_improvement']>0 and best['worst_year_log_loss_improvement']>=0)
 folds.to_csv(out/'development_folds.csv',index=False); summ.to_csv(out/'spec_grid_summary.csv',index=False); coef.to_csv(out/'development_coefficients.csv',index=False)
 manifest={'status':'M3_PROMOTED_DEVELOPMENT' if promoted else 'M3_REJECTED_DEVELOPMENT','comparison':'M3 batting-order-path residual versus fixed M1 champion','m1_ridge_lambda_fixed':M1_LAMBDA,'development_seasons':[2021,2022,2023,2024],'selection_test_seasons':[2022,2023,2024],'holdout_season':2025,'holdout_opened':False,'candidate_specs':SPECS,'ridge_grid':list(RIDGE),'best_development_spec':best,'promotion_eligible':promoted,'market_derivative_data_used':False}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Price-blind local shrinkage optimization for I2 with nested development CV.

Purpose: test the cohort finding directly instead of selecting on one global fit.
- Freeze probability regions from p100 P(Under).
- Use 10 fixed development deciles.
- Estimate bin-specific loss curves over shrinkage strengths.
- Fit a monotone non-increasing shrinkage path as P(Under) rises.
- Select smoothness by leave-one-development-season-out CV (2022/23/24).
- Refit on all 2022-24, then evaluate once on sealed 2025.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

KEY=['game_id','season']; S=np.array([0.,10.,25.,50.,75.,100.,200.]); DEV=[2022,2023,2024]
LAMBDAS=[0.,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3]

def clip(p): return np.clip(np.asarray(p,float),1e-9,1-1e-9)
def ll(y,p):
 y=np.asarray(y,float); p=clip(p); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def br(y,p): y=np.asarray(y,float); p=np.asarray(p,float); return float(np.mean((y-p)**2))
def load(root,meta):
 m=None
 for v in meta:
  x=pd.read_csv(root/v['name']/'strict_oos_predictions.csv')[KEY+['baseline_prediction','actual_over']].copy()
  x[f"p_{int(v['prior_strength'])}"]=1-pd.to_numeric(x.baseline_prediction,errors='coerce'); x['actual_under']=1-pd.to_numeric(x.actual_over,errors='coerce')
  x=x.drop(columns=['baseline_prediction','actual_over'])
  m=x if m is None else m.merge(x.drop(columns=['actual_under']),on=KEY,validate='one_to_one')
 return m.sort_values(KEY).reset_index(drop=True)
def pred_at(panel,s):
 cols=[f'p_{int(x)}' for x in S]; mat=panel[cols].to_numpy(float); out=np.empty(len(panel))
 for i,t in enumerate(s): out[i]=np.interp(t,S,mat[i])
 return out
def assign_bins(anchor,edges): return np.clip(np.searchsorted(edges,anchor,side='right'),0,len(edges)).astype(int)
def bin_loss(panel,bins,mask):
 y=panel.actual_under.to_numpy(float); rows=np.full((10,len(S)),np.nan)
 for b in range(10):
  mb=mask&(bins==b)
  if not mb.any(): continue
  for j,s in enumerate(S): rows[b,j]=ll(y[mb],panel[f'p_{int(s)}'].to_numpy(float)[mb])
 return rows
def fit_path(losses,lam):
 B,J=losses.shape; dp=np.full((B,J),np.inf); back=np.full((B,J),-1,int); dp[0]=losses[0]
 for b in range(1,B):
  for j in range(J):
   vals=[]
   for k in range(j,J): vals.append(dp[b-1,k]+losses[b,j]+lam*(S[k]-S[j])**2)
   krel=int(np.argmin(vals)); k=j+krel; dp[b,j]=vals[krel]; back[b,j]=k
 j=int(np.argmin(dp[-1])); path=[j]
 for b in range(B-1,0,-1): j=back[b,j]; path.append(j)
 path=path[::-1]; return np.array([S[j] for j in path]),float(np.min(dp[-1]))
def path_predict(panel,bins,path): return pred_at(panel,np.array([path[b] for b in bins],float))
def metrics(panel,p,mask):
 y=panel.actual_under.to_numpy(float)[mask]; q=p[mask]
 return {'n':int(mask.sum()),'log_loss':ll(y,q),'brier':br(y,q),'mean_p_under':float(q.mean()),'actual_under_rate':float(y.mean()),'cal_gap_pp':float((y.mean()-q.mean())*100)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
 a=ap.parse_args(); meta=json.loads(Path(a.manifest).read_text()); panel=load(Path(a.variants_dir),meta); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 anchor=panel.p_100.to_numpy(float); dev=panel.season.isin(DEV).to_numpy(); hold=panel.season.eq(2025).to_numpy(); y=panel.actual_under.to_numpy(float)
 edges=panel.loc[dev,'p_100'].quantile(np.arange(.1,1,.1)).to_numpy(float); bins=assign_bins(anchor,edges)
 raw=[]; losses=bin_loss(panel,bins,dev)
 for b in range(10):
  mb=dev&(bins==b)
  for j,s in enumerate(S):
   raw.append({'bin':b+1,'p_low':float(-np.inf if b==0 else edges[b-1]),'p_high':float(np.inf if b==9 else edges[b]),'shrinkage':s,'n':int(mb.sum()),'dev_log_loss':losses[b,j]})
 pd.DataFrame(raw).to_csv(out/'local_loss_surface.csv',index=False)
 cv=[]
 for lam in LAMBDAS:
  fold_ll=[]; fold_rows=[]
  for valyr in DEV:
   tr=panel.season.isin([z for z in DEV if z!=valyr]).to_numpy(); va=panel.season.eq(valyr).to_numpy()
   path,_=fit_path(bin_loss(panel,bins,tr),lam); p=path_predict(panel,bins,path); score=ll(y[va],p[va]); fold_ll.append(score)
   fold_rows.append({'lambda':lam,'validation_year':valyr,'validation_log_loss':score,'path':'|'.join(str(int(x)) for x in path)})
  cv.extend(fold_rows); cv.append({'lambda':lam,'validation_year':'MEAN','validation_log_loss':float(np.mean(fold_ll)),'path':''})
 cvdf=pd.DataFrame(cv); cvdf.to_csv(out/'lambda_cross_validation.csv',index=False)
 means=cvdf[cvdf.validation_year.astype(str)=='MEAN'].copy(); bestlam=float(means.sort_values(['validation_log_loss','lambda']).iloc[0]['lambda'])
 path,_=fit_path(losses,bestlam); grad=path_predict(panel,bins,path)
 rows=[]
 for name,p in [('LOCAL_CV',grad)]+[(f'CONST_{int(s)}',panel[f'p_{int(s)}'].to_numpy(float)) for s in S]:
  for split,mask in [('DEV_2022_2024',dev),('HOLDOUT_2025',hold),('ALL',np.ones(len(panel),bool))]:
   rows.append({'model':name,'split':split,'bin':'ALL',**metrics(panel,p,mask)})
   for b in range(10): rows.append({'model':name,'split':split,'bin':b+1,**metrics(panel,p,mask&(bins==b))})
 perf=pd.DataFrame(rows); perf.to_csv(out/'model_and_decile_performance.csv',index=False)
 repl=[]
 for yr in [2022,2023,2024,2025]:
  m=panel.season.eq(yr).to_numpy(); L=bin_loss(panel,bins,m)
  for b in range(10):
   if np.all(np.isnan(L[b])): continue
   j=int(np.nanargmin(L[b])); repl.append({'season':yr,'bin':b+1,'n':int((m&(bins==b)).sum()),'best_shrinkage':S[j],'best_log_loss':L[b,j]})
 pd.DataFrame(repl).to_csv(out/'season_bin_local_optima.csv',index=False)
 manifest={'status':'PASS','price_used':False,'development_seasons':DEV,'sealed_validation_season':2025,'anchor':'p100 baseline P(Under)','fixed_decile_edges':[float(x) for x in edges],'shrinkage_grid':[float(x) for x in S],'lambda_grid':LAMBDAS,'selected_lambda':bestlam,'selected_bin_shrinkage':[float(x) for x in path],'constraint':'non-increasing shrinkage as P(Under) rises','selection':'leave-one-development-season-out CV; sealed 2025 untouched','caveat':'strict historical replay, not byte-for-byte live simulator'}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2)); print('\nCV'); print(means.sort_values('validation_log_loss').to_string(index=False)); print('\nOVERALL'); print(perf[perf.bin.astype(str).eq('ALL')].to_string(index=False))
if __name__=='__main__': main()

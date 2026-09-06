#!/usr/bin/env python3
"""Production-readiness validation for I2 v0.4 Local-CV shrinkage.

Price-blind. Uses the same strict chronological OOS replay variants as the local-CV
experiment. Reconstructs the frozen local-CV shrinkage path from development seasons
2022-2024, evaluates 2025, compares against constant-shrinkage controls, bootstraps
log-loss differences, and recalibrates the Under qualification gate on the new
probabilities without using sportsbook price.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['game_id','season']
S=np.array([0.,10.,25.,50.,75.,100.,200.])
DEV=[2022,2023,2024]
HOLD=2025
LAMBDAS=[0.,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3]

def clip(p): return np.clip(np.asarray(p,float),1e-9,1-1e-9)
def ll_vec(y,p):
    y=np.asarray(y,float); p=clip(p)
    return -(y*np.log(p)+(1-y)*np.log(1-p))
def ll(y,p): return float(ll_vec(y,p).mean())
def brier(y,p): y=np.asarray(y,float); p=np.asarray(p,float); return float(np.mean((y-p)**2))
def wilson(w,n,z=1.959963984540054):
    if n==0: return (np.nan,np.nan)
    ph=w/n; d=1+z*z/n; c=(ph+z*z/(2*n))/d; h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return c-h,c+h

def load_panel(root:Path, meta):
    m=None
    for v in meta:
        x=pd.read_csv(root/v['name']/'strict_oos_predictions.csv')[KEY+['baseline_prediction','actual_over']].copy()
        s=int(v['prior_strength'])
        x[f'p_{s}']=1-pd.to_numeric(x.baseline_prediction,errors='coerce')
        x['actual_under']=1-pd.to_numeric(x.actual_over,errors='coerce')
        x=x.drop(columns=['baseline_prediction','actual_over'])
        m=x if m is None else m.merge(x.drop(columns=['actual_under']),on=KEY,validate='one_to_one')
    return m.sort_values(KEY).reset_index(drop=True)

def pred_at(panel,sarr):
    mat=panel[[f'p_{int(s)}' for s in S]].to_numpy(float); out=np.empty(len(panel))
    for i,s in enumerate(sarr): out[i]=np.interp(s,S,mat[i])
    return out

def assign_bins(anchor,edges): return np.clip(np.searchsorted(edges,anchor,side='right'),0,len(edges)).astype(int)
def bin_loss(panel,bins,mask):
    y=panel.actual_under.to_numpy(float); L=np.full((10,len(S)),np.nan)
    for b in range(10):
        mb=mask&(bins==b)
        for j,s in enumerate(S):
            if mb.any(): L[b,j]=ll(y[mb],panel[f'p_{int(s)}'].to_numpy(float)[mb])
    return L

def fit_path(losses,lam):
    B,J=losses.shape; dp=np.full((B,J),np.inf); back=np.full((B,J),-1,int); dp[0]=losses[0]
    for b in range(1,B):
        for j in range(J):
            vals=[dp[b-1,k]+losses[b,j]+lam*(S[k]-S[j])**2 for k in range(j,J)]
            kr=int(np.argmin(vals)); k=j+kr; dp[b,j]=vals[kr]; back[b,j]=k
    j=int(np.argmin(dp[-1])); idx=[j]
    for b in range(B-1,0,-1): j=back[b,j]; idx.append(j)
    return np.array([S[j] for j in idx[::-1]])

def local_cv(panel):
    anchor=panel.p_100.to_numpy(float); dev=panel.season.isin(DEV).to_numpy(); y=panel.actual_under.to_numpy(float)
    edges=panel.loc[dev,'p_100'].quantile(np.arange(.1,1,.1)).to_numpy(float); bins=assign_bins(anchor,edges)
    cv=[]
    for lam in LAMBDAS:
        scores=[]
        for yr in DEV:
            tr=panel.season.isin([s for s in DEV if s!=yr]).to_numpy(); va=panel.season.eq(yr).to_numpy()
            path=fit_path(bin_loss(panel,bins,tr),lam)
            pred=pred_at(panel,np.array([path[b] for b in bins]))
            scores.append(ll(y[va],pred[va]))
        cv.append((float(np.mean(scores)),lam))
    cv.sort(); bestlam=cv[0][1]
    path=fit_path(bin_loss(panel,bins,dev),bestlam)
    pred=pred_at(panel,np.array([path[b] for b in bins]))
    return edges,bins,bestlam,path,pred,cv

def bootstrap_delta(y,p_new,p_old,nboot=5000,seed=20260906):
    y=np.asarray(y,float); a=ll_vec(y,p_new); b=ll_vec(y,p_old); d=a-b
    rng=np.random.default_rng(seed); n=len(d); vals=np.empty(nboot)
    for i in range(nboot): vals[i]=d[rng.integers(0,n,n)].mean()
    return {'delta_log_loss':float(d.mean()),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),
            'prob_local_better':float(np.mean(vals<0))}

def threshold_sweep(panel,pred):
    y=panel.actual_under.to_numpy(int); dev=panel.season.isin(DEV).to_numpy(); hold=panel.season.eq(HOLD).to_numpy()
    rows=[]
    for t in np.arange(.55,.701,.0025):
        for split,mask in [('DEV_2022_2024',dev),('HOLDOUT_2025',hold),('ALL',np.ones(len(panel),bool))]:
            m=mask&(pred>=t); n=int(m.sum()); w=int(y[m].sum()) if n else 0; lo,hi=wilson(w,n)
            rates=[]
            for yr in DEV:
                q=m&(panel.season.to_numpy()==yr)
                if q.sum(): rates.append(float(y[q].mean()))
            rows.append({'threshold':round(float(t),4),'split':split,'n':n,'wins':w,'hit_rate':(w/n if n else np.nan),
                         'mean_model_p':(float(pred[m].mean()) if n else np.nan),'wilson_low':lo,'wilson_high':hi,
                         'dev_season_sd_pp':(float(np.std(rates,ddof=1)*100) if len(rates)>1 else np.nan)})
    df=pd.DataFrame(rows)
    devdf=df[(df.split=='DEV_2022_2024')&(df.n>=150)].copy()
    # Conservative rank: maximize Wilson lower bound, then N, then lower threshold.
    cand=devdf.sort_values(['wilson_low','n','threshold'],ascending=[False,False,True]).iloc[0].to_dict() if len(devdf) else None
    return df,cand

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); meta=json.loads(Path(a.manifest).read_text()); panel=load_panel(Path(a.variants_dir),meta)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    edges,bins,bestlam,path,local,cv=local_cv(panel)
    y=panel.actual_under.to_numpy(float); p100=panel.p_100.to_numpy(float)
    dev=panel.season.isin(DEV).to_numpy(); hold=panel.season.eq(HOLD).to_numpy(); allm=np.ones(len(panel),bool)
    rows=[]
    for model,p in [('LOCAL_CV',local)]+[(f'CONST_{int(s)}',panel[f'p_{int(s)}'].to_numpy(float)) for s in S]:
        for split,m in [('DEV_2022_2024',dev),('HOLDOUT_2025',hold),('ALL',allm)]:
            rows.append({'model':model,'split':split,'n':int(m.sum()),'log_loss':ll(y[m],p[m]),'brier':brier(y[m],p[m]),
                         'mean_p_under':float(p[m].mean()),'actual_under_rate':float(y[m].mean()),'cal_gap_pp':float((y[m].mean()-p[m].mean())*100)})
    perf=pd.DataFrame(rows); perf.to_csv(out/'overall_performance.csv',index=False)
    boot=[]
    for split,m in [('DEV_2022_2024',dev),('HOLDOUT_2025',hold),('ALL',allm)]:
        r=bootstrap_delta(y[m],local[m],p100[m]); r['split']=split; boot.append(r)
    pd.DataFrame(boot).to_csv(out/'bootstrap_local_vs_p100.csv',index=False)
    sweep,cand=threshold_sweep(panel,local); sweep.to_csv(out/'under_threshold_sweep.csv',index=False)
    if cand:
        holdrow=sweep[(sweep.split=='HOLDOUT_2025')&(sweep.threshold==cand['threshold'])].iloc[0].to_dict()
    else: holdrow=None
    panel_out=pd.DataFrame({'game_id':panel.game_id,'season':panel.season,'p_under_p100':p100,'p_under_local_cv':local,'actual_under':y,'local_bin':bins+1})
    panel_out.to_csv(out/'v04_oos_predictions.csv',index=False)
    # Production gates: local must beat p100 overall and in 2025, bootstrap P>0.80 overall, and candidate gate must retain >=150 dev games and >=40 holdout games.
    p_local_all=float(perf.query("model=='LOCAL_CV' and split=='ALL'").log_loss.iloc[0]); p100_all=float(perf.query("model=='CONST_100' and split=='ALL'").log_loss.iloc[0])
    p_local_h=float(perf.query("model=='LOCAL_CV' and split=='HOLDOUT_2025'").log_loss.iloc[0]); p100_h=float(perf.query("model=='CONST_100' and split=='HOLDOUT_2025'").log_loss.iloc[0])
    b_all=[r for r in boot if r['split']=='ALL'][0]
    gates={'beats_p100_overall':p_local_all<p100_all,'beats_p100_2025':p_local_h<p100_h,'bootstrap_prob_better_overall_ge_0_80':b_all['prob_local_better']>=.80,
           'under_gate_dev_n_ge_150':bool(cand and cand['n']>=150),'under_gate_2025_n_ge_40':bool(holdrow and holdrow['n']>=40)}
    status='PASS' if all(gates.values()) else 'FAIL'
    manifest={'status':status,'model':'I2 v0.4 Local-CV Production Candidate','price_used':False,'development_seasons':DEV,'validation_season':HOLD,
              'selected_lambda':bestlam,'selected_bin_shrinkage':[float(x) for x in path],'fixed_decile_edges':[float(x) for x in edges],
              'production_gates':gates,'selected_under_threshold_from_dev':cand,'validation_at_selected_under_threshold':holdrow,
              'note':'2025 has been inspected in prior research; treat this as production-candidate validation, not a pristine untouched holdout. Fresh 2026 prospective validation remains the final external confirmation layer.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=float)+'\n')
    print(json.dumps(manifest,indent=2,default=float)); print('\nPERFORMANCE'); print(perf.to_string(index=False)); print('\nBOOTSTRAP'); print(pd.DataFrame(boot).to_string(index=False))

if __name__=='__main__': main()

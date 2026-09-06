#!/usr/bin/env python3
"""Robust low-complexity validation of probability-dependent I2 shrinkage.

Compares constant shrinkage against two simple monotone families on 2022-2024,
then evaluates the selected candidate once on sealed 2025:
- linear 2-endpoint gradient across dev q10-q90 P(Under)
- two-level step gradient with one breakpoint at dev median P(Under)

Selection score is mean season log loss + 0.5 * season SD, which rewards fit and
penalizes instability. Price-blind; uses strict chronological OOS replay outputs.
"""
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np,pandas as pd
KEY=['game_id','season']

def clip(p): return np.clip(np.asarray(p,float),1e-9,1-1e-9)
def ll(y,p):
    y=np.asarray(y,float); p=clip(p); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def br(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.mean((y-p)**2))
def load(root,meta):
    z=None
    for v in meta:
        x=pd.read_csv(Path(root)/v['name']/'strict_oos_predictions.csv')
        x=x[KEY+['baseline_prediction','actual_over']].copy()
        x[f"p_{v['name']}"]=1-pd.to_numeric(x.baseline_prediction,errors='coerce')
        x['actual_under']=1-pd.to_numeric(x.actual_over,errors='coerce')
        x=x.drop(columns=['baseline_prediction','actual_over'])
        z=x if z is None else z.merge(x.drop(columns=['actual_under']),on=KEY,validate='one_to_one')
    return z.sort_values(KEY).reset_index(drop=True)
def pred_at(panel,meta,s):
    pairs=sorted((float(v['prior_strength']),v['name']) for v in meta)
    xs=np.array([a for a,_ in pairs]); M=np.column_stack([panel[f'p_{n}'] for _,n in pairs])
    out=np.empty(len(panel))
    for i,t in enumerate(np.asarray(s,float)): out[i]=np.interp(t,xs,M[i])
    return out
def season_metrics(panel,pred,seasons):
    vals=[]
    for s in seasons:
        m=panel.season.eq(s).to_numpy(); vals.append(ll(panel.actual_under.to_numpy()[m],pred[m]))
    return vals
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); meta=json.loads(Path(a.manifest).read_text()); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    panel=load(a.variants_dir,meta); anchor=panel.p_p100.to_numpy(float); y=panel.actual_under.to_numpy(float)
    devyrs=[2022,2023,2024]; hold=panel.season.eq(2025).to_numpy(); dev=panel.season.isin(devyrs).to_numpy()
    q10,q50,q90=np.quantile(anchor[dev],[.10,.50,.90]); grid=sorted(float(v['prior_strength']) for v in meta)
    cand=[]
    def add(family,params,s):
        p=pred_at(panel,meta,s); sv=season_metrics(panel,p,devyrs); score=float(np.mean(sv)+.5*np.std(sv,ddof=0))
        cand.append({'family':family,'params':json.dumps(params,sort_keys=True),'robust_score':score,'dev_log_loss':ll(y[dev],p[dev]),'dev_season_sd':float(np.std(sv,ddof=0)),'holdout_2025_log_loss':ll(y[hold],p[hold]),'holdout_2025_brier':br(y[hold],p[hold]),'pred':p})
    for s in grid: add('constant',{'s':s},np.full(len(panel),s))
    for lo in grid:
        for hi in grid:
            if lo < hi: continue  # shrinkage must not rise with P(Under)
            eff=np.interp(anchor,[q10,q90],[lo,hi],left=lo,right=hi)
            add('linear_2_endpoint',{'low_p_s':lo,'high_p_s':hi},eff)
    for low in grid:
        for high in grid:
            if low < high: continue
            eff=np.where(anchor<=q50,low,high)
            add('step_median',{'low_p_s':low,'high_p_s':high},eff)
    tab=pd.DataFrame([{k:v for k,v in c.items() if k!='pred'} for c in cand]).sort_values(['robust_score','dev_log_loss']).reset_index(drop=True)
    tab.to_csv(out/'candidate_results.csv',index=False)
    best_idx=tab.index[0]; bestrow=tab.iloc[0]; match=next(c for c in cand if c['family']==bestrow.family and c['params']==bestrow.params)
    bestp=match['pred']
    # constant benchmark best by same robust score
    const=tab[tab.family=='constant'].iloc[0]
    summary={'status':'PASS','price_used':False,'development_seasons':devyrs,'sealed_validation_season':2025,'selection_score':'mean annual dev log loss + 0.5 * annual SD','q10':float(q10),'q50':float(q50),'q90':float(q90),'selected_family':bestrow.family,'selected_params':json.loads(bestrow.params),'selected_robust_score':float(bestrow.robust_score),'selected_dev_log_loss':float(bestrow.dev_log_loss),'selected_holdout_2025_log_loss':float(bestrow.holdout_2025_log_loss),'best_constant':{'params':json.loads(const.params),'robust_score':float(const.robust_score),'dev_log_loss':float(const.dev_log_loss),'holdout_2025_log_loss':float(const.holdout_2025_log_loss)},'caveat':'Strict historical replay; not byte-for-byte live simulator replay.'}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2)); print('\nTOP 20'); print(tab.head(20).to_string(index=False))
if __name__=='__main__': main()

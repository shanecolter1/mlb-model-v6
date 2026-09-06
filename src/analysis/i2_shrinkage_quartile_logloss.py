#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['game_id','season']

def logloss(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),1e-9,1-1e-9)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((y-p)**2))

def load(path):
    x=pd.read_csv(path)
    x['p_over']=pd.to_numeric(x['baseline_prediction'],errors='coerce')
    x['p_under']=1-x.p_over
    x['actual_over']=pd.to_numeric(x['actual_over'],errors='coerce')
    x['actual_under']=1-x.actual_over
    return x.dropna(subset=['p_under','actual_under']).copy()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); meta=json.loads(Path(a.manifest).read_text()); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    loaded={v['name']:load(Path(a.variants_dir)/v['name']/'strict_oos_predictions.csv') for v in meta}
    current=loaded['p100']
    cuts=current.p_under.quantile([0.25,0.50,0.75]).to_dict(); q25,q50,q75=cuts[0.25],cuts[0.50],cuts[0.75]
    def lab(p):
        if p<=q25: return 'Q1_HIGH_OVER'
        if p<=q50: return 'Q2_MODERATE_OVER_NEUTRAL'
        if p<=q75: return 'Q3_MODERATE_UNDER_NEUTRAL'
        return 'Q4_HIGH_UNDER'
    anchor=current[KEY+['p_under']].copy(); anchor['cohort']=anchor.p_under.map(lab)
    rows=[]
    for v in meta:
        x=loaded[v['name']]
        z=anchor[KEY+['cohort']].merge(x[KEY+['p_under','p_over','actual_under','actual_over']],on=KEY,how='left',validate='one_to_one')
        rows.append({'cohort':'ALL','variant':v['name'],'prior_strength':v['prior_strength'],'n':len(z),'log_loss':logloss(z.actual_under,z.p_under),'brier':brier(z.actual_under,z.p_under)})
        for cohort in ['Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL','Q4_HIGH_UNDER']:
            q=z[z.cohort==cohort]
            if cohort=='Q1_HIGH_OVER': y,p=q.actual_over,q.p_over
            else: y,p=q.actual_under,q.p_under
            rows.append({'cohort':cohort,'variant':v['name'],'prior_strength':v['prior_strength'],'n':len(q),'log_loss':logloss(y,p),'brier':brier(y,p)})
    tab=pd.DataFrame(rows)
    base=tab[tab.variant=='p100'][['cohort','log_loss','brier']].rename(columns={'log_loss':'p100_log_loss','brier':'p100_brier'})
    tab=tab.merge(base,on='cohort',how='left')
    tab['delta_log_loss_vs_p100']=tab.log_loss-tab.p100_log_loss
    tab['delta_brier_vs_p100']=tab.brier-tab.p100_brier
    tab.to_csv(out/'fixed_cohort_logloss_by_shrinkage.csv',index=False)
    print(tab.sort_values(['cohort','prior_strength'],ascending=[True,False]).to_string(index=False))

if __name__=='__main__': main()

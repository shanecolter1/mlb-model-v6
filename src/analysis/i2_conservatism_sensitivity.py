#!/usr/bin/env python3
"""Summarize I2 replay sensitivity to conservative shrinkage/regularization choices.

This analysis is price-blind. It compares chronological OOS replay variants that
change only two explicit conservative controls in the strict replay:
  1) opening-total prior shrinkage strength;
  2) ridge regularization on baseball-context coefficients.

It reports both moving-threshold cohorts and fixed cohorts anchored to the current
P(Under) >= 59.5% / 59.75% baseline selections, so probability compression can be
separated from simple changes in which games qualify.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['game_id','season']


def wilson(w,n,z=1.959963984540054):
    if n<=0: return (float('nan'),float('nan'))
    p=w/n; den=1+z*z/n
    ctr=(p+z*z/(2*n))/den
    rad=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return ctr-rad,ctr+rad


def ll(y,p):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))


def br(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((y-p)**2))


def load_variant(path:Path):
    x=pd.read_csv(path)
    x['p_under']=1-pd.to_numeric(x['prediction'],errors='coerce')
    x['base_p_under']=1-pd.to_numeric(x['baseline_prediction'],errors='coerce')
    x['actual_under']=1-pd.to_numeric(x['actual_over'],errors='coerce')
    return x


def summarize_one(path:Path, variant:str, prior_strength:float, ridge_lambda:float):
    x=load_variant(path)
    y_over=1-x.actual_under
    rows=[]
    for source,pcol in [('context','p_under'),('baseline','base_p_under')]:
        p=pd.to_numeric(x[pcol],errors='coerce'); y=x.actual_under
        overall={
            'variant':variant,'source':source,'prior_strength':prior_strength,'ridge_lambda':ridge_lambda,
            'n_all':int(len(x)),'log_loss_over':ll(y_over,1-p),'brier_over':br(y_over,1-p),
            'mean_p_under_all':float(p.mean()),'actual_under_rate_all':float(y.mean()),
        }
        for t in [0.59,0.5925,0.595,0.5975,0.60,0.61,0.62]:
            q=x[p>=t]; n=len(q); w=int(q.actual_under.sum()) if n else 0
            hit=w/n if n else float('nan'); mp=float(q[pcol].mean()) if n else float('nan')
            lo,hi=wilson(w,n)
            overall.update({f'n_{t:.4f}':n,f'hit_{t:.4f}':hit,f'meanp_{t:.4f}':mp,
                            f'gap_pp_{t:.4f}':(hit-mp)*100 if n else float('nan'),
                            f'wilson_low_{t:.4f}':lo,f'wilson_high_{t:.4f}':hi})
        rows.append(overall)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants-dir',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args()
    meta=json.loads(Path(a.manifest).read_text())
    rows=[]; loaded={}
    for v in meta:
        p=Path(a.variants_dir)/v['name']/'strict_oos_predictions.csv'
        loaded[v['name']]=load_variant(p)
        rows += summarize_one(p,v['name'],float(v['prior_strength']),float(v['ridge_lambda']))
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tab=pd.DataFrame(rows)
    tab.to_csv(out/'conservatism_sensitivity_summary.csv',index=False)

    focus_cols=['variant','source','prior_strength','ridge_lambda','n_all','log_loss_over','brier_over',
                'n_0.5950','hit_0.5950','meanp_0.5950','gap_pp_0.5950','wilson_low_0.5950','wilson_high_0.5950',
                'n_0.5975','hit_0.5975','meanp_0.5975','gap_pp_0.5975']
    tab[focus_cols].to_csv(out/'focus_595_5975.csv',index=False)

    # Fixed-cohort test: hold the exact games constant using current baseline selection.
    anchor=loaded['current_p100_r10'][KEY+['base_p_under','actual_under']].copy()
    fixed=[]
    for gate in [0.595,0.5975]:
        cohort=anchor[anchor.base_p_under>=gate][KEY+['actual_under']].copy()
        n=len(cohort); hit=float(cohort.actual_under.mean()) if n else float('nan')
        for v in meta:
            x=loaded[v['name']][KEY+['p_under','base_p_under']].copy()
            q=cohort.merge(x,on=KEY,how='left',validate='one_to_one')
            for source,pcol in [('baseline','base_p_under'),('context','p_under')]:
                mp=float(q[pcol].mean())
                fixed.append({'anchor_gate':gate,'anchor':'current_p100_r10 baseline','variant':v['name'],
                              'source':source,'prior_strength':v['prior_strength'],'ridge_lambda':v['ridge_lambda'],
                              'fixed_n':n,'fixed_realized_hit_rate':hit,'fixed_mean_model_probability':mp,
                              'fixed_gap_pp':(hit-mp)*100})
    fixed_tab=pd.DataFrame(fixed)
    fixed_tab.to_csv(out/'fixed_cohort_probability_compression.csv',index=False)

    manifest={'status':'PASS','price_used':False,'market_fields_used':[],
              'purpose':'Test whether explicit replay conservatism suppresses true high-Under probability.',
              'variants':meta,
              'fixed_cohort_anchor':'current prior=100 baseline qualification at 59.5% and 59.75%',
              'important_caveat':'Diagnoses the strict historical replay/challenger, not a byte-for-byte live simulator replay.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nMOVING THRESHOLD COHORTS')
    print(tab[focus_cols].to_string(index=False))
    print('\nFIXED COHORTS')
    print(fixed_tab.to_string(index=False))

if __name__=='__main__': main()

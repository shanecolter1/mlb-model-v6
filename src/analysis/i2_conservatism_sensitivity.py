#!/usr/bin/env python3
"""Summarize I2 replay sensitivity to conservative shrinkage/regularization choices.

This analysis is price-blind. It compares chronological OOS replay variants that
change only two explicit conservative controls in the strict replay:
  1) opening-total prior shrinkage strength;
  2) ridge regularization on baseball-context coefficients.

It does not alter or use sportsbook derivative prices.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd


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


def summarize_one(path:Path, variant:str, prior_strength:float, ridge_lambda:float):
    x=pd.read_csv(path)
    p_over=pd.to_numeric(x['prediction'],errors='coerce')
    base_over=pd.to_numeric(x['baseline_prediction'],errors='coerce')
    y_over=pd.to_numeric(x['actual_over'],errors='coerce')
    x=x.assign(p_under=1-p_over,base_p_under=1-base_over,actual_under=1-y_over)
    rows=[]
    for source,pcol in [('context','p_under'),('baseline','base_p_under')]:
        p=pd.to_numeric(x[pcol],errors='coerce')
        y=pd.to_numeric(x['actual_under'],errors='coerce')
        overall={
            'variant':variant,'source':source,'prior_strength':prior_strength,'ridge_lambda':ridge_lambda,
            'n_all':int(len(x)),'log_loss_over':ll(y_over,1-p) if source=='context' else ll(y_over,base_over),
            'brier_over':br(y_over,1-p) if source=='context' else br(y_over,base_over),
            'mean_p_under_all':float(p.mean()),'actual_under_rate_all':float(y.mean()),
        }
        for t in [0.59,0.5925,0.595,0.5975,0.60,0.61,0.62]:
            q=x[p>=t]
            n=len(q); w=int(q.actual_under.sum()) if n else 0
            hit=w/n if n else float('nan'); mp=float(q[pcol].mean()) if n else float('nan')
            lo,hi=wilson(w,n)
            overall.update({
                f'n_{t:.4f}':n,
                f'hit_{t:.4f}':hit,
                f'meanp_{t:.4f}':mp,
                f'gap_pp_{t:.4f}':(hit-mp)*100 if n else float('nan'),
                f'wilson_low_{t:.4f}':lo,
                f'wilson_high_{t:.4f}':hi,
            })
        rows.append(overall)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants-dir',required=True)
    ap.add_argument('--manifest',required=True,help='JSON list of variant metadata')
    ap.add_argument('--output-dir',required=True)
    a=ap.parse_args()
    meta=json.loads(Path(a.manifest).read_text())
    rows=[]
    for v in meta:
        p=Path(a.variants_dir)/v['name']/'strict_oos_predictions.csv'
        rows += summarize_one(p,v['name'],float(v['prior_strength']),float(v['ridge_lambda']))
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tab=pd.DataFrame(rows)
    tab.to_csv(out/'conservatism_sensitivity_summary.csv',index=False)

    # Focus table for the empirically important 59.5% gate.
    focus_cols=['variant','source','prior_strength','ridge_lambda','n_all','log_loss_over','brier_over',
                'n_0.5950','hit_0.5950','meanp_0.5950','gap_pp_0.5950','wilson_low_0.5950','wilson_high_0.5950',
                'n_0.5975','hit_0.5975','meanp_0.5975','gap_pp_0.5975']
    tab[focus_cols].to_csv(out/'focus_595_5975.csv',index=False)

    manifest={
        'status':'PASS','price_used':False,'market_fields_used':[],
        'purpose':'Test whether explicit replay conservatism (prior shrinkage and ridge regularization) suppresses true high-Under probability.',
        'variants':meta,
        'important_caveat':'This diagnoses the strict historical replay/challenger, not a byte-for-byte replay of the live simulator.'
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(tab[focus_cols].to_string(index=False))

if __name__=='__main__': main()

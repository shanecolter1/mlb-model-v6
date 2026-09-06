#!/usr/bin/env python3
"""Price-blind I2 shrinkage experiment with fixed and moving quartile cohorts.

Only the opening-total prior shrinkage strength changes across variants. Ridge and
all other replay mechanics remain fixed. Primary cohort analysis freezes quartile
membership from the current prior_strength=100 baseline P(Under) ranking so that
probability changes are measured on exactly the same games.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['game_id','season']
Z=1.959963984540054


def clip(p): return np.clip(np.asarray(p,float),1e-9,1-1e-9)
def logloss(y,p):
    y=np.asarray(y,float); p=clip(p)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((y-p)**2))
def wilson(w,n):
    if not n: return float('nan'),float('nan')
    ph=w/n; den=1+Z*Z/n
    ctr=(ph+Z*Z/(2*n))/den
    rad=Z*math.sqrt(ph*(1-ph)/n+Z*Z/(4*n*n))/den
    return ctr-rad,ctr+rad

def calibration_intercept_slope(y,p):
    """Newton solve logistic calibration y ~ a + b*logit(p)."""
    y=np.asarray(y,float); p=clip(p); x=np.log(p/(1-p))
    X=np.column_stack([np.ones(len(x)),x])
    beta=np.array([0.0,1.0])
    for _ in range(100):
        eta=X@beta; mu=1/(1+np.exp(-np.clip(eta,-35,35)))
        w=np.maximum(mu*(1-mu),1e-8)
        g=X.T@(y-mu)
        H=-(X.T*w)@X
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: break
        beta=beta-step
        if np.max(np.abs(step))<1e-10: break
    return float(beta[0]),float(beta[1])

def mace(y,p,bins=10):
    d=pd.DataFrame({'y':np.asarray(y,float),'p':np.asarray(p,float)}).dropna()
    try: d['bin']=pd.qcut(d.p,q=min(bins,d.p.nunique()),duplicates='drop')
    except Exception: return float('nan')
    g=d.groupby('bin',observed=True).agg(n=('y','size'),obs=('y','mean'),pred=('p','mean'))
    return float(np.average(np.abs(g.obs-g.pred),weights=g.n))

def load(path):
    x=pd.read_csv(path)
    x['p_over']=pd.to_numeric(x['baseline_prediction'],errors='coerce')
    x['p_under']=1-x.p_over
    x['actual_over']=pd.to_numeric(x['actual_over'],errors='coerce')
    x['actual_under']=1-x.actual_over
    return x.dropna(subset=['p_under','actual_under']).copy()

def cohort_stats(q,pcol,ycol,label,variant,prior_strength,cohort_mode):
    n=len(q); w=int(q[ycol].sum()) if n else 0
    hit=w/n if n else float('nan'); mp=float(q[pcol].mean()) if n else float('nan')
    lo,hi=wilson(w,n)
    seas=q.groupby('season')[ycol].mean() if n else pd.Series(dtype=float)
    return {
      'cohort_mode':cohort_mode,'cohort':label,'variant':variant,'prior_strength':prior_strength,
      'n':n,'wins':w,'losses':n-w,'mean_probability':mp,'realized_hit_rate':hit,
      'calibration_gap_pp':(hit-mp)*100 if n else float('nan'),
      'wilson_low':lo,'wilson_high':hi,'season_hit_rate_sd_pp':float(seas.std(ddof=0)*100) if len(seas)>0 else float('nan'),
      **{f'hit_{int(s)}':float(seas.get(s,float("nan"))) for s in [2022,2023,2024,2025]}
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); meta=json.loads(Path(a.manifest).read_text()); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    loaded={v['name']:load(Path(a.variants_dir)/v['name']/'strict_oos_predictions.csv') for v in meta}
    current=loaded['p100']
    # Quartiles are frozen from current baseline P(Under): Q1=high Over ... Q4=high Under.
    cuts=current.p_under.quantile([0.25,0.50,0.75]).to_dict(); q25,q50,q75=cuts[0.25],cuts[0.50],cuts[0.75]
    def fixed_label(p):
        if p<=q25: return 'Q1_HIGH_OVER'
        if p<=q50: return 'Q2_MODERATE_OVER_NEUTRAL'
        if p<=q75: return 'Q3_MODERATE_UNDER_NEUTRAL'
        return 'Q4_HIGH_UNDER'
    anchor=current[KEY+['p_under','actual_under','actual_over']].copy(); anchor['fixed_cohort']=anchor.p_under.map(fixed_label)

    overall=[]; fixed=[]; moving=[]
    for v in meta:
        name=v['name']; ps=float(v['prior_strength']); x=loaded[name]
        y=x.actual_under.to_numpy(); p=x.p_under.to_numpy(); ci,cs=calibration_intercept_slope(y,p)
        overall.append({'variant':name,'prior_strength':ps,'n':len(x),'log_loss_under':logloss(y,p),'brier_under':brier(y,p),
                        'calibration_intercept':ci,'calibration_slope':cs,'mace_10bin':mace(y,p),
                        'mean_p_under':float(np.mean(p)),'actual_under_rate':float(np.mean(y))})
        # Fixed cohorts anchored to p100 membership.
        z=anchor[KEY+['fixed_cohort']].merge(x[KEY+['p_under','p_over','actual_under','actual_over']],on=KEY,how='left',validate='one_to_one')
        for lab in ['Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL','Q4_HIGH_UNDER']:
            q=z[z.fixed_cohort==lab]
            if lab=='Q1_HIGH_OVER': fixed.append(cohort_stats(q,'p_over','actual_over',lab,name,ps,'fixed_current_p100_quartile'))
            else: fixed.append(cohort_stats(q,'p_under','actual_under',lab,name,ps,'fixed_current_p100_quartile'))
        # Secondary moving quartiles under each variant.
        xx=x.copy(); qq=xx.p_under.quantile([0.25,0.50,0.75]); a1,a2,a3=qq.loc[0.25],qq.loc[0.50],qq.loc[0.75]
        xx['moving_cohort']=np.select([xx.p_under<=a1,xx.p_under<=a2,xx.p_under<=a3],['Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL'],default='Q4_HIGH_UNDER')
        for lab in ['Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL','Q4_HIGH_UNDER']:
            q=xx[xx.moving_cohort==lab]
            if lab=='Q1_HIGH_OVER': moving.append(cohort_stats(q,'p_over','actual_over',lab,name,ps,'moving_variant_quartile'))
            else: moving.append(cohort_stats(q,'p_under','actual_under',lab,name,ps,'moving_variant_quartile'))

    pd.DataFrame(overall).sort_values('prior_strength',ascending=False).to_csv(out/'overall_fit_by_shrinkage.csv',index=False)
    pd.DataFrame(fixed).to_csv(out/'fixed_quartile_cohort_results.csv',index=False)
    pd.DataFrame(moving).to_csv(out/'moving_quartile_cohort_results.csv',index=False)
    anchor[KEY+['p_under','fixed_cohort']].to_csv(out/'current_p100_fixed_cohort_membership.csv',index=False)
    manifest={'status':'PASS','price_used':False,'market_fields_used':[],
              'experiment':'I2 shrinkage quartile sensitivity','changed_parameter_only':'prior_strength',
              'ridge_lambda_fixed':10,'variants':meta,'current_anchor_variant':'p100',
              'quartile_cutpoints_p_under':{'q25':float(q25),'q50':float(q50),'q75':float(q75)},
              'primary_analysis':'fixed cohort membership from current p100 baseline ranking',
              'secondary_analysis':'moving quartiles for each variant',
              'caveat':'Strict chronological OOS historical replay, not byte-for-byte live simulator replay.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('OVERALL FIT')
    print(pd.DataFrame(overall).sort_values('prior_strength',ascending=False).to_string(index=False))
    print('\nFIXED QUARTILES')
    print(pd.DataFrame(fixed).to_string(index=False))

if __name__=='__main__': main()

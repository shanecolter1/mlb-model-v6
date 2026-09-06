#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EPS=1e-9
EDGES=np.array([0.5238396498423443,0.5491157480364879,0.5505588397195031,0.5722073331366149,0.5765829070767106,0.5802455678499445,0.5868605910212334,0.5886883820373516,0.5947180190516053])
PATH=np.array([200.,10.,10.,10.,10.,10.,10.,10.,10.,10.])

def clip(p): return np.clip(np.asarray(p,float),EPS,1-EPS)
def logit(p): p=clip(p); return np.log(p/(1-p))
def logistic(x): return 1/(1+np.exp(-np.clip(np.asarray(x,float),-40,40)))
def ll(y,p):
    y=np.asarray(y,float); p=clip(p); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def br(y,p): y=np.asarray(y,float); p=np.asarray(p,float); return float(np.mean((y-p)**2))

def fit_priors(train,strength):
    broad=float(train.actual_over.mean())
    g=train.groupby('opening_total').actual_over.agg(['sum','count'])
    pri={float(k):(float(r['sum'])+strength*broad)/(float(r['count'])+strength) for k,r in g.iterrows()}
    return broad,pri

def at_knots(x,broad,pri):
    keys=sorted(pri)
    if not keys: return broad
    t=float(x)
    if t<=keys[0]: return pri[keys[0]]
    if t>=keys[-1]: return pri[keys[-1]]
    j=np.searchsorted(keys,t)
    lo,hi=keys[j-1],keys[j]
    w=(t-lo)/(hi-lo)
    return (1-w)*pri[lo]+w*pri[hi]

def nearest_prior(x,broad,pri):
    keys=list(pri)
    if not keys:return broad
    k=min(keys,key=lambda z:abs(z-float(x)))
    return pri[k]

def bin_idx(p_under_100): return int(np.clip(np.searchsorted(EDGES,p_under_100,side='right'),0,9))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',required=True); ap.add_argument('--parent',required=True); ap.add_argument('--p100',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.master,low_memory=False)
    if 'benchmark_matched' in raw.columns: raw=raw[raw.benchmark_matched==True].copy()
    raw['season']=pd.to_numeric(raw.season,errors='coerce'); raw['opening_total']=pd.to_numeric(raw.dk_total_open_total,errors='coerce'); raw['i2_runs']=pd.to_numeric(raw.inning2_total_runs,errors='coerce'); raw['actual_over']=(raw.i2_runs>=1).astype(float)
    raw=raw.dropna(subset=['season','opening_total','i2_runs']).copy(); raw['season']=raw.season.astype(int)

    parent=pd.read_csv(a.parent)
    p100=pd.read_csv(a.p100)
    keep=[c for c in ['game_id','season','opening_total','baseline_prediction','prediction'] if c in p100.columns]
    p100=p100[keep].drop_duplicates(['game_id','season'])
    m=parent.merge(p100,on=['game_id','season'],how='inner',validate='one_to_one')
    # Parent file carries actual_under and v0.4 probability on exactly the same game ids.
    m['actual_under']=pd.to_numeric(m.actual_under,errors='coerce')
    m['current_v04']=pd.to_numeric(m.p_under_local_cv,errors='coerce')
    m['context_logit_delta']=logit(1-pd.to_numeric(m.prediction,errors='coerce'))-logit(1-pd.to_numeric(m.baseline_prediction,errors='coerce'))

    rows=[]
    for season in sorted(m.season.unique()):
        train=raw[raw.season<season]
        te=m[m.season==season].copy()
        if train.empty or te.empty: continue
        broad100,pri100=fit_priors(train,100.)
        # Local-CV bin remains selected from the original discrete p100 anchor, exactly as production.
        anchor_under=np.array([1-nearest_prior(t,broad100,pri100) for t in te.opening_total])
        bins=np.array([bin_idx(p) for p in anchor_under]); strengths=PATH[bins]
        interp=[]; discrete=[]
        for (_,r),s in zip(te.iterrows(),strengths):
            broad,pri=fit_priors(train,float(s))
            discrete.append(1-nearest_prior(r.opening_total,broad,pri))
            interp.append(1-at_knots(r.parent_expected_total,broad,pri))
        te['recalc_v04_prior']=discrete
        te['interp_prior']=interp
        te['local_bin']=bins+1; te['local_shrinkage']=strengths
        # Same historical context delta applied to both arms as a sensitivity layer.
        d=te.context_logit_delta.to_numpy(float)
        te['recalc_v04_with_context']=logistic(logit(te.recalc_v04_prior)+d)
        te['interp_with_context']=logistic(logit(te.interp_prior)+d)
        rows.append(te)
    o=pd.concat(rows,ignore_index=True)

    metrics=[]
    for split,mask in [('ALL',np.ones(len(o),bool)),('DEV_2023_2024',o.season.isin([2023,2024]).to_numpy()),('2025',o.season.eq(2025).to_numpy())]:
        x=o.loc[mask]
        for name,col in [('CURRENT_V04','current_v04'),('RECALC_DISCRETE_PRIOR','recalc_v04_prior'),('INTERPOLATED_PRIOR','interp_prior'),('RECALC_DISCRETE_PLUS_SAME_CONTEXT','recalc_v04_with_context'),('INTERPOLATED_PLUS_SAME_CONTEXT','interp_with_context')]:
            metrics.append({'split':split,'model':name,'n':len(x),'logloss':ll(x.actual_under,x[col]),'brier':br(x.actual_under,x[col]),'mean_p_under':float(x[col].mean()),'actual_under':float(x.actual_under.mean())})
    perf=pd.DataFrame(metrics); perf.to_csv(out/'performance.csv',index=False)

    # Support diagnostics: if interpolation works as intended it should fill gaps without forcing normality.
    def moments(s):
        s=pd.Series(s).dropna(); return {'mean':float(s.mean()),'sd':float(s.std()),'skew':float(s.skew()),'excess_kurtosis':float(s.kurt()),'p01':float(s.quantile(.01)),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99))}
    shape={'current_v04':moments(o.current_v04),'interpolated_prior':moments(o.interp_prior),'interpolated_plus_context':moments(o.interp_with_context),'parent_expected_total':moments(o.parent_expected_total)}
    (out/'shape.json').write_text(json.dumps(shape,indent=2)+'\n')
    o[['game_id','season','opening_total','parent_expected_total','actual_under','current_v04','recalc_v04_prior','interp_prior','context_logit_delta','recalc_v04_with_context','interp_with_context','local_bin','local_shrinkage']].to_csv(out/'oos_predictions.csv',index=False)
    manifest={'experiment':'v0.4 exact structural-prior interpolation test','production_changed':False,'price_used_for_probability':False,'market_i2_price_used':False,'local_cv_edges_unchanged':True,'local_cv_path_unchanged':True,'interpolation':'linear between chronological empirical opening-total prior knots, evaluated at continuous parent-style baseball expected total','context_sensitivity':'same historical strict-context logit delta applied identically to discrete and interpolated arms','caveat':'parent expected total uses retrospective final-feed lineup / actual starter identities with prior-date statistics; oracle-style structural test, not pristine pregame replay','n':len(o),'shape':shape}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(perf.to_string(index=False)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

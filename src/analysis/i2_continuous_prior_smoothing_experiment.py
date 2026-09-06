#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

SHRINKS=[0,10,25,50,75,100,200]
ALPHAS=[0.0,0.25,0.5,0.75,1.0]
DEV_SEASONS=[2022,2023,2024]
VAL_SEASON=2025

def fit_prior(train, shrink):
    broad=float(train.actual_under.mean())
    g=train.groupby('opening_total').actual_under.agg(['sum','count']).reset_index().sort_values('opening_total')
    g['raw_rate']=(g['sum']+shrink*broad)/(g['count']+shrink)
    iso=IsotonicRegression(increasing=False,out_of_bounds='clip')
    g['iso_rate']=iso.fit_transform(g.opening_total.to_numpy(float),g.raw_rate.to_numpy(float),sample_weight=g['count'].to_numpy(float))
    spline=PchipInterpolator(g.opening_total.to_numpy(float),g.iso_rate.to_numpy(float),extrapolate=True)
    return spline,g

def predict(spline, frame, alpha):
    p=np.clip(spline(frame.opening_total.to_numpy(float)),0.01,0.99)
    # existing team-context effect expressed as a continuous logit delta on Over;
    # invert sign for Under and scale only on development folds.
    return expit(logit(p)-alpha*frame.ctx_logit_delta.to_numpy(float))

def metrics(y,p):
    return {'n':int(len(y)),'logloss':float(log_loss(y,p)),'brier':float(brier_score_loss(y,p)),'mean_p_under':float(np.mean(p)),'actual_under':float(np.mean(y)),'calibration_gap_pp':float(100*(np.mean(y)-np.mean(p)))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase2',required=True)
    ap.add_argument('--v04',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    phase2=Path(args.phase2); v04=Path(args.v04)
    base=pd.read_csv(next(phase2.rglob('baseline_oos_predictions.csv')))
    local=pd.read_csv(next(v04.rglob('v04_oos_predictions.csv')))
    d=base.merge(local[['game_id','p_under_local_cv']],on='game_id',how='inner',validate='one_to_one')
    d['actual_under']=1-d['actual_over']
    d['p_under_p100']=1-d['baseline_prediction']
    bo=np.clip(d['baseline_prediction'].to_numpy(float),1e-6,1-1e-6)
    co=np.clip(d['context_prediction'].to_numpy(float),1e-6,1-1e-6)
    d['ctx_logit_delta']=logit(co)-logit(bo)

    dev=d[d.season.isin(DEV_SEASONS)].copy(); val=d[d.season==VAL_SEASON].copy()
    cv=[]
    for held in DEV_SEASONS:
        tr=dev[dev.season!=held]; va=dev[dev.season==held]
        for sh in SHRINKS:
            spline,_=fit_prior(tr,sh)
            for a in ALPHAS:
                p=predict(spline,va,a)
                cv.append({'heldout_season':held,'shrink':sh,'alpha':a,**metrics(va.actual_under.to_numpy(float),p)})
    cv=pd.DataFrame(cv)
    agg=cv.groupby(['shrink','alpha']).agg(mean_logloss=('logloss','mean'),sd_logloss=('logloss','std'),mean_brier=('brier','mean')).reset_index()
    agg['robust_score']=agg.mean_logloss+0.5*agg.sd_logloss
    best=agg.sort_values(['robust_score','mean_logloss','mean_brier']).iloc[0]
    best_sh=int(best.shrink); best_alpha=float(best.alpha)

    spline,buckets=fit_prior(dev,best_sh)
    p_val=predict(spline,val,best_alpha)
    p_dev=predict(spline,dev,best_alpha)

    summary=[]
    for split,frame,p_smooth in [('DEV_2022_2024',dev,p_dev),('VAL_2025',val,p_val)]:
        y=frame.actual_under.to_numpy(float)
        for name,p in [('SMOOTH_CONTINUOUS_PRIOR',p_smooth),('V04_LOCAL_CV',frame.p_under_local_cv.to_numpy(float)),('P100',frame.p_under_p100.to_numpy(float))]:
            summary.append({'split':split,'model':name,**metrics(y,p)})
    summary=pd.DataFrame(summary)

    # Distribution support diagnostics: 1-pp bins. This is descriptive, not a model-selection criterion.
    dist=[]
    for name,p in [('SMOOTH_CONTINUOUS_PRIOR',p_val),('V04_LOCAL_CV',val.p_under_local_cv.to_numpy(float))]:
        pp=100*np.asarray(p)
        for lo in range(40,75):
            n=int(((pp>=lo)&(pp<lo+1)).sum())
            dist.append({'model':name,'bin_lo':lo,'bin_hi':lo+1,'n':n})
    dist=pd.DataFrame(dist)

    cv.to_csv(out/'cv_fold_metrics.csv',index=False)
    agg.to_csv(out/'cv_candidate_summary.csv',index=False)
    buckets.to_csv(out/'fitted_total_curve_dev.csv',index=False)
    summary.to_csv(out/'performance_summary.csv',index=False)
    dist.to_csv(out/'validation_probability_support.csv',index=False)
    pd.DataFrame({'game_id':val.game_id,'season':val.season,'opening_total':val.opening_total,'actual_under':val.actual_under,'p_under_smooth':p_val,'p_under_v04':val.p_under_local_cv,'p_under_p100':val.p_under_p100}).to_csv(out/'validation_predictions.csv',index=False)

    manifest={
      'experiment':'I2 continuous structural-prior smoothing',
      'price_used':False,
      'development_seasons':DEV_SEASONS,
      'validation_season':VAL_SEASON,
      'selected_shrink':best_sh,
      'selected_context_alpha':best_alpha,
      'selection_rule':'minimize mean dev-season logloss + 0.5*SD(logloss)',
      'method':'Fit total->P(Under) on development data using monotone isotonic bucket estimates, then PCHIP interpolation; optionally apply scaled continuous prior-date team-context logit delta. Hyperparameters selected only on 2022-2024 folds; 2025 evaluated once.',
      'important_limitation':'Historical strict replay contains discrete opening_total but does not contain the live parent model decimal projected full-game run total. This tests smoothing the current total anchor and continuous context adjustment; it does not yet test replacing the anchor with the live baseball-only decimal projected total.',
      'selected_cv':best.to_dict(),
      'validation':summary[summary.split=='VAL_2025'].to_dict(orient='records')
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(summary.to_string(index=False))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

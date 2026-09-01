#!/usr/bin/env python3
"""Validate bullpen identity mixtures by how well they recover actual reliever skill.

Development only: 2021-2024. 2025 is never loaded.

The preceding bullpen identity layer showed stable exact-identity ranking in some
innings but not all. For matchup prediction, the object we ultimately need is
not the reliever name itself but the probability-weighted pitcher skill state.
This script therefore evaluates each candidate window/usage blend by comparing
its predicted bullpen skill mixture with the actual first reliever's strictly
prior-date 365-day raw skill vector:
  K rate, BB/HBP baserunner rate, HR rate, non-HR hit rate.

Pitcher skill histories are rebuilt directly from normalized PA events with a
strict [game_date-365d, game_date) cutoff. No shrinkage or same-day history is
used. Candidate reliever probabilities come from the previously materialized
M4 bullpen identity artifact. Missing-history candidate mass is omitted and the
available skill mixture is renormalized; coverage mass is reported explicitly.

Selection uses chronological 2022/2023/2024 folds. Error dimensions are scaled
by the training-fold SD of the actual reliever skill targets so no hand-assigned
metric weights are introduced. This is state/skill-mixture validation only, not
run prediction and not production promotion.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

TEST_YEARS=[2022,2023,2024]
METRICS=['k','baserunner','hr','nonhr_hit']
EVENT_MAP={
 'k':{'strikeout'},
 'baserunner':{'walk','hit_by_pitch'},
 'hr':{'home_run'},
 'nonhr_hit':{'single','double','triple'},
}
EPS=1e-12

def read(p): return pd.read_parquet(p) if Path(p).suffix=='.parquet' else pd.read_csv(p,low_memory=False)

def build_pitcher_daily(pa):
    x=pa.copy(); x['game_date']=pd.to_datetime(x.game_date,errors='coerce').dt.normalize(); ev=x.event.astype(str)
    for m,vals in EVENT_MAP.items(): x[f'n_{m}']=ev.isin(vals).astype('int16')
    cols=[f'n_{m}' for m in METRICS]
    d=x.dropna(subset=['pitcher_id','game_date']).groupby(['pitcher_id','game_date'])[cols].sum().reset_index()
    opp=x.dropna(subset=['pitcher_id','game_date']).groupby(['pitcher_id','game_date']).size().rename('pa').reset_index()
    return d.merge(opp,on=['pitcher_id','game_date'],how='inner')

def build_index(daily):
    out={}
    cols=['pa']+[f'n_{m}' for m in METRICS]
    for pid,g in daily.groupby('pitcher_id',sort=False):
        g=g.sort_values('game_date'); dates=g.game_date.to_numpy(dtype='datetime64[ns]')
        vals=g[cols].to_numpy(float); cs=np.vstack([np.zeros((1,len(cols))),np.cumsum(vals,axis=0)])
        out[pid]=(dates,cs)
    return out

def rates_asof(index,pid,date):
    rec=index.get(pid)
    if rec is None: return None
    dates,cs=rec; date=np.datetime64(pd.Timestamp(date).to_datetime64()); lo=date-np.timedelta64(365,'D')
    a=int(np.searchsorted(dates,lo,side='left')); b=int(np.searchsorted(dates,date,side='left'))
    v=cs[b]-cs[a]; pa=v[0]
    if pa<=0: return None
    return np.array([v[i+1]/pa for i in range(len(METRICS))],float)

def prepare(targets,cands,index):
    tm=targets.copy(); tm['game_date']=pd.to_datetime(tm.game_date,errors='coerce').dt.normalize()
    actual=[]
    for _,r in tm.iterrows(): actual.append(rates_asof(index,r.actual_reliever_id,r.game_date))
    for j,m in enumerate(METRICS): tm[f'actual_{m}']=[np.nan if v is None else float(v[j]) for v in actual]
    # Attach candidate target date once then compute strict-date skill vectors.
    cm=cands.merge(tm[['target_id','window','game_date']],on=['target_id','window'],how='left',validate='many_to_one')
    vals=[]
    for _,r in cm.iterrows(): vals.append(rates_asof(index,r.candidate_pitcher_id,r.game_date))
    for j,m in enumerate(METRICS): cm[f'skill_{m}']=[np.nan if v is None else float(v[j]) for v in vals]
    return tm,cm

def evaluate(tm,cm):
    rows=[]; groups={(int(k[0]),int(k[1])):g for k,g in cm.groupby(['target_id','window'],sort=False)}
    specs=tm[['window']].drop_duplicates().sort_values('window').window.astype(int).tolist()
    # Alpha candidates are inferred from identity formulation's known convex blend.
    alphas=[0.0,0.25,0.5,0.75]
    for year in TEST_YEARS:
      train=tm[tm.season<year]
      # Training-only SD scales by inning and metric; fallback all-inning training SD.
      global_sd={m:max(float(train[f'actual_{m}'].std(ddof=0)),1e-6) for m in METRICS}
      for inn in range(2,10):
        tr=train[train.inning==inn]; te=tm[(tm.season==year)&(tm.inning==inn)]
        if len(te)<50: continue
        sds={m:max(float(tr[f'actual_{m}'].std(ddof=0)) if tr[f'actual_{m}'].notna().sum()>20 else global_sd[m],1e-6) for m in METRICS}
        for window in specs:
          b=te[te.window==window]
          if len(b)<50: continue
          for alpha in alphas:
            errs=[]; covs=[]; metric_sq={m:[] for m in METRICS}; metric_abs={m:[] for m in METRICS}
            for _,t in b.iterrows():
              g=groups.get((int(t.target_id),int(window)))
              if g is None or len(g)==0: continue
              base=(1-alpha)*g.global_share.to_numpy(float)+alpha*g.inning_share.to_numpy(float)
              skill=g[[f'skill_{m}' for m in METRICS]].to_numpy(float)
              valid=np.isfinite(skill).all(axis=1); covered=float(base[valid].sum())
              if covered<=EPS: continue
              w=base[valid]/covered; pred=(skill[valid]*w[:,None]).sum(axis=0)
              act=np.array([t[f'actual_{m}'] for m in METRICS],float)
              if not np.isfinite(act).all(): continue
              zsq=[]
              for j,m in enumerate(METRICS):
                e=float(pred[j]-act[j]); metric_sq[m].append(e*e); metric_abs[m].append(abs(e)); zsq.append((e/sds[m])**2)
              errs.append(float(np.mean(zsq))); covs.append(covered)
            if not errs: continue
            row={'test_year':year,'inning':inn,'window':window,'alpha_inning_usage':alpha,'n_scored':len(errs),'mean_candidate_skill_mass_coverage':float(np.mean(covs)),'standardized_vector_mse':float(np.mean(errs))}
            for m in METRICS:
              row[f'{m}_mse']=float(np.mean(metric_sq[m])); row[f'{m}_mae']=float(np.mean(metric_abs[m]))
            rows.append(row)
    return pd.DataFrame(rows)

def summarize(f):
    aggs={'mean_standardized_vector_mse':('standardized_vector_mse','mean'),'worst_year_standardized_vector_mse':('standardized_vector_mse','max'),'mean_skill_mass_coverage':('mean_candidate_skill_mass_coverage','mean')}
    for m in METRICS:
      aggs[f'mean_{m}_mse']=(f'{m}_mse','mean'); aggs[f'mean_{m}_mae']=(f'{m}_mae','mean')
    s=f.groupby(['inning','window','alpha_inning_usage'],as_index=False).agg(**aggs)
    best=(s.sort_values(['inning','mean_standardized_vector_mse'],ascending=[True,True]).groupby('inning',as_index=False).head(1).sort_values('inning'))
    return s,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plate-appearances',type=Path,required=True); ap.add_argument('--targets',type=Path,required=True); ap.add_argument('--candidates',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa=read(a.plate_appearances); pa['season']=pd.to_datetime(pa.game_date,errors='coerce').dt.year
    if set(pa.season.dropna().astype(int).unique())!={2021,2022,2023,2024}: raise RuntimeError('development seasons must be exactly 2021-2024')
    if (pa.season>=2025).any(): raise RuntimeError('2025 leakage')
    tm=read(a.targets); cm=read(a.candidates)
    daily=build_pitcher_daily(pa); idx=build_index(daily); tm,cm=prepare(tm,cm,idx); folds=evaluate(tm,cm); summary,best=summarize(folds)
    folds.to_csv(a.output_dir/'m4_bullpen_skill_mixture_folds.csv',index=False); summary.to_csv(a.output_dir/'m4_bullpen_skill_mixture_summary.csv',index=False); best.to_csv(a.output_dir/'m4_bullpen_skill_mixture_best_by_inning.csv',index=False)
    tm.to_parquet(a.output_dir/'m4_bullpen_skill_targets.parquet',index=False)
    manifest={'status':'PASS','architecture':'M4_bullpen_skill_mixture_fidelity','development_seasons':[2021,2022,2023,2024],'test_folds':TEST_YEARS,'holdout_season':2025,'holdout_opened':False,'skill_window_days':365,'skill_metrics':METRICS,'skill_rates':'raw_strictly_prior_date','same_day_history_included':False,'market_data_used':False,'metric_weighting':'training-fold SD standardization only; no hand weights','candidate_skill_missing_handling':'renormalize available probability mass and report coverage','best_skill_mixture_spec_by_inning':best.to_dict('records'),'automatic_production_promotion':False,'note':'Validates probability-weighted bullpen quality state, not exact identity and not runs.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Build and validate the M2 inning-specific starter/bullpen state layer.

Development only (2021-2024). Historical starter identity is retrospective/Tier-B
for research; all state-history statistics are strictly prior-date and exclude
same-day earlier games.

For each team pitching half-inning I1-I9, targets are:
- starter_begins_inning: actual first pitcher of the half is the game's starter;
- starter_pa_share: share of PAs in that half handled by the starter.

Candidate pregame state signals are raw prior starter-history rates over season,
30d, 90d and 365d windows. No smoothing/shrinkage is introduced. Missing player
history falls back only to training-fold inning baselines during validation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS=["season","30d","90d","365d"]
TEST_YEARS=[2022,2023,2024]
EPS=1e-12

def read(p):
    p=Path(p); return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,low_memory=False)
def find_one(root,name):
    hits=list(Path(root).rglob(name))
    if len(hits)!=1: raise RuntimeError(f'{name} expected once under {root}; found {hits}')
    return hits[0]
def logloss(y,p):
    p=np.clip(p,EPS,1-EPS); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def brier(y,p): return float(np.mean((p-y)**2))

def build_targets(root):
    pa_parts=[]; st_parts=[]
    for season in [2021,2022,2023,2024]:
        d=Path(root)/f'normalized-mlb-{season}'
        pa=read(find_one(d,'plate_appearances.parquet')); pa['season']=season; pa_parts.append(pa)
        st=read(find_one(d,'starters.parquet')); st['season']=season; st_parts.append(st)
    pa=pd.concat(pa_parts,ignore_index=True); st=pd.concat(st_parts,ignore_index=True)
    pa['game_date']=pd.to_datetime(pa.game_date,errors='coerce').dt.normalize(); st['game_date']=pd.to_datetime(st.game_date,errors='coerce').dt.normalize()
    pa=pa[pd.to_numeric(pa.inning,errors='coerce').between(1,9)].copy(); pa['inning']=pd.to_numeric(pa.inning).astype(int)
    st=st[['game_id','game_date','season','team_id','pitcher_id']].dropna(subset=['pitcher_id']).copy()
    st=st.rename(columns={'pitcher_id':'starter_id','team_id':'pitching_team_id'})
    if st.duplicated(['game_id','pitching_team_id']).any(): raise RuntimeError('starter table nonunique game/team')

    # One record per pitching team x inning, using realized PA sequence only to define target.
    pa=pa.sort_values(['game_id','inning','half_inning','play_index'],kind='mergesort')
    grp=pa.groupby(['game_id','game_date','season','inning','pitching_team_id'],sort=False)
    rows=[]
    for key,g in grp:
        game_id,date,season,inning,team=key
        starter_row=st[(st.game_id==game_id)&(st.pitching_team_id==team)]
        if len(starter_row)!=1: continue
        sid=starter_row.iloc[0].starter_id
        total=len(g); starter_pa=int((g.pitcher_id==sid).sum())
        first_pitcher=g.iloc[0].pitcher_id
        rows.append({'game_id':game_id,'game_date':date,'season':int(season),'inning':int(inning),'pitching_team_id':team,
                     'starter_id':sid,'starter_begins_inning':int(first_pitcher==sid),'starter_pa_share':starter_pa/total if total else np.nan,'pa_count':int(total)})
    out=pd.DataFrame(rows)
    return out

def add_asof_history(x):
    # Daily starter outcomes per pitcher; all same-date starts share one pre-day history state.
    daily=(x.groupby(['starter_id','game_date','season','inning'],as_index=False)
           .agg(starts=('game_id','nunique'),begins=('starter_begins_inning','sum'),share_sum=('starter_pa_share','sum')))
    results=x.copy()
    for inning in range(1,10):
        di=daily[daily.inning==inning].copy().sort_values(['starter_id','game_date'])
        # season cumulative, strictly earlier date because rows are daily aggregates and shift(1)
        di['season_starts_prior']=di.groupby(['starter_id','season']).starts.transform(lambda s:s.shift(1).fillna(0).cumsum())
        di['season_begins_prior']=di.groupby(['starter_id','season']).begins.transform(lambda s:s.shift(1).fillna(0).cumsum())
        di['season_share_prior']=di.groupby(['starter_id','season']).share_sum.transform(lambda s:s.shift(1).fillna(0).cumsum())
        for days in [30,90,365]:
            pieces=[]
            for pid,g in di.groupby('starter_id',sort=False):
                g=g.sort_values('game_date'); z=g.set_index('game_date')[['starts','begins','share_sum']].rolling(f'{days}D',closed='left').sum().fillna(0).reset_index()
                z['starter_id']=pid; pieces.append(z)
            r=pd.concat(pieces,ignore_index=True).rename(columns={'starts':f'{days}d_starts_prior','begins':f'{days}d_begins_prior','share_sum':f'{days}d_share_prior'})
            di=di.merge(r,on=['starter_id','game_date'],how='left')
        keep=['starter_id','game_date']
        for w in WINDOWS:
            prefix='season' if w=='season' else w
            keep += [f'{prefix}_starts_prior',f'{prefix}_begins_prior',f'{prefix}_share_prior']
        di=di[keep].copy(); di['inning']=inning
        results=results.merge(di,on=['starter_id','game_date','inning'],how='left',validate='many_to_one')
    for w in WINDOWS:
        prefix='season' if w=='season' else w
        den=pd.to_numeric(results[f'{prefix}_starts_prior'],errors='coerce')
        results[f'{w}_starter_begin_rate']=pd.to_numeric(results[f'{prefix}_begins_prior'],errors='coerce')/den.replace(0,np.nan)
        results[f'{w}_starter_pa_share_mean']=pd.to_numeric(results[f'{prefix}_share_prior'],errors='coerce')/den.replace(0,np.nan)
    return results

def validate(x):
    rows=[]
    for window in WINDOWS:
        col=f'{window}_starter_begin_rate'
        for inning in range(2,10):
            for year in TEST_YEARS:
                tr=x[(x.season<year)&(x.inning==inning)].copy(); te=x[(x.season==year)&(x.inning==inning)].copy()
                if len(te)<100: continue
                base=float(tr.starter_begins_inning.mean())
                p=np.where(te[col].notna(),te[col],base).astype(float)
                y=te.starter_begins_inning.to_numpy(float)
                p0=np.full(len(te),base)
                rows.append({'window':window,'inning':inning,'test_year':year,'n_test':int(len(te)),'history_coverage':float(te[col].notna().mean()),
                             'baseline_logloss':logloss(y,p0),'model_logloss':logloss(y,p),'baseline_brier':brier(y,p0),'model_brier':brier(y,p),
                             'mean_predicted_starter_begin':float(np.mean(p)),'actual_starter_begin':float(np.mean(y))})
    r=pd.DataFrame(rows); r['logloss_improvement']=r.baseline_logloss-r.model_logloss; r['brier_improvement']=r.baseline_brier-r.model_brier
    s=(r.groupby(['window','inning'],as_index=False).agg(mean_logloss_improvement=('logloss_improvement','mean'),worst_year_logloss_improvement=('logloss_improvement','min'),
        mean_brier_improvement=('brier_improvement','mean'),worst_year_brier_improvement=('brier_improvement','min'),mean_history_coverage=('history_coverage','mean')))
    best=(s.sort_values(['inning','mean_logloss_improvement'],ascending=[True,False]).groupby('inning',as_index=False).head(1).sort_values('inning'))
    return r,s,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--normalized-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=build_targets(a.normalized_root)
    if set(x.season.unique())!={2021,2022,2023,2024}: raise RuntimeError('development seasons incomplete')
    if (x.season>=2025).any(): raise RuntimeError('2025 leakage')
    x=add_asof_history(x)
    folds,summary,best=validate(x)
    x.to_parquet(a.output_dir/'m2_pitcher_state_matrix.parquet',index=False); folds.to_csv(a.output_dir/'m2_pitcher_state_fold_results.csv',index=False)
    summary.to_csv(a.output_dir/'m2_pitcher_state_summary.csv',index=False); best.to_csv(a.output_dir/'m2_pitcher_state_best_window_by_inning.csv',index=False)
    state=x.groupby('inning',as_index=False).agg(n=('game_id','size'),starter_begin_rate=('starter_begins_inning','mean'),starter_pa_share=('starter_pa_share','mean'))
    state.to_csv(a.output_dir/'m2_empirical_state_by_inning.csv',index=False)
    manifest={'status':'PASS','architecture':'M2_pitcher_state_identification','development_seasons':[2021,2022,2023,2024],'test_folds':TEST_YEARS,
      'holdout_season':2025,'holdout_opened':False,'innings':list(range(1,10)),'targets':['starter_begins_inning','starter_pa_share'],
      'candidate_windows':WINDOWS,'history_timing':'strictly_prior_date_same_day_excluded','smoothing_used':False,'shrinkage_used':False,'market_data_used':False,
      'starter_identity_class':'retrospective_actual_first_pitcher_unverified_pregame_Tier_B','automatic_production_promotion':False,
      'best_raw_history_window_by_inning_for_begin_target':best.to_dict('records'),
      'note':'This validates starter-vs-bullpen state identification, not inning run prediction. I1 is structurally starter by actual-first-pitcher definition; window comparison begins at I2.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

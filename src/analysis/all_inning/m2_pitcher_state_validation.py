#!/usr/bin/env python3
"""Validate M2 starter/bullpen state identification for innings I1-I9.

2021-2024 development only. Historical starter identity is Tier-B retrospective;
all history is strictly prior-date and same-day earlier games are excluded.
No smoothing, shrinkage, market data, or 2025 outcomes are used.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
WINDOWS=['season','30d','90d','365d']; TEST_YEARS=[2022,2023,2024]; EPS=1e-12

def read(p): p=Path(p); return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,low_memory=False)
def one(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f'{name}: {h}')
    return h[0]
def ll(y,p):
    p=np.clip(p,EPS,1-EPS); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def br(y,p): return float(np.mean((p-y)**2))

def build_targets(root):
    pp=[]; ss=[]
    for season in [2021,2022,2023,2024]:
        d=Path(root)/f'normalized-mlb-{season}'
        p=read(one(d,'plate_appearances.parquet')); p['season']=season; pp.append(p)
        s=read(one(d,'starters.parquet')); s['season']=season; ss.append(s)
    pa=pd.concat(pp,ignore_index=True); st=pd.concat(ss,ignore_index=True)
    pa['game_date']=pd.to_datetime(pa.game_date,errors='coerce').dt.normalize(); st['game_date']=pd.to_datetime(st.game_date,errors='coerce').dt.normalize()
    pa=pa[pd.to_numeric(pa.inning,errors='coerce').between(1,9)].copy(); pa['inning']=pd.to_numeric(pa.inning).astype(int)
    st=st[['game_id','game_date','season','team_id','pitcher_id']].dropna(subset=['pitcher_id']).rename(columns={'team_id':'pitching_team_id','pitcher_id':'starter_id'})
    if st.duplicated(['game_id','pitching_team_id']).any(): raise RuntimeError('starter key nonunique')
    pa=pa.merge(st[['game_id','pitching_team_id','starter_id']],on=['game_id','pitching_team_id'],how='inner',validate='many_to_one')
    pa=pa.sort_values(['game_id','inning','half_inning','play_index'],kind='mergesort')
    pa['is_starter_pa']=(pa.pitcher_id==pa.starter_id).astype(int)
    first=(pa.groupby(['game_id','game_date','season','inning','pitching_team_id','starter_id'],sort=False).first().reset_index()[['game_id','game_date','season','inning','pitching_team_id','starter_id','pitcher_id']]
           .rename(columns={'pitcher_id':'first_pitcher_id'}))
    agg=(pa.groupby(['game_id','game_date','season','inning','pitching_team_id','starter_id'],as_index=False).agg(pa_count=('play_index','size'),starter_pa=('is_starter_pa','sum')))
    out=agg.merge(first,on=['game_id','game_date','season','inning','pitching_team_id','starter_id'],how='left',validate='one_to_one')
    out['starter_begins_inning']=(out.first_pitcher_id==out.starter_id).astype(int); out['starter_pa_share']=out.starter_pa/out.pa_count
    return out.drop(columns=['first_pitcher_id'])

def add_history(x):
    daily=(x.groupby(['starter_id','game_date','season','inning'],as_index=False).agg(starts=('game_id','nunique'),begins=('starter_begins_inning','sum'),share_sum=('starter_pa_share','sum')))
    pieces=[]
    for inning in range(1,10):
        d=daily[daily.inning==inning].sort_values(['starter_id','game_date']).copy()
        for c in ['starts','begins','share_sum']:
            d[f'season_{c}_prior']=d.groupby(['starter_id','season'])[c].transform(lambda s:s.shift(1).fillna(0).cumsum())
        base=d[['starter_id','game_date','inning','season_starts_prior','season_begins_prior','season_share_sum_prior']].copy()
        for days in [30,90,365]:
            rr=[]
            for pid,g in d.groupby('starter_id',sort=False):
                z=(g.sort_values('game_date').set_index('game_date')[['starts','begins','share_sum']].rolling(f'{days}D',closed='left').sum().fillna(0).reset_index())
                z['starter_id']=pid; rr.append(z)
            r=pd.concat(rr,ignore_index=True).rename(columns={'starts':f'{days}d_starts_prior','begins':f'{days}d_begins_prior','share_sum':f'{days}d_share_sum_prior'})
            base=base.merge(r,on=['starter_id','game_date'],how='left')
        pieces.append(base)
    hist=pd.concat(pieces,ignore_index=True)
    out=x.merge(hist,on=['starter_id','game_date','inning'],how='left',validate='many_to_one')
    for w in WINDOWS:
        p='season' if w=='season' else w; den=pd.to_numeric(out[f'{p}_starts_prior'],errors='coerce')
        out[f'{w}_starter_begin_rate']=pd.to_numeric(out[f'{p}_begins_prior'],errors='coerce')/den.replace(0,np.nan)
        out[f'{w}_starter_pa_share_mean']=pd.to_numeric(out[f'{p}_share_sum_prior'],errors='coerce')/den.replace(0,np.nan)
    return out

def validate(x):
    rows=[]
    for w in WINDOWS:
        c=f'{w}_starter_begin_rate'
        for inn in range(2,10):
            for yr in TEST_YEARS:
                tr=x[(x.season<yr)&(x.inning==inn)]; te=x[(x.season==yr)&(x.inning==inn)]
                if len(te)<100: continue
                base=float(tr.starter_begins_inning.mean()); y=te.starter_begins_inning.to_numpy(float); p=np.where(te[c].notna(),te[c],base).astype(float); p0=np.full(len(te),base)
                rows.append({'window':w,'inning':inn,'test_year':yr,'n_test':int(len(te)),'history_coverage':float(te[c].notna().mean()),'baseline_logloss':ll(y,p0),'model_logloss':ll(y,p),'baseline_brier':br(y,p0),'model_brier':br(y,p),'mean_predicted_starter_begin':float(p.mean()),'actual_starter_begin':float(y.mean())})
    r=pd.DataFrame(rows); r['logloss_improvement']=r.baseline_logloss-r.model_logloss; r['brier_improvement']=r.baseline_brier-r.model_brier
    s=r.groupby(['window','inning'],as_index=False).agg(mean_logloss_improvement=('logloss_improvement','mean'),worst_year_logloss_improvement=('logloss_improvement','min'),mean_brier_improvement=('brier_improvement','mean'),worst_year_brier_improvement=('brier_improvement','min'),mean_history_coverage=('history_coverage','mean'))
    best=s.sort_values(['inning','mean_logloss_improvement'],ascending=[True,False]).groupby('inning',as_index=False).head(1).sort_values('inning')
    return r,s,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--normalized-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=add_history(build_targets(a.normalized_root))
    if set(x.season.unique())!={2021,2022,2023,2024} or (x.season>=2025).any(): raise RuntimeError('season governance failed')
    folds,summary,best=validate(x)
    x.to_parquet(a.output_dir/'m2_pitcher_state_matrix.parquet',index=False); folds.to_csv(a.output_dir/'m2_pitcher_state_fold_results.csv',index=False); summary.to_csv(a.output_dir/'m2_pitcher_state_summary.csv',index=False); best.to_csv(a.output_dir/'m2_pitcher_state_best_window_by_inning.csv',index=False)
    state=x.groupby('inning',as_index=False).agg(n=('game_id','size'),starter_begin_rate=('starter_begins_inning','mean'),starter_pa_share=('starter_pa_share','mean')); state.to_csv(a.output_dir/'m2_empirical_state_by_inning.csv',index=False)
    j={'status':'PASS','architecture':'M2_pitcher_state_identification','development_seasons':[2021,2022,2023,2024],'test_folds':TEST_YEARS,'holdout_season':2025,'holdout_opened':False,'innings':list(range(1,10)),'targets':['starter_begins_inning','starter_pa_share'],'candidate_windows':WINDOWS,'history_timing':'strictly_prior_date_same_day_excluded','smoothing_used':False,'shrinkage_used':False,'market_data_used':False,'starter_identity_class':'retrospective_actual_first_pitcher_unverified_pregame_Tier_B','automatic_production_promotion':False,'empirical_state_by_inning':state.to_dict('records'),'best_raw_history_window_by_inning_for_begin_target':best.to_dict('records'),'note':'State identification only; not a run-outcome challenger. I1 is structural starter state and I2-I9 are validated.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(j,indent=2),encoding='utf-8'); print(json.dumps(j,indent=2))
if __name__=='__main__': main()

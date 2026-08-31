#!/usr/bin/env python3
"""Build game-level M2 starter-retention + bullpen research features.

All features are reconstructed from completed historical baseball events strictly
before the target game date. Same-day prior games are deliberately excluded.
No betting-market information is used here. No shrinkage strength is assumed:
raw rates/support are emitted and window choice is left to development-only
validation.

M2 is layered on the already-built M1 matrix. Candidate families:
- starter I2 retention history: 30/90/365 day raw rates
- team bullpen quality: 30/90/365 day raw relief event rates
- team bullpen workload/availability proxy: prior 1/2/3/7/14 day relief BF

The two defensive team contexts are averaged to one full-I2 game row because the
outcome is total runs across Top 2 + Bottom 2. No nonlinear transform is imposed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

RETENTION_WINDOWS=(30,90,365)
QUALITY_WINDOWS=(30,90,365)
WORKLOAD_WINDOWS=(1,2,3,7,14)


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,low_memory=False)


def prior_window_sums(history: pd.DataFrame, targets: pd.DataFrame, entity_col: str,
                      history_date_col: str, target_date_col: str,
                      value_cols: list[str], windows: tuple[int,...]) -> pd.DataFrame:
    """Exact calendar-day sums using history dates strictly before target date."""
    out=targets[[entity_col,target_date_col]].copy().reset_index().rename(columns={'index':'_target_index'})
    for w in windows:
        for c in value_cols:
            out[f'{w}d_{c}']=0.0
        out[f'{w}d_support_rows']=0

    h=history.dropna(subset=[entity_col,history_date_col]).copy()
    h[history_date_col]=pd.to_datetime(h[history_date_col],errors='coerce').dt.normalize()
    t=out.dropna(subset=[entity_col,target_date_col]).copy()
    t[target_date_col]=pd.to_datetime(t[target_date_col],errors='coerce').dt.normalize()

    hgroups={k:g.sort_values(history_date_col) for k,g in h.groupby(entity_col,sort=False)}
    for ent,tg in t.groupby(entity_col,sort=False):
        hg=hgroups.get(ent)
        if hg is None or hg.empty:
            continue
        hd=hg[history_date_col].to_numpy(dtype='datetime64[ns]')
        vals={c:pd.to_numeric(hg[c],errors='coerce').fillna(0.0).to_numpy(float) for c in value_cols}
        cums={c:np.r_[0.0,np.cumsum(v)] for c,v in vals.items()}
        td=tg[target_date_col].to_numpy(dtype='datetime64[ns]')
        right=np.searchsorted(hd,td,side='left')  # excludes all same-date observations
        target_idx=tg.index.to_numpy()
        for w in windows:
            left_dates=td-np.timedelta64(int(w),'D')
            left=np.searchsorted(hd,left_dates,side='left')
            for c in value_cols:
                out.loc[target_idx,f'{w}d_{c}']=cums[c][right]-cums[c][left]
            out.loc[target_idx,f'{w}d_support_rows']=right-left
    return out.sort_values('_target_index').drop(columns=['_target_index']).reset_index(drop=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--m1-matrix',type=Path,required=True)
    ap.add_argument('--plate-appearances',type=Path,required=True)
    ap.add_argument('--starters',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)

    m1=read(a.m1_matrix).copy()
    pa=read(a.plate_appearances).copy()
    st=read(a.starters).copy()
    for d,c in [(m1,'game_date'),(pa,'game_date'),(st,'game_date')]:
        d[c]=pd.to_datetime(d[c],errors='coerce').dt.normalize()

    need_st=['game_id','game_date','team_id','pitcher_id']
    miss=[c for c in need_st if c not in st.columns]
    if miss: raise RuntimeError(f'starters missing {miss}')
    st=st[need_st].dropna(subset=['game_id','team_id','pitcher_id']).drop_duplicates(['game_id','team_id']).copy()
    st=st[st.game_id.isin(m1.game_id)].copy()
    if st.duplicated(['game_id','team_id']).any(): raise RuntimeError('starter table non-unique by game/team')

    # Classify each pitching PA as starter or relief using only final historical identity.
    need_pa=['game_id','game_date','inning','pitching_team_id','pitcher_id','event']
    miss=[c for c in need_pa if c not in pa.columns]
    if miss: raise RuntimeError(f'plate appearances missing {miss}')
    x=pa[need_pa].copy()
    x=x.merge(st[['game_id','team_id','pitcher_id']].rename(columns={'team_id':'pitching_team_id','pitcher_id':'starter_id'}),
              on=['game_id','pitching_team_id'],how='inner',validate='many_to_one')
    x['is_starter']=(x.pitcher_id==x.starter_id)
    x['is_relief']=~x['is_starter']
    x['inning']=pd.to_numeric(x['inning'],errors='coerce')
    x['ev_k']=(x.event.astype(str)=='strikeout').astype(int)
    x['ev_bb']=(x.event.astype(str)=='walk').astype(int)
    x['ev_hr']=(x.event.astype(str)=='home_run').astype(int)
    x['ev_hit']=x.event.astype(str).isin(['single','double','triple','home_run']).astype(int)
    x['bf']=1

    # Historical starter I2 retention outcomes, one row per starter outing.
    i2=x[x.inning==2].copy()
    ret=(i2.groupby(['game_id','game_date','pitching_team_id','starter_id'],as_index=False)
           .agg(i2_total_bf=('bf','sum'),i2_starter_bf=('is_starter','sum')))
    ret['starter_i2_reached']=(ret.i2_starter_bf>0).astype(float)
    ret['starter_i2_share']=np.where(ret.i2_total_bf>0,ret.i2_starter_bf/ret.i2_total_bf,np.nan)
    ret=ret.rename(columns={'starter_id':'pitcher_id'})

    target=st[['game_id','game_date','team_id','pitcher_id']].copy().reset_index(drop=True)
    rsum=prior_window_sums(
        ret.rename(columns={'game_date':'history_date'}),
        target[['pitcher_id','game_date']].rename(columns={'game_date':'target_date'}),
        'pitcher_id','history_date','target_date',
        ['starter_i2_reached','starter_i2_share'],RETENTION_WINDOWS)
    for w in RETENTION_WINDOWS:
        n=pd.to_numeric(rsum[f'{w}d_support_rows'],errors='coerce').fillna(0.0)
        target[f'starter_i2_reached_rate_{w}d']=np.where(n>0,rsum[f'{w}d_starter_i2_reached']/n,np.nan)
        target[f'starter_i2_share_{w}d']=np.where(n>0,rsum[f'{w}d_starter_i2_share']/n,np.nan)
        target[f'starter_retention_starts_{w}d']=n

    # Relief history by team/date. Target dates come from every game/team, so feature
    # existence cannot leak whether the bullpen was actually used in the target game.
    relief=x[x.is_relief].copy()
    daily=(relief.groupby(['pitching_team_id','game_date'],as_index=False)
                 .agg(bf=('bf','sum'),k=('ev_k','sum'),bb=('ev_bb','sum'),hr=('ev_hr','sum'),hit=('ev_hit','sum')))
    daily['nonhr_hit']=daily['hit']-daily['hr']
    daily=daily.rename(columns={'pitching_team_id':'team_id','game_date':'history_date'})

    qsum=prior_window_sums(
        daily,target[['team_id','game_date']].rename(columns={'game_date':'target_date'}),
        'team_id','history_date','target_date',['bf','k','bb','hr','nonhr_hit'],QUALITY_WINDOWS)
    for w in QUALITY_WINDOWS:
        bf=pd.to_numeric(qsum[f'{w}d_bf'],errors='coerce').fillna(0.0)
        for metric in ['k','bb','hr','nonhr_hit']:
            target[f'bullpen_{metric}_rate_{w}d']=np.where(bf>0,qsum[f'{w}d_{metric}']/bf,np.nan)
        target[f'bullpen_quality_bf_{w}d']=bf

    wsum=prior_window_sums(
        daily,target[['team_id','game_date']].rename(columns={'game_date':'target_date'}),
        'team_id','history_date','target_date',['bf'],WORKLOAD_WINDOWS)
    for w in WORKLOAD_WINDOWS:
        target[f'bullpen_bf_{w}d']=pd.to_numeric(wsum[f'{w}d_bf'],errors='coerce').fillna(0.0)

    feature_cols=[]
    for w in RETENTION_WINDOWS:
        feature_cols += [f'starter_i2_reached_rate_{w}d',f'starter_i2_share_{w}d']
    for w in QUALITY_WINDOWS:
        feature_cols += [f'bullpen_k_rate_{w}d',f'bullpen_bb_rate_{w}d',f'bullpen_hr_rate_{w}d',f'bullpen_nonhr_hit_rate_{w}d']
    for w in WORKLOAD_WINDOWS:
        feature_cols += [f'bullpen_bf_{w}d']

    # Full-inning outcome = two half innings, so average the two pitching-team contexts.
    agg=target.groupby('game_id',as_index=False)[feature_cols].mean()
    out=m1.merge(agg,on='game_id',how='left',validate='one_to_one')
    out['m2_statistics_timing_class']='asof_safe_strictly_prior_date_same_day_excluded'
    out['m2_identity_timing_class']='retrospective_actual_starter_unverified_pregame'
    out.to_parquet(a.output,index=False)

    manifest={
      'rows':int(len(out)),
      'm1_rows':int(len(m1)),
      'game_match_rate':float(out[feature_cols].notna().any(axis=1).mean()),
      'retention_windows_days':list(RETENTION_WINDOWS),
      'bullpen_quality_windows_days':list(QUALITY_WINDOWS),
      'bullpen_workload_windows_days':list(WORKLOAD_WINDOWS),
      'm2_candidate_features':feature_cols,
      'feature_nonnull_coverage':{c:float(out[c].notna().mean()) for c in feature_cols},
      'same_day_prior_games_included':False,
      'future_information_in_statistics':False,
      'starter_identity_pregame_verified':False,
      'bullpen_shrinkage_used':False,
      'market_data_added_by_m2':False,
      'aggregation':'arithmetic mean of two pitching-team contexts; no nonlinear transform'
    }
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

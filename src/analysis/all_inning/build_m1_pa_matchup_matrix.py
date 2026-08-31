#!/usr/bin/env python3
"""Build retrospective PA-level matchup-skill research matrix for I1-I9.

M1's job is to validate which batter/pitcher skill dimensions matter conditional
on an actual matchup. Realized batter/pitcher identities are used ONLY as a
retrospective research oracle; all rate features are strictly prior-date. M2/M3
will later estimate pregame participant-state probabilities.

No 2025 rows, market data, shrinkage, or same-day history are used.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

WINDOWS=['season','30d','90d','365d']
METRICS={
  'k':['ev_strikeout'],
  'baserunner':['ev_walk','ev_hit_by_pitch'],
  'hr':['ev_home_run'],
  'nonhr_hit':['ev_single','ev_double','ev_triple'],
}

def read(p):
    p=Path(p); return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,low_memory=False)

def find_one(root,name):
    hits=list(Path(root).rglob(name))
    if len(hits)!=1: raise RuntimeError(f'{name} expected once under {root}; found {hits}')
    return hits[0]

def add_metric_rates(entity, prefix):
    e=entity.copy()
    for metric, parts in METRICS.items():
        cols=[f'{prefix}_{p}_rate_raw' for p in parts]
        missing=[c for c in cols if c not in e.columns]
        if missing: raise RuntimeError(f'entity_asof missing {missing}')
        e[f'{prefix}_{metric}_rate_raw']=e[cols].sum(axis=1,min_count=len(cols))
    keep=['entity_id','as_of_date']+[f'{prefix}_{m}_rate_raw' for m in METRICS]
    return e[keep]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--normalized-root',type=Path,required=True)
    ap.add_argument('--reusable-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)

    parts=[]
    for season in [2021,2022,2023,2024]:
        p=find_one(a.normalized_root/f'normalized-mlb-{season}','plate_appearances.parquet')
        z=read(p); z['season']=season; parts.append(z)
    pa=pd.concat(parts,ignore_index=True)
    pa['game_date']=pd.to_datetime(pa.game_date,errors='coerce').dt.normalize()
    pa=pa[pd.to_numeric(pa.inning,errors='coerce').between(1,9)].copy()
    pa['inning']=pd.to_numeric(pa.inning,errors='coerce').astype(int)
    if (pa.season>=2025).any(): raise RuntimeError('2025 holdout leakage')

    ev=pa['event'].astype(str)
    pa['y_k']=(ev=='strikeout').astype('int8')
    pa['y_baserunner']=ev.isin(['walk','hit_by_pitch']).astype('int8')
    pa['y_hr']=(ev=='home_run').astype('int8')
    pa['y_nonhr_hit']=ev.isin(['single','double','triple']).astype('int8')

    entity=read(find_one(a.reusable_root,'entity_asof.parquet'))
    entity['as_of_date']=pd.to_datetime(entity.as_of_date,errors='coerce').dt.normalize()
    batter=entity[entity.entity_type=='batter'].copy()
    pitcher=entity[entity.entity_type=='pitcher'].copy()

    base=pa[['game_id','game_date','season','inning','half_inning','play_index','batting_team_id','pitching_team_id','batter_id','pitcher_id','batter_side','pitcher_hand','y_k','y_baserunner','y_hr','y_nonhr_hit']].copy()
    base=base.rename(columns={'half_inning':'half'})
    for w in WINDOWS:
        b=add_metric_rates(batter,w).rename(columns={'entity_id':'batter_id','as_of_date':'game_date',**{f'{w}_{m}_rate_raw':f'batter_{w}_{m}_rate' for m in METRICS}})
        p=add_metric_rates(pitcher,w).rename(columns={'entity_id':'pitcher_id','as_of_date':'game_date',**{f'{w}_{m}_rate_raw':f'pitcher_{w}_{m}_rate' for m in METRICS}})
        base=base.merge(b,on=['batter_id','game_date'],how='left',validate='many_to_one')
        base=base.merge(p,on=['pitcher_id','game_date'],how='left',validate='many_to_one')

    base['platoon_same_hand']=np.where(base.batter_side.isin(['L','R'])&base.pitcher_hand.isin(['L','R']),(base.batter_side==base.pitcher_hand).astype(int),np.nan)
    base['statistics_timing_class']='asof_safe_strictly_prior_date'
    base['participant_identity_class']='retrospective_realized_matchup_oracle_not_pregame_feature'
    base['market_data_used']=False
    base=base.sort_values(['game_date','game_id','inning','half','play_index'],kind='mergesort')
    base.to_parquet(a.output,index=False)

    coverage={}
    for w in WINDOWS:
        for m in METRICS:
            cols=[f'batter_{w}_{m}_rate',f'pitcher_{w}_{m}_rate']
            coverage[f'{w}_{m}']=float(base[cols].notna().all(axis=1).mean())
    manifest={
      'status':'PASS','architecture':'M1_PA_level_retrospective_matchup_skill_matrix','rows':int(len(base)),
      'development_seasons':sorted(int(v) for v in base.season.unique()),'innings':sorted(int(v) for v in base.inning.unique()),
      'holdout_season':2025,'holdout_opened':False,'windows_registered_as_candidates':WINDOWS,
      'metrics':list(METRICS),'raw_rates_only':True,'shrunken_rates_used':False,'same_day_prior_games_included':False,
      'participant_identity_class':'retrospective_realized_matchup_oracle_not_pregame_feature',
      'purpose':'validate matchup skill dimensions only; M2/M3 will estimate pregame participant states',
      'market_data_used':False,'coverage':coverage
    }
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

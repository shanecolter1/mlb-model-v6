#!/usr/bin/env python3
"""Build game-level M1 starter + opposing-offense research matrix.

M1 uses only leakage-safe baseball histories plus the isolated pregame full-game
total used by locked M0. Starter identities are retrospective/unverified pregame
(Tier B research), but all statistics are strictly prior-date.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TEAM_ID_TO_CODE={108:'LAA',109:'ARI',110:'BAL',111:'BOS',112:'CHC',113:'CIN',114:'CLE',115:'COL',116:'DET',117:'HOU',118:'KC',119:'LAD',120:'WSH',121:'NYM',133:'OAK',134:'PIT',135:'SD',136:'SEA',137:'SF',138:'STL',139:'TB',140:'TEX',141:'TOR',142:'MIN',143:'PHI',144:'ATL',145:'CHW',146:'MIA',147:'NYY',158:'MIL'}

def read(path):
    path=Path(path)
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,low_memory=False)

def find_one(root,name):
    hits=list(Path(root).rglob(name))
    if len(hits)!=1: raise RuntimeError(f'{name} expected exactly once under {root}; found {[str(x) for x in hits]}')
    return hits[0]

def pick(df,cands):
    for c in cands:
        if c in df.columns:return c
    raise KeyError(f'None of required columns exist: {cands}')

def rate_cols(df):
    return {
      'k':pick(df,['365d_ev_strikeout_rate_raw','365d_ev_strikeout_rate_shrunk','season_ev_strikeout_rate_raw','season_ev_strikeout_rate_shrunk']),
      'bb':pick(df,['365d_ev_walk_rate_raw','365d_ev_walk_rate_shrunk','season_ev_walk_rate_raw','season_ev_walk_rate_shrunk']),
      'hr':pick(df,['365d_ev_home_run_rate_raw','365d_ev_home_run_rate_shrunk','season_ev_home_run_rate_raw','season_ev_home_run_rate_shrunk']),
      'hit':pick(df,['365d_ev_hit_rate_raw','365d_ev_hit_rate_shrunk','season_ev_hit_rate_raw','season_ev_hit_rate_shrunk']),
    }

def norm_code(x):
    if pd.isna(x):return None
    s=str(x).upper().strip(); aliases={'ANA':'LAA','CHA':'CHW','CHN':'CHC','LAN':'LAD','NYA':'NYY','NYN':'NYM','SDN':'SD','SFN':'SF','SLN':'STL','TBA':'TB','KCA':'KC','WAS':'WSH'}
    return aliases.get(s,s)

def add_game_number(gi):
    """Recover Retrosheet game_number without using outcome information.

    Retrosheet uses game_number=0 for a single game and 1/2 for a doubleheader.
    The reusable game index preserves game_datetime but not that field. Same-date,
    same-matchup duplicates are therefore ordered chronologically to recover 1/2;
    singleton matchup-date groups are assigned 0. No score or outcome is used.
    """
    x=gi.copy()
    keys=['game_date','away_team_code','home_team_code']
    if 'game_number' in x.columns:
        x['game_number']=pd.to_numeric(x['game_number'],errors='coerce').astype('Int64')
        return x
    group_size=x.groupby(keys,dropna=False)['game_id'].transform('size')
    if 'game_datetime' not in x.columns:
        if (group_size>1).any():
            raise RuntimeError('game_index has same-date matchup duplicates but lacks game_datetime/game_number')
        x['game_number']=0
        return x
    x['_game_datetime_sort']=pd.to_datetime(x['game_datetime'],errors='coerce',utc=True)
    x=x.sort_values(keys+['_game_datetime_sort','game_id'],kind='mergesort').copy()
    group_size=x.groupby(keys,dropna=False)['game_id'].transform('size')
    seq=x.groupby(keys,dropna=False).cumcount()+1
    x['game_number']=np.where(group_size.eq(1),0,seq).astype(int)
    return x.drop(columns=['_game_datetime_sort'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--master',type=Path,required=True); ap.add_argument('--artifact-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    m=read(a.master)
    need=['game_date','away_team_code','home_team_code','game_number','dk_total_open_total','inning2_total_runs']; miss=[c for c in need if c not in m.columns]
    if miss: raise RuntimeError(f'Historical master missing {miss}')
    m=m[need].copy(); m.game_date=pd.to_datetime(m.game_date,errors='coerce').dt.normalize(); m.away_team_code=m.away_team_code.map(norm_code); m.home_team_code=m.home_team_code.map(norm_code)
    m.game_number=pd.to_numeric(m.game_number,errors='coerce').astype('Int64')
    m.dk_total_open_total=pd.to_numeric(m.dk_total_open_total,errors='coerce'); m.inning2_total_runs=pd.to_numeric(m.inning2_total_runs,errors='coerce'); m=m[m.dk_total_open_total.notna()&m.inning2_total_runs.notna()].copy()

    gi=read(find_one(a.artifact_root,'game_index.parquet')); st=read(find_one(a.artifact_root,'starter_asof.parquet')); ent=read(find_one(a.artifact_root,'entity_asof.parquet')); team=read(find_one(a.artifact_root,'team_asof.parquet'))
    for d,c in [(gi,'game_date'),(st,'game_date'),(ent,'as_of_date'),(team,'as_of_date')]: d[c]=pd.to_datetime(d[c],errors='coerce').dt.normalize()
    gi['away_team_code']=pd.to_numeric(gi.away_team_id,errors='coerce').map(TEAM_ID_TO_CODE); gi['home_team_code']=pd.to_numeric(gi.home_team_id,errors='coerce').map(TEAM_ID_TO_CODE)
    gi=add_game_number(gi)
    idx=gi[['game_id','game_date','game_number','away_team_id','home_team_id','away_team_code','home_team_code']].copy()
    join_keys=['game_date','away_team_code','home_team_code','game_number']
    if idx.duplicated(join_keys).any():
        raise RuntimeError('game_index remains non-unique after Retrosheet game-number reconstruction')
    if m.duplicated(join_keys).any():
        raise RuntimeError('historical master is non-unique on date/teams/game_number')
    joined=m.merge(idx,on=join_keys,how='inner',validate='one_to_one')
    if len(joined)!=len(m):
        missing=m.merge(idx[join_keys],on=join_keys,how='left',indicator=True)
        missing=missing[missing['_merge']=='left_only'][join_keys]
        raise RuntimeError(f'Historical/game-index join incomplete: matched {len(joined)} of {len(m)} eligible master rows; first missing keys={missing.head(20).to_dict("records")}')

    pr=rate_cols(ent[ent.entity_type=='pitcher']); br=rate_cols(team[team.team_role=='batting'])
    p=ent[ent.entity_type=='pitcher'][['entity_id','as_of_date',*pr.values()]].copy().rename(columns={'entity_id':'pitcher_id','as_of_date':'game_date',pr['k']:'starter_k',pr['bb']:'starter_bb',pr['hr']:'starter_hr',pr['hit']:'starter_hit'})
    s=st[['game_id','game_date','team_side','team_id','pitcher_id','identity_timing_class','statistics_timing_class']].merge(p,on=['pitcher_id','game_date'],how='left',validate='many_to_one')
    b=team[team.team_role=='batting'][['team_id','as_of_date',*br.values()]].copy().rename(columns={'as_of_date':'game_date',br['k']:'off_k',br['bb']:'off_bb',br['hr']:'off_hr',br['hit']:'off_hit'})

    halves=[]
    for bat_side,pit_side,team_id_col in [('away','home','away_team_id'),('home','away','home_team_id')]:
        base=joined[['game_id','game_date','game_number','away_team_code','home_team_code','away_team_id','home_team_id','dk_total_open_total','inning2_total_runs']].copy()
        off=base[['game_id','game_date',team_id_col]].rename(columns={team_id_col:'team_id'}).merge(b,on=['team_id','game_date'],how='left',validate='many_to_one')
        pit=s[s.team_side==pit_side].copy()
        h=base.merge(off[['game_id','off_k','off_bb','off_hr','off_hit']],on='game_id',how='left',validate='one_to_one').merge(pit[['game_id','starter_k','starter_bb','starter_hr','starter_hit']],on='game_id',how='left',validate='one_to_one'); h['half']=bat_side; halves.append(h)
    h=pd.concat(halves,ignore_index=True)
    for c in ['starter_k','starter_bb','starter_hr','starter_hit','off_k','off_bb','off_hr','off_hit']: h[c]=pd.to_numeric(h[c],errors='coerce')
    h['starter_nonhr_hit']=h.starter_hit-h.starter_hr; h['off_nonhr_hit']=h.off_hit-h.off_hr
    h['contact_interaction_half']=(1-h.starter_k)*(1-h.off_k); h['power_interaction_half']=h.starter_hr*h.off_hr; h['baserunner_interaction_half']=h.starter_bb*h.off_bb
    keys=['game_id','game_date','game_number','away_team_code','home_team_code','dk_total_open_total','inning2_total_runs']
    agg=h.groupby(keys,as_index=False).agg(starter_k_rate=('starter_k','mean'),starter_bb_rate=('starter_bb','mean'),starter_hr_rate=('starter_hr','mean'),starter_nonhr_hit_rate=('starter_nonhr_hit','mean'),opponent_k_rate=('off_k','mean'),opponent_bb_rate=('off_bb','mean'),opponent_hr_rate=('off_hr','mean'),opponent_nonhr_hit_rate=('off_nonhr_hit','mean'),contact_interaction=('contact_interaction_half','mean'),power_interaction=('power_interaction_half','mean'),baserunner_interaction=('baserunner_interaction_half','mean'))
    agg['season']=agg.game_date.dt.year.astype(int); agg['starter_identity_timing_class']='retrospective_actual_first_pitcher_unverified_pregame'; agg['statistics_timing_class']='asof_safe_strictly_prior_date'; agg['market_columns_retained']='dk_total_open_total_only'
    agg.to_parquet(a.output,index=False)
    feats=['starter_k_rate','starter_bb_rate','starter_hr_rate','starter_nonhr_hit_rate','opponent_k_rate','opponent_bb_rate','opponent_hr_rate','opponent_nonhr_hit_rate','contact_interaction','power_interaction','baserunner_interaction']
    manifest={'rows':int(len(agg)),'seasons':sorted(agg.season.unique().tolist()),'eligible_master_rows':int(len(m)),'matched_master_rows':int(len(joined)),'join_keys':join_keys,'doubleheader_disambiguation':'Retrosheet convention: singleton=0; same-date matchup duplicates ordered by game_datetime as 1/2; no outcome information','m1_features':feats,'feature_nonnull_coverage':{c:float(agg[c].notna().mean()) for c in feats},'pitcher_source_columns':pr,'offense_source_columns':br,'future_information_in_statistics':False,'starter_identity_pregame_verified':False,'market_data_retained':['dk_total_open_total'],'market_derivative_features_retained':False}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

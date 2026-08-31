#!/usr/bin/env python3
"""Build game-level M3 batting-order-path features on top of fixed M1.

The batting-order path table is the previously validated raw empirical as-of
start-slot distribution. Current-game lineup identities are retrospective/final
feed and therefore remain Tier-B research identities; all hitter statistics are
strictly prior-date. No smoothing, market derivatives, or assumed coefficients
are added.

For each batting team, M3 computes the expected quality of the first three hitters
of I2 under the empirical P(I2 starts at lineup slot s). Three hitters is the
minimum possible number of plate appearances in a clean-base inning, so this does
not assume an arbitrary inning length. Metrics use the same raw 365-day event-rate
family already used in M1: K, BB, HR, and non-HR hit.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

RATE_COLS={
 'k':'365d_ev_strikeout_rate_raw',
 'bb':'365d_ev_walk_rate_raw',
 'hr':'365d_ev_home_run_rate_raw',
 'hit':'365d_ev_hit_rate_raw',
}


def read(path):
    p=Path(path); return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,low_memory=False)

def find_one(root,name):
    h=list(Path(root).rglob(name))
    if len(h)!=1: raise RuntimeError(f'{name} expected exactly once under {root}; found {h}')
    return h[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--m1-matrix',type=Path,required=True)
    ap.add_argument('--reusable-root',type=Path,required=True)
    ap.add_argument('--order-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)

    m1=read(a.m1_matrix).copy()
    line=read(find_one(a.reusable_root,'lineup_asof.parquet')).copy()
    path=read(find_one(a.order_root,'batting_order_path_asof.parquet')).copy()
    for d,c in [(m1,'game_date'),(line,'game_date'),(path,'as_of_date')]:
        d[c]=pd.to_datetime(d[c],errors='coerce').dt.normalize()

    req_line=['game_id','game_date','team_id','batting_order_slot','identity_timing_class','statistics_timing_class',*RATE_COLS.values()]
    miss=[c for c in req_line if c not in line.columns]
    if miss: raise RuntimeError(f'lineup_asof missing {miss}')
    req_path=['team_id','as_of_date','inning','prior_team_inning_games','source_class','expected_start_slot',*[f'p_start_slot_{i}' for i in range(1,10)]]
    miss=[c for c in req_path if c not in path.columns]
    if miss: raise RuntimeError(f'batting_order_path_asof missing {miss}')

    path=path[path.inning==2][req_path].copy().rename(columns={'as_of_date':'game_date'})
    if path.duplicated(['team_id','game_date']).any(): raise RuntimeError('I2 batting-order path non-unique by team/date')
    line=line[line.game_id.isin(m1.game_id)].copy()
    line['batting_order_slot']=pd.to_numeric(line.batting_order_slot,errors='coerce').astype('Int64')
    if line.duplicated(['game_id','team_id','batting_order_slot']).any(): raise RuntimeError('lineup non-unique by game/team/slot')

    rows=[]
    probs=[f'p_start_slot_{i}' for i in range(1,10)]
    for (gid,team),g in line.groupby(['game_id','team_id'],sort=False):
        gd=g.game_date.iloc[0]
        p=path[(path.team_id==team)&(path.game_date==gd)]
        if len(p)!=1:
            rows.append({'game_id':gid,'team_id':team}); continue
        p=p.iloc[0]
        slot={int(r.batting_order_slot):r for _,r in g.dropna(subset=['batting_order_slot']).iterrows()}
        rec={'game_id':gid,'team_id':team,
             'order_prior_games':float(p.prior_team_inning_games) if pd.notna(p.prior_team_inning_games) else np.nan,
             'order_expected_start_slot':float(p.expected_start_slot) if pd.notna(p.expected_start_slot) else np.nan}
        for i in range(1,9): rec[f'order_p_start_{i}']=pd.to_numeric(p[f'p_start_slot_{i}'],errors='coerce')
        # Raw path probability itself is preserved; no smoothing/reliability transform.
        for metric,col in RATE_COLS.items():
            weighted=0.0; mass=0.0
            lineup_vals=[]
            for s,r in slot.items():
                v=pd.to_numeric(r[col],errors='coerce')
                if pd.notna(v): lineup_vals.append(float(v))
            lineup_mean=float(np.mean(lineup_vals)) if lineup_vals else np.nan
            for start in range(1,10):
                pr=pd.to_numeric(p[f'p_start_slot_{start}'],errors='coerce')
                if pd.isna(pr) or pr<=0: continue
                vals=[]
                for j in range(3):
                    ss=((start-1+j)%9)+1
                    if ss not in slot: vals=[]; break
                    v=pd.to_numeric(slot[ss][col],errors='coerce')
                    if pd.isna(v): vals=[]; break
                    vals.append(float(v))
                if len(vals)==3:
                    weighted += float(pr)*float(np.mean(vals)); mass += float(pr)
            seq=weighted/mass if mass>0 else np.nan
            rec[f'order_seq_{metric}_rate']=seq
            rec[f'order_seq_{metric}_contrast']=seq-lineup_mean if pd.notna(seq) and pd.notna(lineup_mean) else np.nan
        rows.append(rec)

    teamf=pd.DataFrame(rows)
    feature_cols=[f'order_p_start_{i}' for i in range(1,9)]
    feature_cols += [f'order_seq_{m}_rate' for m in RATE_COLS]
    feature_cols += [f'order_seq_{m}_contrast' for m in RATE_COLS]
    # Full-I2 target has two batting halves; average away/home batting contexts.
    gamef=teamf.groupby('game_id',as_index=False)[feature_cols+['order_prior_games']].mean()
    out=m1.merge(gamef,on='game_id',how='left',validate='one_to_one')
    out['m3_statistics_timing_class']='asof_safe_strictly_prior_date'
    out['m3_lineup_identity_timing_class']='retrospective_final_feed_unverified_pregame'
    out['m3_order_source']='validated_raw_empirical_prior_date_start_slot_distribution'
    out.to_parquet(a.output,index=False)

    manifest={
      'rows':int(len(out)),'m1_rows':int(len(m1)),
      'm3_candidate_features':feature_cols,
      'feature_nonnull_coverage':{c:float(out[c].notna().mean()) for c in feature_cols},
      'order_path_nonnull_games':int(out['order_prior_games'].notna().sum()),
      'statistics_strictly_prior_date':True,
      'same_day_prior_games_in_statistics':False,
      'lineup_identity_pregame_verified':False,
      'batting_order_smoothing_added':False,
      'hitter_rate_source':'365d raw event rates; no shrinkage added by M3',
      'sequence_definition':'path-weighted mean raw rate of first three I2 hitters',
      'market_data_added_by_m3':False,
    }
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

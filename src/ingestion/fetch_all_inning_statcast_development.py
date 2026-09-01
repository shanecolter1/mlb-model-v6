#!/usr/bin/env python3
"""Fetch development-only Baseball Savant Statcast pitch data in bounded chunks.

This source layer is restricted to 2021-2024. 2025 is a sealed holdout and is
rejected. The output contains pitch/contact fields needed for empirical feature
family challengers; it is not a model and contains no betting-market data.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from pybaseball import statcast

KEEP = [
    'game_date','game_pk','at_bat_number','pitch_number','inning','inning_topbot',
    'batter','pitcher','stand','p_throws','pitch_type','description','events',
    'balls','strikes','zone','plate_x','plate_z',
    'release_speed','effective_speed','release_spin_rate','release_extension',
    'pfx_x','pfx_z','spin_axis',
    'launch_speed','launch_angle','launch_speed_angle','barrel',
    'estimated_ba_using_speedangle','estimated_woba_using_speedangle',
    'woba_value','woba_denom','babip_value','iso_value',
    'bb_type','hc_x','hc_y','home_team','away_team','game_type'
]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--season',type=int,required=True,choices=[2021,2022,2023,2024])
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--chunk-days',type=int,default=14)
    return p.parse_args()

def season_bounds(y:int):
    # Full MLB playing envelope; Baseball Savant simply returns no rows on idle dates.
    return pd.Timestamp(f'{y}-03-15'), pd.Timestamp(f'{y}-11-10')

def main():
    a=parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    if a.season>=2025: raise RuntimeError('2025 holdout must remain sealed')
    start,end=season_bounds(a.season)
    parts=[]; chunks=[]
    cur=start
    while cur<=end:
        stop=min(cur+pd.Timedelta(days=a.chunk_days-1),end)
        last=None
        for attempt in range(3):
            try:
                x=statcast(cur.strftime('%Y-%m-%d'),stop.strftime('%Y-%m-%d'),verbose=False,parallel=True)
                last=None; break
            except Exception as e:
                last=e; time.sleep(3*(attempt+1))
        if last is not None: raise last
        if x is None or x.empty:
            chunks.append({'start':str(cur.date()),'end':str(stop.date()),'rows':0})
        else:
            cols=[c for c in KEEP if c in x.columns]
            z=x[cols].copy()
            z['source_season']=a.season
            parts.append(z)
            chunks.append({'start':str(cur.date()),'end':str(stop.date()),'rows':int(len(z))})
        cur=stop+pd.Timedelta(days=1)
    if not parts: raise RuntimeError(f'No Statcast rows returned for {a.season}')
    out=pd.concat(parts,ignore_index=True)
    key=[c for c in ['game_pk','at_bat_number','pitch_number'] if c in out.columns]
    if key: out=out.drop_duplicates(key,keep='last')
    if 'game_date' in out.columns:
        years=pd.to_datetime(out.game_date,errors='coerce').dt.year.dropna().astype(int)
        if (years>=2025).any(): raise RuntimeError('2025 leakage in Statcast source')
    path=a.output_dir/f'statcast_pitches_{a.season}.parquet'
    out.to_parquet(path,index=False)
    manifest={
      'status':'PASS','source':'Baseball Savant Statcast via pybaseball.statcast',
      'season':a.season,'rows':int(len(out)),'columns':list(out.columns),
      'chunk_days':a.chunk_days,'chunks':chunks,'holdout_season':2025,
      'holdout_opened':False,'market_data_used':False,
      'purpose':'development-only pitch/contact feature-family source layer'
    }
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='chunks'},indent=2))
if __name__=='__main__': main()

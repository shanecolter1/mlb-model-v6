#!/usr/bin/env python3
"""Audit locked empirical total x inning baseline on 2021-2024 development data.

This is implementation fidelity only. No smoothing, shrinkage, model fitting, or
2025 outcome access is allowed. The canonical input is game x inning x half.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EXACT_BUCKETS = ["0","1","2","3","4+"]

def bucket_runs(x):
    if pd.isna(x): return None
    x=int(x)
    return str(x) if x < 4 else "4+"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--matrix',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.matrix)
    required=['game_id','game_date','season','inning','half','dk_total_open_total','half_played','runs_half']
    miss=[c for c in required if c not in x.columns]
    if miss: raise RuntimeError(f'missing required columns {miss}')
    if (pd.to_numeric(x.season,errors='coerce')>=2025).any(): raise RuntimeError('2025 holdout leakage detected')
    if x.duplicated(['game_id','inning','half']).any(): raise RuntimeError('nonunique canonical key')
    if x.loc[~x.half_played.astype(bool),'runs_half'].notna().any(): raise RuntimeError('unplayed half encoded with runs')

    # Full-inning outcome is defined only where both halves were played. I9 occurrence is
    # audited separately rather than treating a missing bottom half as zero.
    p=x.pivot(index=['game_id','game_date','season','inning','dk_total_open_total'],columns='half',values=['half_played','runs_half']).reset_index()
    p.columns=['_'.join([str(z) for z in c if str(z)!='']).rstrip('_') if isinstance(c,tuple) else c for c in p.columns]
    p['both_played']=p['half_played_top'].fillna(False).astype(bool)&p['half_played_bottom'].fillna(False).astype(bool)
    p['full_inning_runs']=np.where(p.both_played,p['runs_half_top'].fillna(0)+p['runs_half_bottom'].fillna(0),np.nan)
    p['any_run']=np.where(p.both_played,(p.full_inning_runs>=1).astype(int),np.nan)
    p['run_bucket']=p.full_inning_runs.map(bucket_runs)

    played=p[p.both_played].copy()
    keys=['dk_total_open_total','inning']
    base=played.groupby(keys,as_index=False).agg(n=('game_id','size'),mean_runs=('full_inning_runs','mean'),p_any_run=('any_run','mean'))
    counts=(played.groupby(keys+['run_bucket']).size().rename('count').reset_index())
    grid=pd.MultiIndex.from_product([
        sorted(played.dk_total_open_total.dropna().unique()),range(1,10),EXACT_BUCKETS
    ],names=['dk_total_open_total','inning','run_bucket']).to_frame(index=False)
    counts=grid.merge(counts,on=keys+['run_bucket'],how='left').fillna({'count':0})
    counts=counts.merge(base[keys+['n']],on=keys,how='left')
    counts['probability']=counts['count']/counts['n']
    wide=counts.pivot_table(index=keys,columns='run_bucket',values='probability',aggfunc='first').reset_index()
    for b in EXACT_BUCKETS:
        if b not in wide.columns: wide[b]=0.0
    wide=base.merge(wide,on=keys,how='left')
    wide['p_1plus']=1-wide['0']
    wide['p_2plus']=wide['2']+wide['3']+wide['4+']
    wide['p_3plus']=wide['3']+wide['4+']
    wide['p_4plus']=wide['4+']
    wide=wide.sort_values(keys)

    # Half-level empirical table is retained because the production engine ultimately
    # combines top and bottom half distributions.
    hx=x[x.half_played.astype(bool)].copy()
    hx['run_bucket']=hx.runs_half.map(bucket_runs)
    hbase=hx.groupby(['dk_total_open_total','inning','half'],as_index=False).agg(n=('game_id','size'),mean_runs=('runs_half','mean'),p_any_run=('runs_half',lambda s: float((s>=1).mean())))
    hcounts=hx.groupby(['dk_total_open_total','inning','half','run_bucket']).size().rename('count').reset_index()
    hcounts=hcounts.merge(hbase[['dk_total_open_total','inning','half','n']],on=['dk_total_open_total','inning','half'],how='left')
    hcounts['probability']=hcounts['count']/hcounts['n']

    # Occurrence table explicitly captures bottom-9 and rare shortened-game missing halves.
    occ=x.groupby(['inning','half'],as_index=False).agg(rows=('game_id','size'),played=('half_played','sum'))
    occ['p_played']=occ.played/occ.rows

    # Hard mathematical fidelity checks.
    sums=counts.groupby(keys,as_index=False).probability.sum()
    finite=sums.probability.notna()
    max_mass_error=float((sums.loc[finite,'probability']-1).abs().max()) if finite.any() else 0.0
    max_any_error=float((wide['p_any_run']-wide['p_1plus']).abs().max())
    if max_mass_error>1e-12: raise RuntimeError(f'exact distribution mass error {max_mass_error}')
    if max_any_error>1e-12: raise RuntimeError(f'any-run identity error {max_any_error}')

    # Registered diagnostic benchmark for total=8.0, I2 on development-only data.
    bench=wide[(wide.dk_total_open_total==8.0)&(wide.inning==2)]
    bench_record=bench.iloc[0].to_dict() if len(bench)==1 else None

    wide.to_csv(a.output_dir/'m0_total_by_inning_full_distribution.csv',index=False)
    counts.to_csv(a.output_dir/'m0_total_by_inning_exact_long.csv',index=False)
    hbase.to_csv(a.output_dir/'m0_total_by_inning_half_summary.csv',index=False)
    hcounts.to_csv(a.output_dir/'m0_total_by_inning_half_exact_long.csv',index=False)
    occ.to_csv(a.output_dir/'m0_half_occurrence.csv',index=False)
    manifest={
      'status':'PASS','purpose':'M0 implementation fidelity only','development_seasons':sorted(int(v) for v in x.season.dropna().unique()),
      'holdout_season':2025,'holdout_opened':False,'smoothing_used':False,'shrinkage_used':False,'fitted_parameters':False,
      'full_inning_definition':'both top and bottom halves played; I9 missing bottom is not zero','cells':int(len(wide)),
      'full_inning_rows':int(len(played)),'max_exact_mass_error':max_mass_error,'max_any_run_identity_error':max_any_error,
      'total_8_i2_development_only':bench_record,
      'note':'The previously registered 42.264678% total=8/I2 benchmark spans the broader historical benchmark set and is not recomputed with 2025 outcomes during development.'
    }
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2,default=lambda z: float(z) if hasattr(z,'item') else str(z)),encoding='utf-8')
    print(json.dumps(manifest,indent=2,default=lambda z: float(z) if hasattr(z,'item') else str(z)))

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Select I2 probability gates on 2022-2024 OOS predictions and validate on sealed 2025.

Price-blind. This script does not choose a sportsbook wager or calculate EV.
It evaluates every Under/Over probability threshold on a development window and
reports its untouched 2025 performance, including year-by-year persistence.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd


def wilson(w, n, z=1.959963984540054):
    if n <= 0:
        return float('nan'), float('nan')
    p = w / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    rad = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return ctr-rad, ctr+rad


def score(df, side, t):
    pcol = 'p_under' if side == 'UNDER' else 'p_over'
    ycol = 'actual_under' if side == 'UNDER' else 'actual_over'
    q = df[pd.to_numeric(df[pcol], errors='coerce') >= t].copy()
    n = len(q)
    w = int(pd.to_numeric(q[ycol], errors='coerce').fillna(0).sum())
    hit = w/n if n else float('nan')
    meanp = float(pd.to_numeric(q[pcol], errors='coerce').mean()) if n else float('nan')
    lo, hi = wilson(w, n)
    return dict(n=n, wins=w, losses=n-w, hit_rate=hit, mean_model_probability=meanp,
                calibration_gap_pp=(hit-meanp)*100 if n else float('nan'),
                wilson95_low=lo, wilson95_high=hi)


def run_one(path, outdir, start, stop, step):
    x = pd.read_csv(path)
    req = {'season','p_over','p_under','actual_over','actual_under'}
    miss = req-set(x.columns)
    if miss: raise SystemExit(f'Missing columns: {sorted(miss)}')
    dev = x[x.season.isin([2022,2023,2024])].copy()
    val = x[x.season.eq(2025)].copy()
    rows=[]
    season_rows=[]
    grid=np.arange(start, stop+step/2, step)
    for side in ['UNDER','OVER']:
        for raw_t in grid:
            t=round(float(raw_t), 6)
            d=score(dev,side,t); v=score(val,side,t)
            per=[]
            for season in [2022,2023,2024]:
                s=score(dev[dev.season.eq(season)],side,t)
                per.append(s)
                season_rows.append({'side':side,'threshold':t,'season':season,**s})
            valid_hits=[s['hit_rate'] for s in per if np.isfinite(s['hit_rate'])]
            season_sd=float(np.std(valid_hits,ddof=1)) if len(valid_hits)>=2 else float('nan')
            rows.append({
                'side':side,'threshold':t,
                **{f'dev_{k}':vv for k,vv in d.items()},
                'dev_season_hit_rate_sd':season_sd,
                'dev_min_season_n':min(s['n'] for s in per),
                'dev_2022_hit_rate':per[0]['hit_rate'],
                'dev_2023_hit_rate':per[1]['hit_rate'],
                'dev_2024_hit_rate':per[2]['hit_rate'],
                **{f'val2025_{k}':vv for k,vv in v.items()},
            })
    outdir.mkdir(parents=True,exist_ok=True)
    tab=pd.DataFrame(rows)
    tab.to_csv(outdir/'threshold_holdout_table.csv',index=False)
    pd.DataFrame(season_rows).to_csv(outdir/'development_by_season.csv',index=False)

    # Ranking is diagnostic, not an automatic production recommendation.
    eligible=tab[(tab.dev_n>=100)&(tab.dev_min_season_n>=25)].copy()
    eligible['rank_wilson']=eligible.groupby('side')['dev_wilson95_low'].rank(method='min',ascending=False)
    eligible.sort_values(['side','rank_wilson','threshold']).to_csv(outdir/'ranked_development_candidates.csv',index=False)
    manifest={
        'status':'PASS','price_used':False,'market_fields_used':[],
        'development_seasons':[2022,2023,2024],'sealed_validation_season':2025,
        'development_predictions':int(len(dev)),'validation_predictions':int(len(val)),
        'threshold_grid':{'start':start,'stop':stop,'step':step},
        'note':'Thresholds are selected/evaluated without sportsbook price. 2025 is not used in development ranking.'
    }
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--strict-input',required=True)
    ap.add_argument('--baseline-input',required=True)
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--start',type=float,default=.40)
    ap.add_argument('--stop',type=float,default=.75)
    ap.add_argument('--step',type=float,default=.005)
    a=ap.parse_args()
    root=Path(a.output_dir)
    run_one(a.strict_input,root/'strict_context',a.start,a.stop,a.step)
    run_one(a.baseline_input,root/'total_only',a.start,a.stop,a.step)
    print(json.dumps({'status':'PASS','output_dir':str(root),'price_used':False},indent=2))

if __name__=='__main__': main()

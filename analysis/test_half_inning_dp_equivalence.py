#!/usr/bin/env python3
"""Numerical-equivalence and latency gate for exact DP half-inning engine."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
import half_inning_scoring_gate as recursive
import half_inning_dp_engine as dp

TOL=1e-10
P95_ALL_9_MS_LIMIT=1000.0


def make_lineups():
    rng=np.random.default_rng(20260823)
    out=[]
    # Typical MLB-like random lineups.
    alpha=np.array([14.0,1.8,3.0,.9,.08,.65])
    for _ in range(24):
        out.append([rng.dirichlet(alpha) for _ in range(9)])
    # Stress distributions emphasizing individual events while remaining normalized.
    for focus in range(6):
        lineup=[]
        for b in range(9):
            p=np.full(6,.01)
            p[focus]=.80
            p[(focus+1)%6]=.15
            p=p/p.sum()
            lineup.append(p)
        out.append(lineup)
    return out


def main():
    table=json.loads((ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json').read_text())
    max_diff=0.0;worst=None;cases=0
    lineups=make_lineups()
    for li,lineup in enumerate(lineups):
        for start in range(9):
            old=recursive.run_dist(lineup,start,table)
            new=dp.run_dist_dp(lineup,start,table)
            diff=float(np.max(np.abs(old-new)))
            cases+=1
            if diff>max_diff:
                max_diff=diff;worst={'lineup_case':li,'start_idx':start,'old':old.tolist(),'new':new.tolist()}
    bench=dp.benchmark(lineups[0],table,repeats=75)
    result={'cases':cases,'tolerance':TOL,'max_abs_probability_difference':max_diff,'worst_case':worst,'latency':bench,'equivalence_status':'PASS' if max_diff<=TOL else 'BLOCKED','latency_status':'PASS' if bench['all_9_starts_p95_ms']<P95_ALL_9_MS_LIMIT else 'BLOCKED'}
    out=ROOT/'data/derived/model_calibration/batter_pitcher_blend/dp_engine_validation.json'
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if max_diff>TOL:raise SystemExit(f'DP EQUIVALENCE BLOCKED: {max_diff} > {TOL}')
    if bench['all_9_starts_p95_ms']>=P95_ALL_9_MS_LIMIT:raise SystemExit(f'DP LATENCY BLOCKED: {bench}')
if __name__=='__main__':main()

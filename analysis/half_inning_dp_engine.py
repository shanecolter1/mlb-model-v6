#!/usr/bin/env python3
"""Exact dynamic-programming half-inning run-distribution engine.

This module is a computational replacement for the recursive research evaluator.
It preserves the same 18-PA cap, 0/1/2/3/4/5/6+ run buckets, batter-order progression,
and empirical event/base-out transition probabilities. No transition pruning,
Monte Carlo approximation, or sample reduction is used.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
import json
import numpy as np

CLASSES=['out','bb','single','double','triple','hr']
CAP=6
MAX_PA=18


def _compile(table):
    """Compile JSON transitions into compact tuples indexed by event, outs, bases."""
    compiled={}
    states=table['states']
    for ev in CLASSES:
        for outs in range(3):
            for mask in range(8):
                key=f'{ev}|{outs}|{mask}'
                trans=[]
                for t in states[key]:
                    no=min(3,outs+int(t['outs_added']))
                    nm=int(t['post_mask'])
                    runs=int(t['runs'])
                    p=float(t['p'])
                    trans.append((no,nm,runs,p))
                compiled[(ev,outs,mask)]=tuple(trans)
    return compiled


def run_dist_dp(lineup_probs,start_idx,table):
    """Return exact [P(0),...,P(5),P(6+)] using iterative probability propagation."""
    if len(lineup_probs)!=9:
        raise ValueError('lineup_probs must contain nine batter probability vectors')
    compiled=table.get('_compiled_dp')
    if compiled is None:
        compiled=_compile(table)
        table['_compiled_dp']=compiled

    # Active state: outs x bases x runs_bucket. Lineup position is deterministic by depth.
    active=np.zeros((3,8,CAP+1),dtype=np.float64)
    active[0,0,0]=1.0
    result=np.zeros(CAP+1,dtype=np.float64)

    for depth in range(MAX_PA):
        batter=(start_idx+depth)%9
        probs=np.asarray(lineup_probs[batter],dtype=np.float64)
        nxt=np.zeros_like(active)
        for outs in range(3):
            for mask in range(8):
                src=active[outs,mask]
                if not np.any(src):
                    continue
                for ev_i,ev in enumerate(CLASSES):
                    pe=float(probs[ev_i])
                    if pe<=0:
                        continue
                    for no,nm,runs,pt in compiled[(ev,outs,mask)]:
                        w=pe*pt
                        if w<=0:
                            continue
                        if no>=3:
                            for rb,mass in enumerate(src):
                                if mass:
                                    result[min(CAP,rb+runs)]+=mass*w
                        else:
                            for rb,mass in enumerate(src):
                                if mass:
                                    nxt[no,nm,min(CAP,rb+runs)]+=mass*w
        active=nxt
        if not np.any(active):
            break

    # Match recursive depth-cap behavior: unresolved mass stops with no additional runs.
    result+=active.sum(axis=(0,1))
    s=float(result.sum())
    if not math.isfinite(s) or s<=0:
        return np.array([1.,0,0,0,0,0,0],dtype=np.float64)
    return result/s


def run_all_starts_dp(lineup_probs,table):
    """Calculate all nine possible starting lineup positions exactly."""
    return np.vstack([run_dist_dp(lineup_probs,s,table) for s in range(9)])


def benchmark(lineup_probs,table,repeats=100):
    # Warm compile/cache.
    run_all_starts_dp(lineup_probs,table)
    samples=[]
    for _ in range(repeats):
        t0=time.perf_counter()
        run_all_starts_dp(lineup_probs,table)
        samples.append((time.perf_counter()-t0)*1000.0)
    a=np.asarray(samples)
    return {
        'repeats':int(repeats),
        'all_9_starts_mean_ms':float(a.mean()),
        'all_9_starts_p95_ms':float(np.percentile(a,95)),
        'all_9_starts_max_ms':float(a.max()),
        'per_start_p95_ms_estimate':float(np.percentile(a,95)/9.0),
    }


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    table=json.loads((root/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json').read_text())
    rng=np.random.default_rng(20260823)
    lineup=[]
    for _ in range(9):
        x=rng.dirichlet(np.array([14.,1.8,3.,.9,.08,.65]))
        lineup.append(x)
    print(json.dumps(benchmark(lineup,table,100),indent=2))

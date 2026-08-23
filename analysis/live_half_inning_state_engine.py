#!/usr/bin/env python3
"""Market-blind live half-inning state engine.

Starts from the actual live baseball state rather than only a clean inning start.
The first PA may use a count-conditioned outcome vector; later PAs use the frozen
pregame/live batter-pitcher vectors. No sportsbook input is accepted.
"""
from __future__ import annotations
import math
import numpy as np
from half_inning_transition_operator import (
    CLASSES,CAP,N_RUN,N_ACTIVE,N_OUTS,N_BASES,_idx,compile_event_matrices
)


def live_half_inning_distribution(
    lineup_probs, current_batter_idx, outs, bases_mask, runs_already, table,
    current_pa_probs=None, max_remaining_pa=36
):
    """Return final-half run distribution and next-inning leadoff distribution.

    Parameters use the observed live state. `current_pa_probs`, when supplied,
    is the six-outcome probability vector conditional on the current pitch count.
    Runs already scored are a deterministic floor in the final-half distribution.
    """
    probs=np.asarray(lineup_probs,dtype=np.float64)
    if probs.shape!=(9,len(CLASSES)): raise ValueError('lineup_probs must be (9,6)')
    if not (0<=current_batter_idx<9): raise ValueError('current_batter_idx must be 0..8')
    if not (0<=outs<=2): raise ValueError('outs must be 0..2')
    if not (0<=bases_mask<=7): raise ValueError('bases_mask must be 0..7')
    if runs_already<0: raise ValueError('runs_already must be nonnegative')
    probs=probs/probs.sum(axis=1,keepdims=True)
    first=None
    if current_pa_probs is not None:
        first=np.asarray(current_pa_probs,dtype=np.float64)
        if first.shape!=(len(CLASSES),) or np.any(first<0) or first.sum()<=0:
            raise ValueError('current_pa_probs must be a valid length-6 vector')
        first=first/first.sum()

    A,B=compile_event_matrices(table)
    active=np.zeros(N_ACTIVE,dtype=np.float64)
    active[_idx(outs,bases_mask,min(CAP,runs_already))]=1.0
    joint=np.zeros((N_RUN,9),dtype=np.float64)

    for depth in range(max_remaining_pa):
        batter=(current_batter_idx+depth)%9
        pa=first if depth==0 and first is not None else probs[batter]
        ended=np.einsum('i,e,eir->r',active,pa,B,optimize=True)
        next_leadoff=(batter+1)%9
        joint[:,next_leadoff]+=ended
        active=np.einsum('i,e,eij->j',active,pa,A,optimize=True)
        if not np.any(active): break

    unresolved=active.reshape(N_OUTS,N_BASES,N_RUN).sum(axis=(0,1))
    final_run_dist=joint.sum(axis=1)+unresolved
    total=float(final_run_dist.sum())
    if not math.isfinite(total) or total<=0: raise RuntimeError('probability mass vanished')
    final_run_dist/=total; joint/=total; unresolved/=total
    next_lineup=joint.sum(axis=0)
    resolved=float(next_lineup.sum())
    if resolved>0: next_lineup/=resolved
    return {
        'final_half_run_distribution':final_run_dist,
        'joint_final_runs_next_lineup':joint,
        'next_lineup_distribution_resolved':next_lineup,
        'unresolved_probability':float(unresolved.sum()),
        'runs_already_floor':int(runs_already),
        'entry_state':{'outs':int(outs),'bases_mask':int(bases_mask),'current_batter_idx':int(current_batter_idx)},
        'count_conditioned_first_pa':first is not None,
    }


def benchmark_live_state(lineup_probs,table,repeats=200):
    import time
    live_half_inning_distribution(lineup_probs,4,1,5,1,table)
    xs=[]
    for _ in range(repeats):
        t=time.perf_counter(); live_half_inning_distribution(lineup_probs,4,1,5,1,table)
        xs.append((time.perf_counter()-t)*1000)
    a=np.asarray(xs)
    return {'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'p99_ms':float(np.percentile(a,99)),'max_ms':float(a.max()),'repeats':repeats}

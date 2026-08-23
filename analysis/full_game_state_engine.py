#!/usr/bin/env python3
"""Market-blind full-game probability propagation scaffold.

This module composes team-specific half-inning run x next-lineup operators into
remaining-game state distributions. It intentionally does NOT use sportsbook
prices or totals.

Production note:
The validated half-inning interface currently exposes 0/1/2/3/4/5/6+ runs.
That is sufficient for half-inning calibration but NOT sufficient for exact
full-game final-score/margin probabilities after multiple innings because the
6+ tail loses exact run count. Therefore this module exposes lineup propagation
now, while final-score promotion is BLOCKED until the operator is generalized to
a wider internal run grid and game-rule layers (walk-offs, skipped B9, extras,
pitcher/bullpen continuation, substitutions) are validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import numpy as np


@dataclass(frozen=True)
class FullGameReadiness:
    lineup_state_propagation: bool = True
    market_blind: bool = True
    exact_final_score_ready: bool = False
    blockers: tuple = (
        'widen internal run grid beyond 6+ tail',
        'current-half arbitrary outs/bases/count entry state',
        'pitcher continuation and bullpen transition model',
        'substitution/pinch-hit lineup updates',
        'home-team bottom-9 skip and walk-off stopping rules',
        'extra-inning automatic-runner rules',
    )


def normalize(v):
    a=np.asarray(v,dtype=np.float64)
    s=float(a.sum())
    if s<=0: raise ValueError('probability vector has no mass')
    return a/s


def propagate_lineup_states(start_lineup_dist, operators: Sequence[np.ndarray]):
    """Propagate a team's batting-order state through future half-innings.

    Parameters
    ----------
    start_lineup_dist : length-9 vector
        Probability that each lineup slot leads the next team half-inning.
    operators : sequence of arrays shape (9,R,9)
        Team-specific half-inning operators. Each may reflect a different
        opposing pitcher/bullpen state.

    Returns a list of dicts containing joint run/next-lineup distributions.
    This is exact with respect to the supplied operators.
    """
    lineup=normalize(start_lineup_dist)
    out=[]
    for n,raw in enumerate(operators):
        op=np.asarray(raw,dtype=np.float64)
        if op.ndim!=3 or op.shape[0]!=9 or op.shape[2]!=9:
            raise ValueError('operator must have shape (9,R,9)')
        joint=np.einsum('s,srk->rk',lineup,op,optimize=True)
        total=float(joint.sum())
        if total<=0: raise ValueError(f'operator {n} has no probability mass')
        joint/=total
        next_lineup=joint.sum(axis=0)
        next_lineup=normalize(next_lineup)
        out.append({
            'half_index':n,
            'joint_runs_next_lineup':joint,
            'run_distribution':joint.sum(axis=1),
            'next_lineup_distribution':next_lineup,
        })
        lineup=next_lineup
    return out


def propagate_two_teams(away_start, home_start, away_ops, home_ops):
    """Propagate lineup state for both clubs independently through supplied halves.

    This is the reusable state backbone for a live full-game engine. Score-state
    coupling and baseball stopping rules are intentionally separate layers.
    """
    return {
        'away':propagate_lineup_states(away_start,away_ops),
        'home':propagate_lineup_states(home_start,home_ops),
        'readiness':FullGameReadiness(),
    }


def lineup_transition_matrix(operator):
    """Marginal 9x9 next-lineup transition matrix from a joint run operator."""
    op=np.asarray(operator,dtype=np.float64)
    if op.ndim!=3 or op.shape[0]!=9 or op.shape[2]!=9:
        raise ValueError('operator must have shape (9,R,9)')
    m=op.sum(axis=1)
    row=m.sum(axis=1,keepdims=True)
    return np.divide(m,row,out=np.zeros_like(m),where=row>0)


def expected_runs_by_start(operator):
    """Diagnostic expected runs when run states are exact integer buckets.

    If the last bucket is an open tail (e.g. 6+), this is a LOWER BOUND because
    the tail is represented by its floor. This function labels that behavior by
    returning both the values and a tail flag.
    """
    op=np.asarray(operator,dtype=np.float64)
    r=op.shape[1]
    dist=op.sum(axis=2)
    vals=np.arange(r,dtype=np.float64)
    return {'lower_bound':dist@vals,'last_bucket_is_tail':True}


if __name__=='__main__':
    print(FullGameReadiness())

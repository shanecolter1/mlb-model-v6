#!/usr/bin/env python3
"""Exact joint half-inning run x next-lineup transition operator.

For each of the nine possible lineup positions that can start a half-inning, this
engine returns the joint probability of:
  * runs scored in the half-inning (0,1,2,3,4,5,6+), and
  * the lineup position due to lead off the team's next half-inning.

The implementation is an exact computational reformulation of the validated
PA/event/base-out recursion. It uses dense NumPy transition matrices so all nine
starting positions are propagated together. No transition pruning, sampling,
market input, or probability approximation is used.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np

CLASSES = ['out','bb','single','double','triple','hr']
CAP = 6
MAX_PA = 18
N_OUTS = 3
N_BASES = 8
N_RUN = CAP + 1
N_ACTIVE = N_OUTS * N_BASES * N_RUN


def _idx(outs: int, bases: int, runs: int) -> int:
    return (outs * N_BASES + bases) * N_RUN + runs


def compile_event_matrices(table: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return active and absorbing matrices for each PA outcome.

    active[e,i,j] is P(next active state j | current state i, event e).
    absorb[e,i,r] is P(inning ends with run bucket r | current state i,event e).
    """
    cached = table.get('_joint_operator_matrices')
    if cached is not None:
        return cached

    active = np.zeros((len(CLASSES), N_ACTIVE, N_ACTIVE), dtype=np.float64)
    absorb = np.zeros((len(CLASSES), N_ACTIVE, N_RUN), dtype=np.float64)
    states = table['states']

    for ei, ev in enumerate(CLASSES):
        for outs in range(3):
            for bases in range(8):
                transitions = states[f'{ev}|{outs}|{bases}']
                for rb in range(N_RUN):
                    i = _idx(outs, bases, rb)
                    for t in transitions:
                        no = min(3, outs + int(t['outs_added']))
                        nb = int(t['post_mask'])
                        rr = min(CAP, rb + int(t['runs']))
                        p = float(t['p'])
                        if no >= 3:
                            absorb[ei, i, rr] += p
                        else:
                            active[ei, i, _idx(no, nb, rr)] += p

    table['_joint_operator_matrices'] = (active, absorb)
    return active, absorb


def joint_run_next_lineup(lineup_probs, table, max_pa: int = MAX_PA) -> Dict[str, np.ndarray | float]:
    """Compute all nine half-inning starting positions simultaneously.

    Returns
    -------
    joint : ndarray (9, 7, 9)
        joint[s,r,k] = P(run bucket r and next half-inning starts with lineup k
                          | this half-inning started with lineup s).
    unresolved : ndarray (9, 7)
        Tiny probability mass still active at max_pa, retained separately rather
        than assigning a false next-lineup position. This is a model-governance
        diagnostic and should be negligible before full-game promotion.
    run_dist : ndarray (9, 7)
        Run distribution including unresolved max-PA mass, matching the legacy
        recursive engine's depth-cap convention exactly.
    """
    probs = np.asarray(lineup_probs, dtype=np.float64)
    if probs.shape != (9, len(CLASSES)):
        raise ValueError(f'lineup_probs must have shape (9,{len(CLASSES)})')
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError('invalid lineup probabilities')
    row_sums = probs.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError('each batter probability vector must have positive mass')
    probs = probs / row_sums[:, None]

    A, B = compile_event_matrices(table)

    # One active distribution for every possible inning-start lineup position.
    active = np.zeros((9, N_ACTIVE), dtype=np.float64)
    active[:, _idx(0, 0, 0)] = 1.0
    joint = np.zeros((9, N_RUN, 9), dtype=np.float64)

    starts = np.arange(9)
    for depth in range(max_pa):
        batters = (starts + depth) % 9
        pa = probs[batters]  # (start, event)

        # Absorbing mass by run bucket for each start. einsum performs the same
        # weighted event summation as the recursive engine in bulk.
        ended = np.einsum('si,se,eir->sr', active, pa, B, optimize=True)
        next_leadoff = (starts + depth + 1) % 9
        joint[starts, :, next_leadoff] += ended

        active = np.einsum('si,se,eij->sj', active, pa, A, optimize=True)
        if not np.any(active):
            break

    unresolved = active.reshape(9, N_OUTS, N_BASES, N_RUN).sum(axis=(1, 2))
    run_dist = joint.sum(axis=2) + unresolved

    # Numerical safety only; normalization should already be one to FP precision.
    totals = run_dist.sum(axis=1)
    for s in range(9):
        if not math.isfinite(float(totals[s])) or totals[s] <= 0:
            run_dist[s] = 0.0
            run_dist[s, 0] = 1.0
        else:
            scale = float(totals[s])
            run_dist[s] /= scale
            joint[s] /= scale
            unresolved[s] /= scale

    return {
        'joint': joint,
        'unresolved': unresolved,
        'run_dist': run_dist,
        'max_unresolved_probability': float(unresolved.sum(axis=1).max()),
    }


def run_all_starts(lineup_probs, table) -> np.ndarray:
    return joint_run_next_lineup(lineup_probs, table)['run_dist']


def propagate_future_half_innings(start_lineup_dist, operators):
    """Propagate lineup state through a sequence of precomputed half-inning operators.

    `start_lineup_dist` is length-9. Each operator is shape (9,7,9).
    Returns a list of joint (runs,next-lineup) distributions, one per future half.
    This is the lineup-state backbone used by the market-blind full-game engine.
    """
    lineup = np.asarray(start_lineup_dist, dtype=np.float64)
    lineup = lineup / lineup.sum()
    outputs = []
    for op in operators:
        op = np.asarray(op, dtype=np.float64)
        if op.shape != (9, 7, 9):
            raise ValueError('operator must have shape (9,7,9)')
        joint = np.einsum('s,srk->rk', lineup, op, optimize=True)
        outputs.append(joint)
        lineup = joint.sum(axis=0)
        sm = lineup.sum()
        if sm > 0:
            lineup /= sm
    return outputs


def benchmark(lineup_probs, table, repeats: int = 75) -> dict:
    joint_run_next_lineup(lineup_probs, table)  # compile/warm
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        joint_run_next_lineup(lineup_probs, table)
        samples.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(samples)
    return {
        'repeats': int(repeats),
        'all_9_joint_operator_mean_ms': float(a.mean()),
        'all_9_joint_operator_p95_ms': float(np.percentile(a, 95)),
        'all_9_joint_operator_p99_ms': float(np.percentile(a, 99)),
        'all_9_joint_operator_max_ms': float(a.max()),
    }

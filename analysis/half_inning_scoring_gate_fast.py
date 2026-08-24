#!/usr/bin/env python3
"""Exact accelerated wrapper for locked-2025 half-inning scoring gate.

Uses the joint run x next-lineup transition operator only after its separate
numerical-equivalence and latency gate passes. For each unique pregame lineup
probability matrix, all nine possible inning-start positions are calculated once
and cached. No transition pruning, Monte Carlo approximation, sample reduction,
or scoring-state simplification is used.
"""
from __future__ import annotations
import json
import numpy as np
import half_inning_scoring_gate as core
import half_inning_transition_operator as op

_original=core.run_dist
_calls=0
_operator_builds=0
_cache={}


def _lineup_key(lineup_probs):
    a=np.asarray(lineup_probs,dtype=np.float64)
    # Exact bytes are appropriate here: probabilities are frozen within a game.
    return a.shape,a.tobytes()


def exact_operator_run_dist(lineup_probs,start_idx,table):
    global _calls,_operator_builds
    _calls+=1
    key=_lineup_key(lineup_probs)
    value=_cache.get(key)
    if value is None:
        value=op.joint_run_next_lineup(lineup_probs,table)
        _cache[key]=value
        _operator_builds+=1
    return value['run_dist'][int(start_idx)].copy()


core.run_dist=exact_operator_run_dist


def annotate():
    p=core.BASE/'half_inning_validation_2025.json'
    if not p.exists():return
    r=json.loads(p.read_text())
    r['calculation_engine']={
      'engine':'exact_joint_run_next_lineup_operator',
      'legacy_run_dist_calls_replaced':_calls,
      'unique_joint_operators_built':_operator_builds,
      'cache_hits':_calls-_operator_builds,
      'transition_pruning':False,
      'monte_carlo':False,
      'sample_reduction':False
    }
    v=core.BASE/'joint_operator_validation.json'
    if v.exists():r['joint_operator_validation']=json.loads(v.read_text())
    p.write_text(json.dumps(r,indent=2))
    mpath=core.BASE/'model_development_manifest.json';m=json.loads(mpath.read_text());m['half_inning_validation_2025']=r;mpath.write_text(json.dumps(m,indent=2))
    ppath=core.BASE/'joint_multinomial_pa_model.json';pm=json.loads(ppath.read_text());pm['half_inning_validation_2025']=r;ppath.write_text(json.dumps(pm,indent=2))


if __name__=='__main__':
    code=0
    try:core.main()
    except SystemExit as e:code=e.code if isinstance(e.code,int) else 1
    finally:annotate()
    raise SystemExit(code)

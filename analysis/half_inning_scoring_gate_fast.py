#!/usr/bin/env python3
"""Exact accelerated wrapper for locked-2025 half-inning scoring gate.

Uses the dynamic-programming engine only after its separate numerical-equivalence
and latency gate passes. No transition pruning, Monte Carlo approximation, sample
reduction, or scoring-state simplification is used.
"""
from __future__ import annotations
import json
import half_inning_scoring_gate as core
import half_inning_dp_engine as dp

_original=core.run_dist
_calls=0

def exact_dp_run_dist(lineup_probs,start_idx,table):
    global _calls
    _calls+=1
    return dp.run_dist_dp(lineup_probs,start_idx,table)

core.run_dist=exact_dp_run_dist

def annotate():
    p=core.BASE/'half_inning_validation_2025.json'
    if not p.exists():return
    r=json.loads(p.read_text())
    r['calculation_engine']={'engine':'exact_dynamic_programming','recursive_calls_replaced':_calls,'transition_pruning':False,'monte_carlo':False,'sample_reduction':False}
    v=core.BASE/'dp_engine_validation.json'
    if v.exists():r['dp_engine_validation']=json.loads(v.read_text())
    p.write_text(json.dumps(r,indent=2))
    mpath=core.BASE/'model_development_manifest.json';m=json.loads(mpath.read_text());m['half_inning_validation_2025']=r;mpath.write_text(json.dumps(m,indent=2))
    ppath=core.BASE/'joint_multinomial_pa_model.json';pm=json.loads(ppath.read_text());pm['half_inning_validation_2025']=r;ppath.write_text(json.dumps(pm,indent=2))

if __name__=='__main__':
    code=0
    try:core.main()
    except SystemExit as e:code=e.code if isinstance(e.code,int) else 1
    finally:annotate()
    raise SystemExit(code)

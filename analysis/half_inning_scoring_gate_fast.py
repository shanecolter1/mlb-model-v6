#!/usr/bin/env python3
"""Numerically accelerated wrapper for the locked-2025 half-inning gate.

The production transition table contains hierarchical-shrinkage tails with many
microscopic outcomes. For validation runtime only, each state retains the most
probable transitions until cumulative mass >= 0.999999 and then renormalizes.
Thus removed probability mass is <=1e-6 per PA state; with the production 18-PA
recursion cap, a conservative union bound is <=1.8e-5 per simulated half-inning.
Both challenger and legacy are evaluated with the identical pruned table.
"""
from __future__ import annotations
import json
from pathlib import Path
import half_inning_scoring_gate as core

TARGET=0.999999
_original=core.run_dist
_pruned_by_id={}
_stats={}

def prune(table):
    key=id(table)
    if key in _pruned_by_id:return _pruned_by_id[key]
    states={};max_removed=0.0;mean_removed=[]
    for state,trans in table['states'].items():
        ordered=sorted(trans,key=lambda x:float(x['p']),reverse=True)
        kept=[];s=0.0
        for t in ordered:
            kept.append(dict(t));s+=float(t['p'])
            if s>=TARGET:break
        removed=max(0.0,1.0-s);max_removed=max(max_removed,removed);mean_removed.append(removed)
        if s<=0:raise RuntimeError(f'No transition mass retained for {state}')
        for t in kept:t['p']=float(t['p'])/s
        states[state]=kept
    out=dict(table);out['states']=states
    _pruned_by_id[key]=out
    _stats.update({'retained_mass_target':TARGET,'max_removed_mass_per_state':max_removed,'mean_removed_mass_per_state':sum(mean_removed)/len(mean_removed),'conservative_18_pa_union_bound':18*max_removed,'state_count':len(states)})
    if 18*max_removed>2e-5:raise RuntimeError(f'Numerical pruning bound too large: {_stats}')
    return out

def fast_run_dist(lineup_probs,start_idx,table):return _original(lineup_probs,start_idx,prune(table))
core.run_dist=fast_run_dist

def annotate():
    p=core.BASE/'half_inning_validation_2025.json'
    if p.exists():
        r=json.loads(p.read_text());r['numerical_acceleration']=dict(_stats);p.write_text(json.dumps(r,indent=2))
        mpath=core.BASE/'model_development_manifest.json';m=json.loads(mpath.read_text());m['half_inning_validation_2025']=r;mpath.write_text(json.dumps(m,indent=2))
        ppath=core.BASE/'joint_multinomial_pa_model.json';pm=json.loads(ppath.read_text());pm['half_inning_validation_2025']=r;ppath.write_text(json.dumps(pm,indent=2))

if __name__=='__main__':
    code=0
    try:core.main()
    except SystemExit as e:code=e.code if isinstance(e.code,int) else 1
    finally:annotate()
    raise SystemExit(code)

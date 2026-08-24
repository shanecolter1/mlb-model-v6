#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
import half_inning_transition_operator as op
import live_half_inning_state_engine as live

TOL=1e-10
UNRESOLVED_TOL=1e-12
EMERGENCY_MAX=72

def main():
    table=json.loads((ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json').read_text())
    rng=np.random.default_rng(20260823)
    alpha=np.array([14.,1.8,3.,.9,.08,.65])
    maxdiff=0.0; floor_fail=False; unresolved=[]; convergence_fail=False
    for _ in range(20):
        lineup=np.vstack([rng.dirichlet(alpha) for _ in range(9)])
        ref=op.joint_run_next_lineup(lineup,table,max_pa=EMERGENCY_MAX)
        for s in range(9):
            got=live.live_half_inning_distribution(
                lineup,s,0,0,0,table,
                unresolved_tolerance=UNRESOLVED_TOL,
                emergency_max_remaining_pa=EMERGENCY_MAX,
            )
            maxdiff=max(maxdiff,float(np.max(np.abs(got['final_half_run_distribution']-ref['run_dist'][s]))))
            unresolved.append(got['unresolved_probability'])
            if not got['converged']: convergence_fail=True
        g=live.live_half_inning_distribution(
            lineup,3,1,5,2,table,
            unresolved_tolerance=UNRESOLVED_TOL,
            emergency_max_remaining_pa=EMERGENCY_MAX,
        )
        if float(g['final_half_run_distribution'][:2].sum())>TOL: floor_fail=True
        if not g['converged']: convergence_fail=True
    lineup=np.vstack([rng.dirichlet(alpha) for _ in range(9)])
    bench=live.benchmark_live_state(lineup,table,repeats=200)
    result={
      'clean_state_max_abs_diff':maxdiff,'tolerance':TOL,
      'deterministic_run_floor':'PASS' if not floor_fail else 'BLOCKED',
      'max_unresolved_typical_synthetic':float(max(unresolved)),
      'convergence_status':'PASS' if not convergence_fail and max(unresolved)<=UNRESOLVED_TOL else 'BLOCKED',
      'unresolved_tolerance':UNRESOLVED_TOL,
      'latency':bench,
      'equivalence_status':'PASS' if maxdiff<=TOL else 'BLOCKED',
      'latency_status':'PASS' if bench['max_ms']<1000 else 'BLOCKED'
    }
    out=ROOT/'data/derived/model_calibration/batter_pitcher_blend/live_state_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if any(result[k]!='PASS' for k in ('equivalence_status','latency_status','deterministic_run_floor','convergence_status')): raise SystemExit(1)
if __name__=='__main__':main()

#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
import half_inning_transition_operator as op
import live_half_inning_state_engine as live

TOL=1e-10

def main():
    table=json.loads((ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json').read_text())
    rng=np.random.default_rng(20260823)
    alpha=np.array([14.,1.8,3.,.9,.08,.65])
    maxdiff=0.0; floor_fail=False; unresolved=[]
    for _ in range(20):
        lineup=np.vstack([rng.dirichlet(alpha) for _ in range(9)])
        ref=op.joint_run_next_lineup(lineup,table,max_pa=36)
        for s in range(9):
            got=live.live_half_inning_distribution(lineup,s,0,0,0,table,max_remaining_pa=36)
            maxdiff=max(maxdiff,float(np.max(np.abs(got['final_half_run_distribution']-ref['run_dist'][s]))))
            unresolved.append(got['unresolved_probability'])
        # deterministic run floor from a non-clean state
        g=live.live_half_inning_distribution(lineup,3,1,5,2,table,max_remaining_pa=36)
        if float(g['final_half_run_distribution'][:2].sum())>TOL: floor_fail=True
    lineup=np.vstack([rng.dirichlet(alpha) for _ in range(9)])
    bench=live.benchmark_live_state(lineup,table,repeats=200)
    result={'clean_state_max_abs_diff':maxdiff,'tolerance':TOL,'deterministic_run_floor':'PASS' if not floor_fail else 'BLOCKED','max_unresolved_typical_synthetic':float(max(unresolved)),'latency':bench,'equivalence_status':'PASS' if maxdiff<=TOL else 'BLOCKED','latency_status':'PASS' if bench['max_ms']<1000 else 'BLOCKED'}
    out=ROOT/'data/derived/model_calibration/batter_pitcher_blend/live_state_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if result['equivalence_status']!='PASS' or result['latency_status']!='PASS' or result['deterministic_run_floor']!='PASS': raise SystemExit(1)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Paired transition-only full-I2 ablation. Descriptive; no refit or promotion."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from diagnose_i2_vnext_bias import comparison, summarize


def paired(rows):
    result=comparison(rows)
    result['distinct_transitions']=result.pop('vnext')
    result['pooled_transitions']=result.pop('production_formula_controlled')
    result['distinct_minus_pooled']=result.pop('vnext_minus_comparator')
    return result


def compare(corrected, original, inputs):
    for r in (corrected,original,inputs):
        assert r['market_inputs_used'] is False
    for key in ['trials_per_game','i1_state_mode','point_in_time_player_refits','observed_i2_state_used_as_predictor','base_model_version','model_training','walkforward_policy','park_rule']:
        assert corrected[key]==original[key],key
    assert corrected.get('i1_environment','neutral')==original.get('i1_environment','neutral')
    assert corrected['i2_model']=='vnext'
    assert original.get('i2_model','vnext')=='vnext'
    assert corrected['trials_per_game']==10000
    assert corrected['transition_model']=='adapted-i2-distinct-out-pa-transitions-v1'
    assert original['transition_model']=='adapted-seasonal-shrunk-pa-transitions-v1'
    rows=corrected['predictions']; old={r['gid']:r for r in original['predictions']}; games={g['gid']:g for g in inputs['games']}
    assert len(rows)==len(old)==len(games)==2430
    assert {r['gid'] for r in rows}==set(old)==set(games)
    for r in rows:
        o=old[r['gid']];g=games[r['gid']]
        for key in ['date','observed_under05','observed_full_i2_runs','park_status','model_effective_from','model_training_end','starter_continuation_audit']:
            assert r[key]==o[key],(r['gid'],key)
        assert g['observed']['under05']==r['observed_under05']
        r['comparison_under05']=o['raw_under05']
    groups={}
    for name,group_key in [('park',lambda r:r['park_status']),('month',lambda r:r['date'][:6])]:
        grouped=defaultdict(list)
        for r in rows:grouped[group_key(r)].append(r)
        groups[name]={k:{'distinct':summarize(v),'pooled':summarize([old[r['gid']] for r in v])} for k,v in sorted(grouped.items())}
    halves={}
    for side,pred,observed in [('top','top2_score_probability','top2_runs'),('bottom','bottom2_score_probability','bottom2_runs')]:
        subset=[{'date':r['date'],'raw_under05':1-r[pred],'comparison_under05':1-old[r['gid']][pred],'observed_under05':int(games[r['gid']]['observed'][observed]==0)} for r in rows]
        halves[side]=paired(subset)
    shifts=[r['raw_under05']-r['comparison_under05'] for r in rows]
    return {'version':'i2-transition-only-ablation-v1','market_inputs_used':False,'promotion_status':'SHADOW_ONLY','games':len(rows),
            'full_season':paired(rows),'later_validation':paired([r for r in rows if r['date']>='20250625']),
            'half_innings':halves,'subgroups':groups,
            'probability_change':{'mean':sum(shifts)/len(shifts),'min':min(shifts),'max':max(shifts),'mean_absolute':sum(abs(v) for v in shifts)/len(shifts)},
            'governance':{'only_intended_change':'K/BIP transition separation using frozen 2021-2024 counts and existing shrinkage strengths','calibration_refit':False,'player_models_refit':False,'same_game_seeds':True,'independent_post_selection_test':False,'paired_bootstrap_draws':5000,'limitations':['Same 2025 sample already inspected; descriptive ablation only.','Monte Carlo paths diverge when transitions differ; 10000 trials do not eliminate simulation error.','Conditional bootstrap excludes model training and multi-day dependence uncertainty.']}}


def main():
    p=argparse.ArgumentParser()
    for name in ['corrected','original','inputs','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=compare(json.loads(a.corrected.read_text()),json.loads(a.original.read_text()),json.loads(a.inputs.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()

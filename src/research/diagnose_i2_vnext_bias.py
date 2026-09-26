#!/usr/bin/env python3
"""Descriptive bias attribution and controlled production-formula comparison.
No model fitting, promotion, calibration selection or market access.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from audit_i2_vnext_holdout import metrics, paired_uncertainty


def comparison(rows):
    y=np.array([r['observed_under05'] for r in rows],float)
    p=np.array([r['raw_under05'] for r in rows],float)
    q=np.array([r['comparison_under05'] for r in rows],float)
    m=metrics(y,p); b=metrics(y,q)
    losses=lambda x: -(y*np.log(np.clip(x,1e-12,1-1e-12))+(1-y)*np.log1p(-np.clip(x,1e-12,1-1e-12)))
    delta=np.column_stack(((p-y)**2-(q-y)**2,losses(p)-losses(q)))
    days=defaultdict(list)
    for i,r in enumerate(rows): days[r['date']].append(i)
    sums=np.array([delta[ix].sum(axis=0) for ix in days.values()])
    counts=np.array([len(ix) for ix in days.values()])
    rng=np.random.default_rng(20260926); samples=[]
    for _ in range(5000):
        ix=rng.integers(len(days),size=len(days))
        samples.append(sums[ix].sum(axis=0)/counts[ix].sum())
    ci=np.quantile(samples,[.025,.975],axis=0)
    return {'n':len(rows),'vnext':m,'production_formula_controlled':b,
            'vnext_minus_comparator':{k:{'estimate':float(delta[:,i].mean()),'ci95':ci[:,i].tolist()} for i,k in enumerate(['brier','logloss'])},
            'interval_method':'5000 paired calendar-date cluster percentile draws; conditional on frozen predictions; descriptive after selection'}


def summarize(rows):
    m=metrics([r['observed_under05'] for r in rows],[r['raw_under05'] for r in rows])
    m['under_overprediction']=m['predicted_under_mean']-m['realized_under_rate']
    m['excess_expected_unders']=sum(r['raw_under05']-r['observed_under05'] for r in rows)
    return m


def main():
    p=argparse.ArgumentParser()
    for name in ['replay','inputs','benchmark','transitions','output']: p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    replay=json.loads(args.replay.read_text()); inputs=json.loads(args.inputs.read_text()); benchmark=json.loads(args.benchmark.read_text())
    assert replay['market_inputs_used'] is False and inputs['market_inputs_used'] is False and benchmark['market_inputs_used'] is False
    assert replay['i1_state_mode']==benchmark['i1_state_mode']=='player_asof'
    assert benchmark['i2_model']=='production_formula'
    assert replay['trials_per_game']==benchmark['trials_per_game']==10000
    assert replay['transition_model']==benchmark['transition_model']
    source={g['gid']:g for g in inputs['games']}; base={r['gid']:r for r in benchmark['predictions']}
    rows=replay['predictions']
    assert len(rows)==len(source)==len(base)==2430
    assert {r['gid'] for r in rows}==set(source)==set(base)
    for r in rows:
        b=base[r['gid']]; g=source[r['gid']]
        assert r['date']==b['date']==g['date']
        assert r['observed_under05']==b['observed_under05']==g['observed']['under05']
        assert r['park_status']==b['park_status']
        r['comparison_under05']=b['raw_under05']
    grouped={}
    for name,key in [('park',lambda r:r['park_status']),('starter_continuation',lambda r:'both_started_i2' if all(source[r['gid']]['audit'][x]==1 for x in ['away_starter_began_i2','home_starter_began_i2']) else 'at_least_one_changed_before_i2'),('month',lambda r:r['date'][:6])]:
        groups=defaultdict(list)
        for r in rows: groups[key(r)].append(r)
        grouped[name]={k:summarize(v) for k,v in sorted(groups.items())}
    halves={}
    for half,pred_key,obs_key in [('top','top2_score_probability','top2_runs'),('bottom','bottom2_score_probability','bottom2_runs')]:
        halfrows=[{'date':r['date'],'observed_under05':int(source[r['gid']]['observed'][obs_key]==0),'raw_under05':1-r[pred_key]} for r in rows]
        halves[half]=summarize(halfrows)
        halves[half]['bias_ci95']=paired_uncertainty(halfrows,.5)['under_overprediction']['ci95']
    table=json.loads(args.transitions.read_text())
    states=[]
    for outs in range(3):
        for mask in range(8):
            cells=table['states'][f'out|{outs}|{mask}']
            states.append({'outs':outs,'base_mask':mask,'probability_multiple_outs':sum(x['p'] for x in cells if x['outs_added']>1),'probability_run_scores':sum(x['p'] for x in cells if x['runs']>0)})
    out={'version':'i2-vnext-bias-diagnosis-v1','market_inputs_used':False,'promotion_status':'SHADOW_ONLY','games':len(rows),
         'full_season':comparison(rows),'later_validation':comparison([r for r in rows if r['date']>='20250625']),
         'bias_subgroups':grouped,'half_inning_bias':halves,
         'transition_audit':{'strikeout_and_ball_in_play_out_share_table':True,'example_runner_on_first_no_outs':next(s for s in states if s['outs']==0 and s['base_mask']==1),'all_states':states,'interpretation':'Pooled out transitions impose the same advancement and multiple-out probabilities on strikeouts and balls in play. Net effect has not been isolated by ablation.'},
         'governance':{'model_parameters_changed':False,'outcomes_used_only_for_scoring_and_diagnostic_grouping':True,'observed_starter_changes_not_used_for_predictions':True,'exact_deployed_production_replay':False,'comparison_scope':benchmark['comparison_scope'],'limitations':['Same data already used for calibration selection; no independent promotion claim.','Shared transitions and I1 engine mean this isolates the I2 event-formula difference, not every production component.','Subgroups and monthly results are descriptive, not causal attribution.','Exact production baseline and transition artifacts pool through 2025; cannot use unchanged for leakage-safe 2025 testing.']}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()

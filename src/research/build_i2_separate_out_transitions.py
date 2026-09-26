#!/usr/bin/env python3
"""Restore distinct K/BIP transitions from archived 2021-24 counts.
Uses already selected 2024 strengths; 2025 is scoring only. Research artifact.
"""
import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('seasonal',ROOT/'analysis/seasonal_transition_shrinkage.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def counts_for(year,folder):
    data=json.loads((folder/f'transitions_{year}.json').read_text())
    result={}
    for key,items in data['states'].items():
        ev,outs,mask=key.split('|')
        result[(ev,int(outs),int(mask))]=Counter({(x['outs_added'],x['post_mask'],x['runs']):x['n'] for x in items})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/derived/model_calibration/seasonal')
    p.add_argument('--output-dir',type=Path,default=ROOT/'data/derived/i2_vnext')
    args=p.parse_args()
    old=json.loads((args.source_dir/'production_pa_transition_table_shrunk.json').read_text())
    k1=old['shrinkage']['k_exact_to_event_outs']; k2=old['shrinkage']['k_event_outs_to_event']
    counts=defaultdict(Counter)
    for year in [2021,2022,2023,2024]:
        for key,values in counts_for(year,args.source_dir).items():counts[key].update(values)
    eo,e=s.parents(counts)
    states=dict(old['states'])
    for ev in ['strikeout','ball_in_play_out']:
        for outs in range(3):
            for mask in range(8):
                key=(ev,outs,mask)
                d=s.shrunk_dist(key,counts.get(key,Counter()),eo,e,k1,k2)
                assert d and abs(sum(d.values())-1)<1e-10
                states[f'{ev}|{outs}|{mask}']=[{'outs_added':o[0],'post_mask':o[1],'runs':o[2],'p':v} for o,v in sorted(d.items())]
    model=dict(old)
    model.update(version='i2-distinct-out-pa-transitions-v1',states=states,training_years=[2021,2022,2023,2024],generic_out='retained only for legacy fallback; prefer distinct strikeout and ball_in_play_out',research_only=True)
    model.pop('validation_2025',None)
    model['governance']={'market_inputs_used':False,'hyperparameters_reselected':False,'new_shrinkage_layer':False,'fit_years':[2021,2022,2023,2024],'selection_year':2024,'2025_used_for_fit':False,'default_live_artifact_changed':False,'walk_hbp_proxy_unchanged':True}
    # Score exact same observed event/state cells; no 2025 tuning.
    test=counts_for(2025,args.source_dir); scores={}; details={}
    for ev in ['strikeout','ball_in_play_out']:
        n=0; ll_old=0.; ll_new=0.
        for (event,outs,mask),cells in test.items():
            if event!=ev:continue
            dist=lambda items:{(x['outs_added'],x['post_mask'],x['runs']):x['p'] for x in items}
            before=dist(old['states'][f'out|{outs}|{mask}']);after=dist(states[f'{ev}|{outs}|{mask}'])
            for outcome,num in cells.items():
                n+=num;ll_old-=num*math.log(max(1e-12,before.get(outcome,0)));ll_new-=num*math.log(max(1e-12,after.get(outcome,0)))
        scores[ev]={'n':n,'pooled_logloss':ll_old/n,'distinct_logloss':ll_new/n,'distinct_minus_pooled':(ll_new-ll_old)/n}
        c=counts[(ev,0,1)];total=sum(c.values())
        details[ev]={'n_2021_2024':total,'observed_multiple_out_rate':sum(v for o,v in c.items() if o[0]>1)/total,'fitted_multiple_out_probability':sum(x['p'] for x in states[f'{ev}|0|1'] if x['outs_added']>1)}
    report={'version':'i2-distinct-out-transition-audit-v1','market_inputs_used':False,'research_only':True,'frozen_strengths':old['shrinkage'],'heldout_2025_conditional_transition_scores':scores,'runner_first_zero_out_example':details,'full_inning_probability_impact':'NOT_YET_REPLAYED','production_promoted':False,'next_step':'Paired full-I2 replay using identical frozen player models, parks, I1 state and seeds; change transition artifact only.'}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'i2_distinct_out_transitions.json').write_text(json.dumps(model,separators=(',',':')))
    (args.output_dir/'distinct_out_transition_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

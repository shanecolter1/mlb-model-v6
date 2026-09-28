#!/usr/bin/env python3
"""Audit frozen 2025 monthly I2 PA vectors against first three actual PAs per half.

I2 start slots and Statcast event outcomes are target-only and never enter the
pregame vector export. All inference is descriptive on previously inspected 2025.
"""
import argparse
import csv
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
from statistics import mean, stdev

EVENTS=('single','double','triple','home_run','walk','hit_by_pitch','strikeout','ball_in_play_out')
REACH=set(EVENTS[:6])

def summary(rows):
    if not rows:return {'n':0}
    n=len(rows)
    events=Counter(x['event'] for x in rows)
    result={'n':n,'logloss':mean(-math.log(max(1e-12,x['p'][x['event']])) for x in rows),
            'multiclass_brier':mean(sum((p-(k==x['event']))**2 for k,p in x['p'].items()) for x in rows)}
    result['reach']={'predicted':mean(sum(x['p'][k] for k in REACH) for x in rows),
                     'observed':sum(events[k] for k in REACH)/n}
    result['event_rates']={k:{'predicted':mean(x['p'][k] for x in rows),'observed':events[k]/n} for k in EVENTS}
    return result

def reach_p(item):
    return sum(item['p'][k] for k in REACH)

def shifted_vector(item, offset):
    p=reach_p(item)
    q=1/(1+math.exp(-(math.log(p/(1-p))+offset)))
    return {k:v*(q/p if k in REACH else (1-q)/(1-p)) for k,v in item['p'].items()}

def fit_reach_offset(rows):
    offset=0.0
    for _ in range(15):
        q=[1/(1+math.exp(-(math.log(reach_p(x)/(1-reach_p(x)))+offset))) for x in rows]
        offset+=sum((x['event'] in REACH)-v for x,v in zip(rows,q))/sum(v*(1-v) for v in q)
    return offset

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--vectors',default='/workspace/scratch/e39cba13c44c/i2_opening_pa_vectors_2025.json')
    ap.add_argument('--statcast',default='/workspace/scratch/e39cba13c44c/vnext_data/i2_pa_statcast.csv')
    ap.add_argument('--slots',default='/workspace/scratch/e39cba13c44c/research_inputs/i1_state_2025_rows.json')
    ap.add_argument('--output',default='data/derived/i2_vnext/phase16_opening_pa_walkforward_audit.json')
    a=ap.parse_args()
    payload=json.load(open(a.vectors)); assert payload['pregame_only'] and not payload['market_inputs_used']
    halves=payload['rows']; by_gid=defaultdict(dict)
    for h in halves:by_gid[h['gid']][h['side']]=h
    slots={(r['gid'],r['side']):int(r['observed_slot']) for r in json.load(open(a.slots))['rows']}
    observed=defaultdict(lambda:defaultdict(list))
    with open(a.statcast,newline='') as f:
        for r in csv.DictReader(f):
            if r['season']=='2025':
                side='bottom' if r['inning_topbot'].lower() in ('bot','bottom') else 'top'
                observed[(r['game_date'].replace('-',''),r['game_pk'])][side].append(r)
    for sides in observed.values():
        for rows in sides.values():rows.sort(key=lambda r:int(r['at_bat_number']))
    by_date=defaultdict(list)
    for (date,pk),sides in observed.items():by_date[date].append((pk,sides))
    outcomes=[]; paired=[]; counters=Counter(); unmatched=[]
    for gid,sides in by_gid.items():
        date=sides['top']['date']
        candidates=[]
        for pk,actual in by_date[date]:
            if any(len(actual[side])<3 for side in ('top','bottom')):continue
            score=0
            for side in ('top','bottom'):
                h=sides[side]; start=slots[(gid,side)]-1
                expected=[h['entries'][(start+i)%9]['batter'] for i in range(3)]
                score+=sum(int(r['batter'])==b for r,b in zip(actual[side][:3],expected))
            if score==6:candidates.append((pk,actual))
        if len(candidates)!=1:
            counters['unmatched_or_ambiguous_games']+=1
            unmatched.append({'gid':gid,'candidate_count':len(candidates)})
            continue
        pk,actual=candidates[0];counters['matched_games']+=1
        game_rows={}
        for side in ('top','bottom'):
            h=sides[side];start=slots[(gid,side)]-1
            first=actual[side][:3]
            if any(int(r['pitcher'])!=h['pitcher'] for r in first):
                counters[f'{side}_pitcher_changed_halves']+=1
                continue
            one=[]
            for i,r in enumerate(first):
                event=r['event_class']
                if event not in EVENTS:
                    counters[f'{side}_unmapped_events']+=1;continue
                item={'gid':gid,'date':date,'side':side,'event':event,
                      'p':h['entries'][(start+i)%9]['vector'],
                      'venue_status':h['venue_status']}
                one.append(item)
            if len(one)==3:
                outcomes.extend(one);game_rows[side]=one
                counters[f'{side}_eligible_halves']+=1
        if len(game_rows)==2:paired.append(game_rows)
    if outcomes:
        sides={side:summary([x for x in outcomes if x['side']==side]) for side in ('top','bottom')}
        groups={label:{side:summary([x for x in outcomes if x['side']==side and (x['venue_status']=='RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT')==matched]) for side in ('top','bottom')} for label,matched in [('matched_venue',True),('neutral_venue',False)]}
        segments={label:{side:summary([x for x in outcomes if x['side']==side and (int(x['date'][4:6])<=month)==early]) for side in ('top','bottom')} for label,early,month in [('through_june',True,6),('july_onward',False,6)]}
        diffs=[]
        for game in paired:
            per={side:mean((x['event'] in REACH)-sum(x['p'][k] for k in REACH) for x in game[side]) for side in ('top','bottom')}
            diffs.append(per['bottom']-per['top'])
        m=mean(diffs);se=stdev(diffs)/math.sqrt(len(diffs))
        paired_result={'n_games':len(diffs),'bottom_minus_top_reach_residual':m,'normal_95pct_ci':[m-1.96*se,m+1.96*se]}
        # A single side intercept on reach mass within the existing PA vector.
        # Fit through June and report untouched July onward, matched venues only.
        matched=lambda x:x['venue_status']=='RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT'
        train=[x for x in outcomes if matched(x) and int(x['date'][4:6])<=6]
        holdout=[x for x in outcomes if matched(x) and int(x['date'][4:6])>=7]
        offsets={side:fit_reach_offset([x for x in train if x['side']==side]) for side in ('top','bottom')}
        changed=[{**x,'p':shifted_vector(x,offsets[x['side']])} for x in holdout]
        holdout_result={'fit_period':'2025 through June, matched prior-season venue',
                        'holdout_period':'2025 July onward, matched prior-season venue',
                        'reach_logit_offsets':offsets,
                        'baseline':{s:summary([x for x in holdout if x['side']==s]) for s in offsets},
                        'adjusted':{s:summary([x for x in changed if x['side']==s]) for s in offsets}}
        holdout_result['combined']={'baseline':summary(holdout),'adjusted':summary(changed)}
    else:sides=groups=segments=paired_result={}
    result={'version':'phase16-opening-pa-walkforward-audit-v1','method':'Pregame model vectors for all 9 hitters per side; actual start slot, game identity, first three PAs and starter continuity joined as targets only.','selection':'Only games with six exact opening hitter identities and halves with three starter PAs and mapped terminal events.','coverage':dict(counters),'unmatched_games':unmatched,'side':sides,'by_venue_status':groups,'by_season_segment':segments,'paired':paired_result,'temporal_reach_offset_probe':holdout_result}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'coverage':dict(counters),'side_reach':{s:v.get('reach') for s,v in sides.items()},'paired':paired_result,'holdout_offsets':holdout_result['reach_logit_offsets'],'holdout_combined_logloss':{k:v['logloss'] for k,v in holdout_result['combined'].items()}},indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Build leakage-safe historical candidate sets for conditional reliever selection.

Candidate pools use only pitchers observed for the same club in PRIOR games.
No pitcher is included because he later appears in the current game. This avoids
answer leakage. Both in-inning and between-inning pitching changes are emitted.

This file is exercised by the development-only bullpen PR validation workflow.
"""
from __future__ import annotations
import csv,json,re
from collections import defaultdict,deque,Counter
from pathlib import Path
import half_inning_scoring_gate as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/bullpen_transitions'
LOOKBACK_DAYS=30
MIN_TEAM_COVERAGE=.999
MIN_ACTUAL_IN_POOL=.90

def iv(r,k,d=-1):
    try:return int(float(r.get(k)))
    except:return d

def clean(v):return str(v or '').strip()

def teams(r,gid):
    hcands=('home_team','hometeam','home','team_home','home_id','home_team_id')
    acands=('away_team','awayteam','visitor_team','visit_team','visteam','visitor','away','team_away','away_id','away_team_id')
    h=next((clean(r.get(k)) for k in hcands if clean(r.get(k))), '')
    a=next((clean(r.get(k)) for k in acands if clean(r.get(k))), '')
    if not h:
        m=re.match(r'^([A-Za-z]{3})',str(gid)); h=m.group(1).upper() if m else ''
    return h,a

def main():
    rows=core.fetch_rows_2025(); obs=[]; diag=Counter(); team_fields=Counter(); team_samples={}
    for r in rows[:10000]:
        for k,v in r.items():
            lk=k.lower()
            if any(x in lk for x in ('team','home','away','visit')) and v not in (None,''):
                team_fields[k]+=1;team_samples.setdefault(k,str(v))
    hist=defaultdict(deque)
    i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid'];j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=[r for r in rows[i:j] if core.truth(r.get('pa'))]
        if not game:i=j;continue
        h,a=teams(game[0],gid)
        if h and a:diag['games_with_both_teams']+=1
        else:diag['games_missing_team_id']+=1
        pools={}
        for team in (h,a):
            if not team:continue
            dq=hist[team]
            while dq and day-dq[0][0]>LOOKBACK_DAYS:dq.popleft()
            p=defaultdict(lambda:{'apps':0,'bf':0,'last_day':None})
            seen_games=set()
            for d,pgid,pid,bf in dq:
                key=(pgid,pid)
                if key in seen_games:continue
                seen_games.add(key);z=p[pid];z['apps']+=1;z['bf']+=bf;z['last_day']=d if z['last_day'] is None else max(z['last_day'],d)
            pools[team]=p
        current_usage=defaultdict(lambda:defaultdict(int))
        for n,r in enumerate(game):
            tb=iv(r,'top_bot',-1);pid=clean(r.get('pitcher'))
            if tb not in (0,1) or not pid:continue
            team=h if tb==0 else a
            if team:current_usage[team][pid]+=1
            if n+1>=len(game):continue
            nxt=game[n+1];tb2=iv(nxt,'top_bot',-1);npid=clean(nxt.get('pitcher'))
            if not npid:continue
            inn=iv(r,'inning',iv(r,'inn_ct',-1));inn2=iv(nxt,'inning',iv(nxt,'inn_ct',-1))
            kind=None
            if tb2==tb and inn2==inn and npid!=pid:kind='in_inning'
            elif tb2==tb and inn2==inn+1 and npid!=pid:kind='between_inning'
            if kind is None:continue
            if not team:
                diag['change_events_missing_team']+=1;continue
            pool=pools.get(team,{})
            candidates=[]
            for cand,z in pool.items():
                if cand==pid:continue
                rest=(day-z['last_day']) if z['last_day'] is not None else None
                candidates.append({'pitcher_id':cand,'prior_apps_30d':z['apps'],'prior_bf_30d':z['bf'],'prior_rest_days':rest})
            actual_in=any(c['pitcher_id']==npid for c in candidates)
            diag['change_events']+=1;diag['actual_in_candidate_pool']+=int(actual_in)
            obs.append({'game_id':gid,'date':str(day),'defense_team':team,'transition_kind':kind,'inning':inn,
                        'outgoing_pitcher_id':pid,'actual_next_pitcher_id':npid,'actual_next_in_candidate_pool':int(actual_in),
                        'candidate_count':len(candidates),'candidates_json':json.dumps(candidates,separators=(',',':'))})
        for team,usage in current_usage.items():
            for pid,bf in usage.items():hist[team].append((day,gid,pid,bf))
        i=j
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'bullpen_reliever_candidate_sets_2025.csv'
    fields=['game_id','date','defense_team','transition_kind','inning','outgoing_pitcher_id','actual_next_pitcher_id','actual_next_in_candidate_pool','candidate_count','candidates_json']
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(obs)
    games_total=diag['games_with_both_teams']+diag['games_missing_team_id']
    team_cov=diag['games_with_both_teams']/games_total if games_total else 0
    actual_cov=diag['actual_in_candidate_pool']/diag['change_events'] if diag['change_events'] else 0
    rep={'market_blind':True,'candidate_definition':f'pitchers used by same club in prior {LOOKBACK_DAYS} days only; current-game future appearances prohibited',
         'rows':len(obs),'team_id_coverage':team_cov,'actual_reliever_candidate_coverage':actual_cov,'diagnostics':dict(diag),
         'team_schema_candidates':dict(team_fields),'team_schema_samples':team_samples,
         'promotion_gates':{'team_id_coverage_ge_0_999':team_cov>=MIN_TEAM_COVERAGE,'actual_reliever_candidate_coverage_ge_0_90':actual_cov>=MIN_ACTUAL_IN_POOL},
         'governance':'candidate coverage may be improved only with information available before the decision; never backfill from current-game future appearances'}
    (OUT/'bullpen_reliever_candidate_summary_2025.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
    if not all(rep['promotion_gates'].values()):raise SystemExit('Reliever candidate-set gate blocked')
if __name__=='__main__':main()

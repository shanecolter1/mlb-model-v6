#!/usr/bin/env python3
"""Fresh 2026 forward test for the frozen 2025 expanded reliever-selection model.

Governance:
- frozen 2025 expanded feature specification;
- frozen regularization C=0.03;
- training sample frozen to 2025 through July 31;
- no 2026 observation is used for model selection, fitting, or threshold tuning;
- 2026 play-by-play and exact-date active rosters come from MLB Stats API;
- workload/role predictors use completed games strictly before the decision game;
- sportsbook/market data are prohibited.
"""
from __future__ import annotations
import csv, json, math, time, urllib.request
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

import bullpen_reliever_selection_model as sel

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions'
CAND_2026=BASE/'bullpen_reliever_candidate_sets_2026.csv'
SUM_2026=BASE/'bullpen_reliever_candidate_summary_2026.json'
OUT=BASE/'bullpen_reliever_selection_forward_2026.json'
YEAR=2026
END_DATE='2026-08-23'
FROZEN_C=0.03
UA='mlb-model-v6 frozen 2026 bullpen forward test'


def get_json(url,timeout=90,retries=4):
    last=None
    for i in range(retries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':UA})
            with urllib.request.urlopen(req,timeout=timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            last=e
            if i+1<retries:time.sleep(0.5*(2**i))
    raise last


def schedule_games():
    url=(f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R'
         f'&startDate=2026-03-01&endDate={END_DATE}')
    data=get_json(url)
    out=[]
    for d in data.get('dates',[]):
        day=d.get('date')
        for g in d.get('games',[]):
            st=g.get('status') or {}
            if st.get('abstractGameState')!='Final' and st.get('detailedState') not in {'Final','Game Over','Completed Early'}:continue
            out.append((day,int(g['gamePk']),g.get('gameDate') or day))
    out.sort(key=lambda x:(x[0],x[2],x[1]))
    return out


def fetch_feed(item):
    day,pk,start=item
    data=get_json(f'https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live',timeout=90)
    return day,pk,start,data


def fetch_roster(team_id,day):
    data=get_json(f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&season={YEAR}&date={day}&hydrate=person')
    out=[]
    for x in data.get('roster',[]):
        pos=x.get('position') or {};person=x.get('person') or {}
        if str(pos.get('abbreviation') or '').upper()!='P' and str(pos.get('type') or '').lower()!='pitcher':continue
        try:pid=int(person['id'])
        except Exception:continue
        ph=person.get('pitchHand') or {}
        out.append((pid,str(ph.get('code') or 'UNK').upper()))
    return out


def play_state(play,home_id,away_id):
    about=play.get('about') or {};match=play.get('matchup') or {};res=play.get('result') or {}
    half=str(about.get('halfInning') or '').lower();inn=int(about.get('inning') or 0)
    try:pid=int((match.get('pitcher') or {})['id'])
    except Exception:return None
    team=home_id if half=='top' else away_id if half=='bottom' else None
    if team is None:return None
    hs=res.get('homeScore');aw=res.get('awayScore')
    try:hs=int(hs);aw=int(aw)
    except Exception:hs=aw=None
    return {'inning':inn,'half':half,'pitcher':pid,'team':team,'home_score_post':hs,'away_score_post':aw}


def defense_diff(state,home_id):
    if state['home_score_post'] is None:return None
    return state['home_score_post']-state['away_score_post'] if state['team']==home_id else state['away_score_post']-state['home_score_post']


def parse_game(day,pk,start,feed):
    teams=(feed.get('gameData') or {}).get('teams') or {}
    try:home_id=int(teams['home']['id']);away_id=int(teams['away']['id'])
    except Exception:return None
    plays=((feed.get('liveData') or {}).get('plays') or {}).get('allPlays') or []
    states=[]
    for p in plays:
        s=play_state(p,home_id,away_id)
        if s:states.append(s)
    usage=defaultdict(lambda:defaultdict(int));first_inning=defaultdict(dict);seen=defaultdict(set);events=[]
    for i,s in enumerate(states):
        team=s['team'];pid=s['pitcher'];inn=s['inning']
        usage[team][pid]+=1
        first_inning[team].setdefault(pid,inn)
        if i+1<len(states):
            n=states[i+1]
            if n['team']==team and n['half']==s['half'] and n['inning']==inn and n['pitcher']!=pid:
                sd=defense_diff(s,home_id)
                events.append({'game_id':pk,'date':day,'defense_team':team,'transition_kind':'in_inning',
                    'inning':inn,'decision_inning':inn,'defensive_score_diff':sd,
                    'outgoing_pitcher_id':pid,'actual_next_pitcher_id':n['pitcher'],
                    'already_used_ids':sorted(seen[team] | {pid})})
        seen[team].add(pid)
    return {'date':day,'game_id':pk,'start':start,'home_id':home_id,'away_id':away_id,
            'usage':usage,'first_inning':first_inning,'events':events}


def build_2026_groups():
    sched=schedule_games()
    if len(sched)<1000:raise SystemExit(f'Unexpectedly few completed 2026 regular-season games: {len(sched)}')
    feeds=[];diag=Counter()
    with ThreadPoolExecutor(max_workers=10) as ex:
        fut={ex.submit(fetch_feed,x):x for x in sched}
        for f in as_completed(fut):
            try:feeds.append(f.result())
            except Exception as e:diag['feed_failures']+=1;diag[f'feed_{type(e).__name__}']+=1
    games=[]
    for x in feeds:
        g=parse_game(*x)
        if g:games.append(g)
    games.sort(key=lambda g:(g['date'],g['start'],g['game_id']))
    diag['completed_schedule_games']=len(sched);diag['parsed_games']=len(games)

    needed={(e['defense_team'],e['date']) for g in games for e in g['events']}
    roster_cache={}
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut={ex.submit(fetch_roster,team,day):(team,day) for team,day in needed}
        for f in as_completed(fut):
            key=fut[f]
            try:roster_cache[key]=f.result()
            except Exception as e:roster_cache[key]=[];diag['roster_fetch_failures']+=1;diag[f'roster_{type(e).__name__}']+=1

    hist=defaultdict(deque);rows=[];groups=[]
    for g in games:
        day_ord=date.fromisoformat(g['date']).toordinal()
        priors={}
        teams=set(g['usage']) | {e['defense_team'] for e in g['events']}
        for team in teams:
            dq=hist[team]
            while dq and day_ord-dq[0][0]>30:dq.popleft()
            p=defaultdict(lambda:{'apps':0,'bf':0,'last_day':None,'late_apps':0,'save_like_apps':0})
            seen_pg=set()
            for d,pgid,pid,bf,fi,save_like in dq:
                k=(pgid,pid)
                if k in seen_pg:continue
                seen_pg.add(k);z=p[pid];z['apps']+=1;z['bf']+=bf;z['last_day']=d if z['last_day'] is None else max(z['last_day'],d)
                z['late_apps']+=int(fi>=7);z['save_like_apps']+=int(save_like)
            priors[team]=p
        for e in g['events']:
            pool=roster_cache.get((e['defense_team'],e['date']),[]);already=set(e['already_used_ids']);cands=[]
            for pid,throws in pool:
                if pid==e['outgoing_pitcher_id']:continue
                z=priors.get(e['defense_team'],{}).get(pid,{'apps':0,'bf':0,'last_day':None,'late_apps':0,'save_like_apps':0})
                apps=z['apps'];rest=(day_ord-z['last_day']) if z['last_day'] is not None else None
                cands.append({'pitcher_id':str(pid),'mlbam_id':pid,'throws':throws,
                    'prior_apps_30d':apps,'prior_bf_30d':z['bf'],'prior_rest_days':rest,
                    'prior_late_inning_share_30d':z['late_apps']/apps if apps else 0.0,
                    'prior_save_like_share_30d':z['save_like_apps']/apps if apps else 0.0,
                    'already_used_this_game':int(pid in already)})
            actual=e['actual_next_pitcher_id'];actual_in=any(c['mlbam_id']==actual for c in cands)
            diag['change_events']+=1;diag['actual_in_candidate_pool']+=int(actual_in);diag['nonempty_candidate_pool']+=int(bool(cands));diag['score_state_present']+=int(e['defensive_score_diff'] is not None)
            row={k:v for k,v in e.items() if k!='already_used_ids'};row['actual_next_pitcher_id']=str(actual)
            row.update({'actual_next_in_candidate_pool':int(actual_in),'candidate_count':len(cands),'candidates_json':json.dumps(cands,separators=(',',':'))});rows.append(row)
            if actual_in and len(cands)>=2 and e['defensive_score_diff'] is not None:
                sd=float(e['defensive_score_diff']);gr={'date':date.fromisoformat(e['date']),'rows':[]}
                for c in cands:
                    gr['rows'].append({'pitcher_id':str(c['pitcher_id']),'prior_apps_30d':float(c['prior_apps_30d']),
                        'prior_bf_30d':float(c['prior_bf_30d']),'prior_rest_days':None if c['prior_rest_days'] is None else float(c['prior_rest_days']),
                        'prior_late_inning_share_30d':float(c['prior_late_inning_share_30d']),
                        'prior_save_like_share_30d':float(c['prior_save_like_share_30d']),
                        'already_used_this_game':float(c['already_used_this_game']),'defensive_score_diff':sd,
                        'transition_kind':'in_inning','inning_band':sel.inning_band(e['decision_inning']),
                        'throws':c['throws'] or 'UNK','score_band':sel.score_band(sd),'y':int(c['mlbam_id']==actual)})
                if sum(r['y'] for r in gr['rows'])==1:groups.append(gr)
        for team,pusage in g['usage'].items():
            for pid,bf in pusage.items():
                fi=g['first_inning'][team].get(pid,-1);hist[team].append((day_ord,g['game_id'],pid,bf,fi,fi>=9))

    BASE.mkdir(parents=True,exist_ok=True)
    fields=['game_id','date','defense_team','transition_kind','inning','decision_inning','defensive_score_diff','outgoing_pitcher_id','actual_next_pitcher_id','actual_next_in_candidate_pool','candidate_count','candidates_json']
    with CAND_2026.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    n=max(1,diag['change_events']);summary={'market_blind':True,
        'source':'MLB Stats API schedule + game feed + exact-date active roster','year':YEAR,'through':END_DATE,
        'transition_scope':'adjacent same-half PA pitching changes, matching frozen 2025 candidate-builder semantics',
        'rows':len(rows),'actual_reliever_candidate_coverage':diag['actual_in_candidate_pool']/n,
        'nonempty_candidate_pool_rate':diag['nonempty_candidate_pool']/n,'score_state_coverage':diag['score_state_present']/n,
        'diagnostics':dict(diag)}
    SUM_2026.write_text(json.dumps(summary,indent=2))
    return groups,summary


def main():
    groups_2025=sel.load();train_2025,_,_=sel.split(groups_2025)
    if len(train_2025)<1000:raise SystemExit(f'insufficient frozen 2025 training choice sets: {len(train_2025)}')
    X,y=sel.flat(train_2025,sel.EXP_NUM,sel.EXP_CAT)
    model=sel.make_model(FROZEN_C,sel.EXP_NUM,sel.EXP_CAT);model.fit(X,y)
    groups_2026,candidate_summary=build_2026_groups()
    if len(groups_2026)<500:raise SystemExit(f'insufficient 2026 forward choice sets: {len(groups_2026)}')
    metrics=sel.evalm(model,groups_2026,sel.EXP_NUM,sel.EXP_CAT)
    gate={'candidate_coverage_ge_0_97':bool(candidate_summary['actual_reliever_candidate_coverage']>=.97),
          'nonempty_candidate_pool_ge_0_99':bool(candidate_summary['nonempty_candidate_pool_rate']>=.99),
          'score_state_coverage_ge_0_999':bool(candidate_summary['score_state_coverage']>=.999),
          'beats_uniform_logloss':bool(metrics['log_loss']<metrics['uniform_log_loss']),
          'beats_prior_usage_logloss':bool(metrics['log_loss']<metrics['prior_usage_log_loss']),
          'beats_prior_usage_top1':bool(metrics['top1_accuracy']>metrics['prior_usage_top1_accuracy']),
          'p99_lt_1000ms':bool(metrics['p99_inference_ms']<1000)}
    rep={'status':'FRESH_2026_FORWARD_TEST','market_blind':True,
         'frozen_before_2026':{'training':'2025 through 2025-07-31','C':FROZEN_C,'features':sel.EXP_NUM+sel.EXP_CAT,
                              'excluded_features':['team identity','pitcher identity','sportsbook/market inputs','future current-game usage']},
         'forward_window':{'start':'2026 regular season','through':END_DATE},'candidate_summary':candidate_summary,
         'forward_metrics':metrics,'promotion_gates':gate,'forward_gate_status':'PASS' if all(gate.values()) else 'BLOCKED',
         'next_gate':'downstream inning/remaining-game run-distribution improvement; reliever-choice forward pass alone does not authorize production promotion'}
    OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
    if not all(gate.values()):raise SystemExit('2026 reliever-selection forward gate blocked')

if __name__=='__main__':main()

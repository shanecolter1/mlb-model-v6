#!/usr/bin/env python3
"""Build leakage-safe historical candidate sets for conditional reliever selection."""
from __future__ import annotations
import csv, io, json, urllib.request
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import half_inning_scoring_gate as core
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/derived/model_calibration/bullpen_transitions';MIN_TEAM_COVERAGE=.999;MIN_ACTUAL_IN_POOL=.97;UA='mlb-model-v6 bullpen roster reconstruction'
def iv(r,k,d=-1):
    try:return int(float(r.get(k)))
    except:return d
def clean(v):return str(v or '').strip()
def ymd(ord_day):return date.fromordinal(int(ord_day)).isoformat()
def get_json(url,timeout=60):
    req=urllib.request.Request(url,headers={'User-Agent':UA})
    with urllib.request.urlopen(req,timeout=timeout) as resp:return json.loads(resp.read().decode('utf-8'))
def team_map():
    data=get_json('https://statsapi.mlb.com/api/v1/teams?sportId=1&season=2025');out={}
    for t in data.get('teams',[]):
        tid=int(t['id'])
        for k in ('teamCode','fileCode','abbreviation'):
            v=clean(t.get(k)).upper();
            if v:out[v]=tid
    return out
def chadwick_maps():
    retro_to_mlb={};mlb_to_retro={};rows=0
    for h in '0123456789abcdef':
        url=f'https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{h}.csv';req=urllib.request.Request(url,headers={'User-Agent':UA})
        with urllib.request.urlopen(req,timeout=60) as resp:
            txt=io.TextIOWrapper(resp,encoding='utf-8-sig',newline='')
            for r in csv.DictReader(txt):
                retro=clean(r.get('key_retro'));mlb=clean(r.get('key_mlbam'))
                if not retro or not mlb:continue
                try:mid=int(float(mlb))
                except Exception:continue
                retro_to_mlb[retro]=mid;mlb_to_retro[mid]=retro;rows+=1
    return retro_to_mlb,mlb_to_retro,rows
def fetch_active_pitchers(team_id,day_iso,mlb_to_retro):
    data=get_json(f'https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active&season=2025&date={day_iso}&hydrate=person');candidates=[];unmapped=0
    for x in data.get('roster',[]):
        pos=x.get('position') or {};person=x.get('person') or {};abbr=clean(pos.get('abbreviation')).upper();ptype=clean(pos.get('type')).lower()
        if abbr!='P' and ptype!='pitcher':continue
        try:mid=int(person['id'])
        except Exception:continue
        retro=mlb_to_retro.get(mid)
        if not retro:unmapped+=1;continue
        ph=person.get('pitchHand') or {};candidates.append((retro,mid,clean(ph.get('code')).upper() or None))
    return candidates,unmapped,len(data.get('roster',[]))
def score_diff_for_defense(r):
    sv=iv(r,'score_v',-999);sh=iv(r,'score_h',-999);tb=iv(r,'top_bot',-1)
    if sv==-999 or sh==-999 or tb not in (0,1):return None
    return (sh-sv) if tb==0 else (sv-sh)
def main():
    rows=core.fetch_rows_2025();tmap=team_map();retro_to_mlb,mlb_to_retro,cw_rows=chadwick_maps();diag=Counter();events=[];games=[];i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid'];j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=[r for r in rows[i:j] if core.truth(r.get('pa'))]
        if not game:i=j;continue
        usage=defaultdict(lambda:defaultdict(int));first_inning=defaultdict(dict);seen_so_far=defaultdict(set)
        for n,r in enumerate(game):
            tb=iv(r,'top_bot',-1);pid=clean(r.get('pitcher'));team=clean(r.get('pitteam')).upper()
            if tb not in (0,1) or not pid:continue
            inn=iv(r,'inning',iv(r,'inn_ct',-1))
            if team:
                usage[team][pid]+=1
                if pid not in first_inning[team]:first_inning[team][pid]=inn
            if n+1<len(game):
                nxt=game[n+1];tb2=iv(nxt,'top_bot',-1);npid=clean(nxt.get('pitcher'))
                if npid:
                    inn2=iv(nxt,'inning',iv(nxt,'inn_ct',-1));kind=None
                    if tb2==tb and inn2==inn and npid!=pid:kind='in_inning'
                    elif tb2==tb and inn2==inn+1 and npid!=pid:kind='between_inning'
                    if kind is not None:events.append({'game_id':gid,'day':day,'date':ymd(day),'defense_team':team,'transition_kind':kind,'inning':inn,'decision_inning':inn2,'defensive_score_diff':score_diff_for_defense(nxt),'outgoing_pitcher_id':pid,'actual_next_pitcher_id':npid,'already_used_ids':sorted(seen_so_far[team]|{pid})})
            seen_so_far[team].add(pid)
        games.append((day,gid,usage,first_inning));i=j
    needed={(e['defense_team'],e['date']) for e in events if e['defense_team']};valid=[];missing_team=[]
    for team,d in sorted(needed):
        tid=tmap.get(team)
        if tid is None:missing_team.append(team)
        else:valid.append((team,d,tid))
    diag['unique_team_dates']=len(needed);diag['team_dates_mapped']=len(valid);diag['team_dates_unmapped']=len(needed)-len(valid);roster_cache={};roster_unmapped=0
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut={ex.submit(fetch_active_pitchers,tid,d,mlb_to_retro):(team,d) for team,d,tid in valid}
        for f in as_completed(fut):
            key=fut[f]
            try:c,unm,total=f.result();roster_cache[key]=c;roster_unmapped+=unm
            except Exception as err:roster_cache[key]=[];diag['roster_fetch_failures']+=1;diag[f'roster_error_{type(err).__name__}']+=1
    hist=defaultdict(deque);event_by_game=defaultdict(list)
    for e in events:event_by_game[e['game_id']].append(e)
    out=[]
    for day,gid,usage,first_inning in games:
        priors={}
        for team in usage:
            dq=hist[team]
            while dq and day-dq[0][0]>30:dq.popleft()
            p=defaultdict(lambda:{'apps':0,'bf':0,'last_day':None,'late_apps':0,'save_like_apps':0});seen=set()
            for d,pgid,pid,bf,firstinn,save_like in dq:
                k=(pgid,pid)
                if k in seen:continue
                seen.add(k);z=p[pid];z['apps']+=1;z['bf']+=bf;z['last_day']=d if z['last_day'] is None else max(z['last_day'],d);z['late_apps']+=int(firstinn>=7);z['save_like_apps']+=int(save_like)
            priors[team]=p
        for e in event_by_game.get(gid,[]):
            team=e['defense_team'];pool=roster_cache.get((team,e['date']),[]);already=set(e['already_used_ids']);candidates=[]
            for retro,mid,throws in pool:
                if retro==e['outgoing_pitcher_id']:continue
                z=priors.get(team,{}).get(retro,{'apps':0,'bf':0,'last_day':None,'late_apps':0,'save_like_apps':0});rest=(day-z['last_day']) if z['last_day'] is not None else None;apps=max(1,z['apps'])
                candidates.append({'pitcher_id':retro,'mlbam_id':mid,'throws':throws,'prior_apps_30d':z['apps'],'prior_bf_30d':z['bf'],'prior_rest_days':rest,'prior_late_inning_share_30d':z['late_apps']/apps if z['apps'] else 0.0,'prior_save_like_share_30d':z['save_like_apps']/apps if z['apps'] else 0.0,'already_used_this_game':int(retro in already)})
            actual_in=any(c['pitcher_id']==e['actual_next_pitcher_id'] for c in candidates);diag['change_events']+=1;diag['actual_in_candidate_pool']+=int(actual_in);diag['nonempty_candidate_pool']+=int(bool(candidates));diag['score_state_present']+=int(e['defensive_score_diff'] is not None)
            row={k:v for k,v in e.items() if k not in ('day','already_used_ids')};row.update({'actual_next_in_candidate_pool':int(actual_in),'candidate_count':len(candidates),'candidates_json':json.dumps(candidates,separators=(',',':'))});out.append(row)
        for team,pusage in usage.items():
            for pid,bf in pusage.items():
                fi=first_inning[team].get(pid,-1);hist[team].append((day,gid,pid,bf,fi,fi>=9))
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'bullpen_reliever_candidate_sets_2025.csv';fields=['game_id','date','defense_team','transition_kind','inning','decision_inning','defensive_score_diff','outgoing_pitcher_id','actual_next_pitcher_id','actual_next_in_candidate_pool','candidate_count','candidates_json']
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    team_cov=diag['team_dates_mapped']/diag['unique_team_dates'] if diag['unique_team_dates'] else 0;actual_cov=diag['actual_in_candidate_pool']/diag['change_events'] if diag['change_events'] else 0;nonempty=diag['nonempty_candidate_pool']/diag['change_events'] if diag['change_events'] else 0;score_cov=diag['score_state_present']/diag['change_events'] if diag['change_events'] else 0
    rep={'market_blind':True,'candidate_definition':'MLB Stats API active roster on exact historical game date; pitchers only; current-game future usage prohibited','team_source':'Retrosheet pitteam at decision boundary','id_crosswalk_source':'Chadwick Register key_retro <-> key_mlbam','rows':len(out),'team_id_coverage':team_cov,'nonempty_candidate_pool_rate':nonempty,'actual_reliever_candidate_coverage':actual_cov,'score_state_coverage':score_cov,'chadwick_crosswalk_rows':cw_rows,'roster_pitchers_without_retro_id':roster_unmapped,'new_decision_features':['defensive_score_diff','decision_inning','already_used_this_game','pitcher throws','prior late-inning share','prior save-like share'],'diagnostics':dict(diag),'missing_team_codes':sorted(set(missing_team)),'promotion_gates':{'team_id_coverage_ge_0_999':team_cov>=MIN_TEAM_COVERAGE,'actual_reliever_candidate_coverage_ge_0_97':actual_cov>=MIN_ACTUAL_IN_POOL,'nonempty_candidate_pool_ge_0_99':nonempty>=.99,'score_state_coverage_ge_0_999':score_cov>=.999},'governance':'eligibility comes from exact-date active roster; all workload/role/same-game features are available by decision time; no candidate is added because he later appeared in the current game'}
    (OUT/'bullpen_reliever_candidate_summary_2025.json').write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
    if not all(rep['promotion_gates'].values()):raise SystemExit('Reliever candidate-set gate blocked')
if __name__=='__main__':main()

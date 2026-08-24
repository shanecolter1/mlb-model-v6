#!/usr/bin/env python3
"""Build market-blind pitcher workload/availability history for bullpen selection.

All workload features are computed using information available before the current
game. Uses prior appearances and batters faced from the same 2025 Retrosheet PA
stream. If true pitch-count fields are present in the source they are diagnosed
for later promotion; this builder never fabricates pitch counts from BF.
"""
from __future__ import annotations
import csv,json
from collections import defaultdict,Counter,deque
from pathlib import Path
import half_inning_scoring_gate as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/bullpen_transitions'

def iv(r,k,d=-1):
    try:return int(float(r.get(k)))
    except:return d

def main():
    rows=core.fetch_rows_2025(); pitch_fields=Counter(); pitch_samples={}
    for r in rows[:10000]:
        for k,v in r.items():
            lk=k.lower()
            if 'pitch' in lk and v not in (None,''):
                pitch_fields[k]+=1; pitch_samples.setdefault(k,str(v))
    # Aggregate each pitcher-game usage first.
    games=[];i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid'];j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=[r for r in rows[i:j] if core.truth(r.get('pa'))]
        usage=defaultdict(lambda:{'bf':0,'first_seen_order':10**9,'team_half':None})
        for n,r in enumerate(game):
            pid=str(r.get('pitcher') or '').strip();tb=iv(r,'top_bot',-1)
            if not pid or tb not in (0,1):continue
            usage[pid]['bf']+=1;usage[pid]['first_seen_order']=min(usage[pid]['first_seen_order'],n);usage[pid]['team_half']=tb
        games.append((day,gid,usage));i=j
    games.sort(key=lambda x:(x[0],x[1]))
    hist=defaultdict(deque); last_day={}; out=[]
    for day,gid,usage in games:
        for pid,u in usage.items():
            prior=[x for x in hist[pid] if x[0]<day]
            app1=sum(1 for d,bf in prior if day-d<=1);app3=sum(1 for d,bf in prior if day-d<=3);app7=sum(1 for d,bf in prior if day-d<=7)
            bf1=sum(bf for d,bf in prior if day-d<=1);bf3=sum(bf for d,bf in prior if day-d<=3);bf7=sum(bf for d,bf in prior if day-d<=7)
            rest=(day-last_day[pid]) if pid in last_day else None
            out.append({'game_id':gid,'date':str(day),'pitcher_id':pid,'prior_rest_days':rest,
                        'prior_appearances_1d':app1,'prior_appearances_3d':app3,'prior_appearances_7d':app7,
                        'prior_bf_1d':bf1,'prior_bf_3d':bf3,'prior_bf_7d':bf7,'current_game_bf':u['bf']})
        for pid,u in usage.items():
            hist[pid].append((day,u['bf'])); last_day[pid]=day
            while hist[pid] and day-hist[pid][0][0]>14:hist[pid].popleft()
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'bullpen_workload_history_2025.csv'
    fields=['game_id','date','pitcher_id','prior_rest_days','prior_appearances_1d','prior_appearances_3d','prior_appearances_7d','prior_bf_1d','prior_bf_3d','prior_bf_7d','current_game_bf']
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
    report={'market_blind':True,'rows':len(out),'leakage_control':'all workload predictors use games strictly before current game',
            'workload_features':['prior rest days','appearances over 1/3/7 days','BF over 1/3/7 days'],
            'pitch_count_schema_candidates':dict(pitch_fields),'pitch_count_samples':pitch_samples,
            'pitch_count_rule':'true pitch counts required when available; BF workload is retained as a separate empirical feature and is never relabeled as pitches',
            'next_phase':'join workload to reliever candidate sets and fit conditional reliever-selection model'}
    (OUT/'bullpen_workload_summary_2025.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

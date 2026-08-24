#!/usr/bin/env python3
"""Build market-blind between-inning pitcher continuation/change history.

Also emits a schema diagnostic for score/run-like Retrosheet fields so score state
can be wired from observed columns rather than guessed. This stage deliberately
keeps between-inning manager decisions separate from in-inning PA-boundary hazard.
"""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
import half_inning_scoring_gate as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/bullpen_transitions'

def iv(r,k,d=-1):
    try:return int(float(r.get(k)))
    except:return d

def main():
    rows=core.fetch_rows_2025()
    score_like=Counter(); samples={}
    for r in rows[:5000]:
        for k,v in r.items():
            lk=k.lower()
            if any(tok in lk for tok in ('score','run','home','away')):
                score_like[k]+=1
                if k not in samples and v not in (None,''): samples[k]=str(v)

    obs=[]; changes=0; continuations=0; games=0
    i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid'];j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=[r for r in rows[i:j] if core.truth(r.get('pa'))]
        games+=1
        by_def_half={0:defaultdict(list),1:defaultdict(list)}
        bf=defaultdict(int); first_pitcher={0:None,1:None}
        for r in game:
            tb=iv(r,'top_bot',-1); inn=iv(r,'inning',iv(r,'inn_ct',-1)); pid=str(r.get('pitcher') or '').strip()
            if tb not in (0,1) or inn<1 or not pid: continue
            if first_pitcher[tb] is None:first_pitcher[tb]=pid
            bf[pid]+=1; by_def_half[tb][inn].append(r)
        for tb in (0,1):
            innings=sorted(by_def_half[tb])
            for inn in innings:
                nxtinn=inn+1
                if nxtinn not in by_def_half[tb]:continue
                cur=by_def_half[tb][inn]; nxt=by_def_half[tb][nxtinn]
                end_pid=str(cur[-1].get('pitcher') or '').strip(); next_pid=str(nxt[0].get('pitcher') or '').strip()
                if not end_pid or not next_pid:continue
                changed=int(next_pid!=end_pid)
                changes+=changed;continuations+=(1-changed)
                last=cur[-1]
                row={
                    'game_id':gid,'date':str(day),'completed_inning':inn,'top_bot_defense':tb,
                    'pitcher_id':end_pid,'next_inning_pitcher_id':next_pid,
                    'pitcher_changed_before_next_defensive_inning':changed,
                    'pitcher_batters_faced_game_to_date':bf[end_pid],
                    'pitcher_is_game_first_for_defense':int(first_pitcher[tb]==end_pid),
                    'batters_in_completed_half_inning':len(cur),
                    'last_batter_lineup_position':iv(last,'lp',0),
                }
                obs.append(row)
        i=j
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'bullpen_between_inning_boundaries_2025.csv'
    fields=['game_id','date','completed_inning','top_bot_defense','pitcher_id','next_inning_pitcher_id','pitcher_changed_before_next_defensive_inning','pitcher_batters_faced_game_to_date','pitcher_is_game_first_for_defense','batters_in_completed_half_inning','last_batter_lineup_position']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(obs)
    report={
      'source':'2025 Retrosheet regular-season PA stream used by locked validation','market_blind':True,
      'games_processed':games,'between_inning_boundaries':len(obs),'pitcher_changes':changes,'pitcher_continuations':continuations,
      'change_rate':changes/len(obs) if obs else None,
      'score_schema_diagnostic':{'candidate_fields':sorted(score_like),'sample_nonempty_values':samples},
      'architecture':'between-inning removal is modeled separately from in-inning removal',
      'next_phase':'fit chronological between-inning hazard; then conditional reliever selection'
    }
    (OUT/'bullpen_between_inning_summary_2025.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

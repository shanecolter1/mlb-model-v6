#!/usr/bin/env python3
"""Build market-blind between-inning pitcher continuation/change history.

Features describe the decision moment before the next defensive inning begins.
Pitcher BF is cumulative only through the completed defensive inning (no future
leakage). Score state uses verified 2025 Retrosheet score_h/score_v fields from
the first PA of the next defensive inning.
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

def score_before(r):
    try:
        h=r.get('score_h'); v=r.get('score_v')
        if h not in (None,'') and v not in (None,''):
            return int(float(h)),int(float(v))
    except Exception:
        pass
    return None

def main():
    rows=core.fetch_rows_2025()
    obs=[]; changes=0; continuations=0; games=0; score_present=0
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
            bf[pid]+=1
            r['_bf_to_date']=bf[pid]
            by_def_half[tb][inn].append(r)
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

                # Decision moment is immediately before next defensive inning.
                state=nxt[0]
                sc=score_before(state); sd=None
                if sc is not None:
                    home,visitor=sc
                    sd=(home-visitor) if tb==0 else (visitor-home)
                    score_present+=1
                last=cur[-1]
                row={
                    'game_id':gid,'date':str(day),'completed_inning':inn,'top_bot_defense':tb,
                    'pitcher_id':end_pid,'next_inning_pitcher_id':next_pid,
                    'pitcher_changed_before_next_defensive_inning':changed,
                    'pitcher_batters_faced_game_to_date':iv(last,'_bf_to_date',-1),
                    'pitcher_is_game_first_for_defense':int(first_pitcher[tb]==end_pid),
                    'batters_in_completed_half_inning':len(cur),
                    'next_batter_lineup_position':iv(state,'lp',0),
                    'defensive_score_diff':sd,
                }
                obs.append(row)
        i=j
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'bullpen_between_inning_boundaries_2025.csv'
    fields=['game_id','date','completed_inning','top_bot_defense','pitcher_id','next_inning_pitcher_id','pitcher_changed_before_next_defensive_inning','pitcher_batters_faced_game_to_date','pitcher_is_game_first_for_defense','batters_in_completed_half_inning','next_batter_lineup_position','defensive_score_diff']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(obs)
    report={
      'source':'2025 Retrosheet regular-season PA stream used by locked validation','market_blind':True,
      'games_processed':games,'between_inning_boundaries':len(obs),'pitcher_changes':changes,'pitcher_continuations':continuations,
      'change_rate':changes/len(obs) if obs else None,
      'score_fields':{'home':'score_h','visitor':'score_v'},
      'score_state_coverage':score_present/len(obs) if obs else 0.0,
      'leakage_control':'pitcher BF is cumulative through completed defensive inning only',
      'architecture':'between-inning removal is modeled separately from in-inning removal',
      'next_phase':'fit chronological between-inning hazard; then conditional reliever selection'
    }
    (OUT/'bullpen_between_inning_summary_2025.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

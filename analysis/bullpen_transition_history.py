#!/usr/bin/env python3
"""Build market-blind historical pitcher-removal / reliever-entry transition data.

Uses the same 2025 Retrosheet PA stream already used by the locked validation.
Each PA boundary becomes an observation of whether the current pitcher continues
or is replaced before the next PA. No sportsbook or market-derived inputs.

This is intentionally a data/diagnostic phase, not yet a fitted bullpen model.
"""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import half_inning_scoring_gate as core

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/bullpen_transitions'


def iv(r,k,default=-1):
    try:return int(r.get(k))
    except Exception:return default


def score_before(r):
    # Retrosheet-derived schemas have varied during ingestion; preserve missingness.
    for a,b in [('home_score','away_score'),('score_home','score_away'),('home_score_pre','away_score_pre')]:
        if r.get(a) not in (None,'') and r.get(b) not in (None,''):
            try:return int(r[a]),int(r[b])
            except Exception:pass
    return None


def main():
    rows=core.fetch_rows_2025()
    obs=[]; summary=Counter(); reliever_entries=Counter(); games=0
    i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid']; j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=[r for r in rows[i:j] if core.truth(r.get('pa'))]
        games+=1
        # Pitcher BF-to-date inside this game and team/half sequence context.
        bf=defaultdict(int)
        # Track first pitcher seen for each defensive half-team as starter proxy.
        first_pitcher={0:None,1:None}
        for n,r in enumerate(game):
            pid=str(r.get('pitcher') or '').strip(); tb=iv(r,'top_bot',-1)
            if pid and tb in (0,1) and first_pitcher[tb] is None:first_pitcher[tb]=pid
            if pid:bf[pid]+=1
            if n+1>=len(game):continue
            nxt=game[n+1]
            # Only compare adjacent PAs within same half-inning. End-of-half removals
            # are handled separately by looking at the first PA of the next defensive half.
            inn=iv(r,'inning',iv(r,'inn_ct',-1)); inn2=iv(nxt,'inning',iv(nxt,'inn_ct',-1))
            tb2=iv(nxt,'top_bot',-1)
            same_half=(inn==inn2 and tb==tb2 and tb in (0,1))
            if not same_half:continue
            npid=str(nxt.get('pitcher') or '').strip()
            if not pid or not npid:continue
            changed=(npid!=pid)
            outs=iv(r,'outs_pre',-1)
            lp=iv(r,'lp',0)
            bases=sum((1<<bit) for bit,k in enumerate(('r1','r2','r3')) if str(r.get(k) or '').strip())
            sc=score_before(r)
            sd=None
            if sc is not None:
                home,away=sc
                # defense is home in top, away in bottom
                sd=(home-away) if tb==0 else (away-home)
            row={
                'game_id':gid,'date':str(day),'inning':inn,'top_bot':tb,
                'outs_pre':outs,'bases_mask':bases,'lineup_position':lp,
                'pitcher_id':pid,'next_pitcher_id':npid,'pitcher_changed_before_next_pa':int(changed),
                'pitcher_batters_faced_game_to_date':bf[pid],
                'pitcher_is_game_first_for_defense':int(first_pitcher.get(tb)==pid),
                'defensive_score_diff':sd,
            }
            obs.append(row); summary['pa_boundaries']+=1
            if changed:
                summary['in_inning_pitcher_changes']+=1; reliever_entries[npid]+=1
        i=j

    OUT.mkdir(parents=True,exist_ok=True)
    csv_path=OUT/'bullpen_transition_pa_boundaries_2025.csv'
    fields=['game_id','date','inning','top_bot','outs_pre','bases_mask','lineup_position','pitcher_id','next_pitcher_id','pitcher_changed_before_next_pa','pitcher_batters_faced_game_to_date','pitcher_is_game_first_for_defense','defensive_score_diff']
    with csv_path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(obs)
    rate=summary['in_inning_pitcher_changes']/summary['pa_boundaries'] if summary['pa_boundaries'] else 0.0
    report={
        'source':'2025 Retrosheet regular-season PA stream used by locked validation',
        'market_blind':True,
        'games_processed':games,
        'pa_boundaries':summary['pa_boundaries'],
        'in_inning_pitcher_changes':summary['in_inning_pitcher_changes'],
        'in_inning_change_rate_per_pa_boundary':rate,
        'unique_relievers_entered':len(reliever_entries),
        'top_reliever_entries':reliever_entries.most_common(25),
        'available_features':['inning','half','outs','bases','lineup position','pitcher batters faced to date','starter-proxy flag','score differential when present'],
        'not_available_historically':['bullpen warming signal','manager intent','real-time bullpen phone activity'],
        'next_phase':'fit removal hazard and conditional reliever-selection models with temporal/season holdout governance',
    }
    (OUT/'bullpen_transition_summary_2025.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

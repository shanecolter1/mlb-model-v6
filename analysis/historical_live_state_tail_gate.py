#!/usr/bin/env python3
"""Historical realistic-state tail gate for live/full-game promotion.

Uses 2025 Retrosheet regular-season PA states and the frozen end-2024 PA model.
Measures unresolved probability at the existing 18-PA cap from actual base/out/
lineup states. This is a computational-governance test; no sportsbook inputs.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
import numpy as np
import half_inning_scoring_gate as core
from live_half_inning_state_engine import live_half_inning_distribution

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/batter_pitcher_blend'
TRANS=ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json'
MAX_STATES=50000


def main():
    params=json.loads((BASE/'joint_multinomial_pa_model.json').read_text())
    state=json.loads((BASE/'end_2024_skill_state.json').read_text())
    table=json.loads(TRANS.read_text())
    hp=params['selected_hyperparameters']; h=float(hp['half_life_days']); prior=float(hp['prior_strength'])
    rows=core.fetch_rows_2025(); bat=core.Store(h,state['batter']); pit=core.Store(h,state['pitcher']); league=core.Store(h,state['league'])
    vals=[]; skip=Counter(); i=0
    while i<len(rows) and len(vals)<MAX_STATES:
        day,gid=rows[i]['_day'],rows[i]['_gid']; j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=rows[i:j]; lr=core.norm(league.get('league',day)+1.0); bcache={}; pcache={}; lineups={0:{},1:{}}; modeled=[]
        for r in game:
            if not core.truth(r.get('pa')):continue
            bid=str(r.get('batter') or '').strip(); pid=str(r.get('pitcher') or '').strip(); tb=core.intval(r.get('top_bot'),-1); lp=core.intval(r.get('lp'),0)
            if not bid or not pid or tb not in (0,1):continue
            if 1<=lp<=9 and lp not in lineups[tb]:lineups[tb][lp]=bid
            if bid not in bcache:bcache[bid]=core.norm(bat.get(bid,day)+prior*lr)
            if pid not in pcache:pcache[pid]=core.norm(pit.get(pid,day)+prior*lr)
            ev=core.classify(r)
            if ev is not None:modeled.append((ev,bid,pid))
        for tb in (0,1):
            if len(lineups[tb])!=9:continue
            order=[lineups[tb][k] for k in range(1,10)]
            team=[r for r in game if core.truth(r.get('pa')) and core.intval(r.get('top_bot'))==tb]
            for r in team:
                if len(vals)>=MAX_STATES:break
                bid=str(r.get('batter') or '').strip(); pid=str(r.get('pitcher') or '').strip(); lp=core.intval(r.get('lp'),0)
                outs=core.intval(r.get('outs_pre'),-1)
                if pid not in pcache or bid not in bcache or not(1<=lp<=9) or outs not in (0,1,2):skip['missing_state']+=1;continue
                mask=0
                for bit,key in enumerate(('r1','r2','r3')):
                    if str(r.get(key) or '').strip():mask|=(1<<bit)
                pr=pcache[pid]; lineup=[]
                for obid in order:
                    br=bcache.get(obid,core.norm(bat.get(obid,day)+prior*lr)); lineup.append(core.model_prob(br,pr,lr,params))
                try:
                    res=live_half_inning_distribution(
                        np.asarray(lineup), current_batter_idx=lp-1, outs=outs,
                        bases_mask=mask, runs_already=0, table=table,
                        max_remaining_pa=18,
                    )
                    vals.append(float(res['unresolved_probability']))
                except Exception as exc:
                    skip[f'engine_error:{type(exc).__name__}']+=1
        for ev,bid,pid in modeled:
            k=core.CLASSES.index(ev);bat.add(bid,day,k);pit.add(pid,day,k);league.add('league',day,k)
        i=j
    if not vals:raise RuntimeError(f'no historical states evaluated; skip={dict(skip)}')
    a=np.asarray(vals)
    result={'historical_states':int(a.size),'source':'2025 Retrosheet regular season actual pre-PA base/out/lineup states','max_unresolved_probability':float(a.max()),'p99_unresolved_probability':float(np.percentile(a,99)),'p999_unresolved_probability':float(np.percentile(a,99.9)),'mean_unresolved_probability':float(a.mean()),'states_over_1e-9':int((a>1e-9).sum()),'states_over_1e-6':int((a>1e-6).sum()),'skip':dict(skip)}
    result['gate_status']='PASS' if result['max_unresolved_probability']<=1e-6 else 'BLOCKED'
    out=BASE/'historical_live_state_tail_validation.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if result['gate_status']!='PASS':raise SystemExit('Historical 18-PA tail gate blocked')
if __name__=='__main__':main()

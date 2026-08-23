#!/usr/bin/env python3
from __future__ import annotations
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json'
EVENTS=['out','bb','single','double','triple','hr']

def assert_close(x,y,tol=1e-9):
    if abs(x-y)>tol: raise AssertionError((x,y))

def dist_for_state(states,event,outs,mask):
    rows=states[f'{event}|{outs}|{mask}']
    s=sum(float(r['p']) for r in rows)
    assert_close(s,1.0,1e-8)
    for r in rows:
        oa=int(r['outs_added']); pm=int(r['post_mask']); runs=int(r['runs']); p=float(r['p'])
        assert 0<=oa<=3-outs
        assert 0<=pm<=7
        assert runs>=0
        assert math.isfinite(p) and p>=0
        if outs+oa>=3: assert pm==0
    return rows

def recurse(states,probs,outs=0,mask=0,depth=0,memo=None):
    memo={} if memo is None else memo
    if outs>=3 or depth>=18:return [1.0,0,0,0,0,0,0]
    key=(outs,mask,depth)
    if key in memo:return memo[key]
    ans=[0.0]*7
    for ev,pk in probs.items():
        for t in states[f'{ev}|{outs}|{mask}']:
            no=min(3,outs+int(t['outs_added'])); nm=0 if no>=3 else int(t['post_mask']); runs=int(t['runs'])
            nxt=recurse(states,probs,no,nm,depth+1,memo)
            w=pk*float(t['p'])
            for k,v in enumerate(nxt): ans[min(6,runs+k)]+=w*v
    total=sum(ans)
    assert total>0 and math.isfinite(total)
    ans=[v/total for v in ans]
    memo[key]=ans
    return ans

def main():
    m=json.loads(MODEL.read_text())
    assert m['market_inputs_used'] is False
    assert m['training_years']==[2021,2022,2023]
    assert m['selection_year']==[2024]
    assert m['locked_validation_year']==[2025]
    assert m['validation_2025']['logloss_shrunk'] < m['validation_2025']['logloss_raw']
    states=m['states']
    expected={f'{e}|{o}|{b}' for e in EVENTS for o in range(3) for b in range(8)}
    assert set(states)==expected, (len(states),len(expected),sorted(expected-set(states))[:5])
    for e in EVENTS:
        for o in range(3):
            for b in range(8): dist_for_state(states,e,o,b)
    # League-like deterministic test mixture; purpose is recursion integrity, not fitting.
    probs={'out':0.69,'bb':0.085,'single':0.145,'double':0.045,'triple':0.004,'hr':0.031}
    assert_close(sum(probs.values()),1.0)
    for o in range(3):
        for b in range(8):
            d=recurse(states,probs,o,b)
            assert len(d)==7
            assert all(math.isfinite(x) and x>=0 for x in d)
            assert_close(sum(d),1.0,1e-8)
    # Confirm the empirical model materially differs from the legacy fixed runner rules.
    dbl0=states['double|0|1']; dbl2=states['double|2|1']
    pscore0=sum(float(r['p']) for r in dbl0 if int(r['runs'])>=1)
    pscore2=sum(float(r['p']) for r in dbl2 if int(r['runs'])>=1)
    assert abs(pscore0-.45)>.05
    assert abs(pscore2-.45)>.05
    assert pscore2>pscore0
    print(json.dumps({'status':'PASS','states':len(states),'validation_2025':m['validation_2025'],'double_1b_score_p_0out':pscore0,'double_1b_score_p_2out':pscore2},indent=2))

if __name__=='__main__': main()

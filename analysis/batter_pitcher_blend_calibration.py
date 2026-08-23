#!/usr/bin/env python3
"""Leakage-safe batter/pitcher event blend calibration from Retrosheet parsed plays.

Governance:
- 2021-2023 development
- 2024 hyperparameter/weight selection
- 2025 locked validation

For each PA, player event rates use only prior PAs from the same season. Rates are
empirical-Bayes shrunk to the season-to-date league rate. The batter blend weight
is tuned independently for bb, single, double, triple, hr; out is residual.
"""
from __future__ import annotations
import csv, io, json, math, urllib.request, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/batter_pitcher_blend'
YEARS=[2021,2022,2023,2024,2025]
EVENTS=['bb','single','double','triple','hr']
WEIGHTS=[i/20 for i in range(21)]
PRIOR_STRENGTHS=[25,50,100,150,200]
EPS=1e-9

def truth(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
def classify(r):
    if not truth(r.get('pa')): return None
    if truth(r.get('single')): return 'single'
    if truth(r.get('double')): return 'double'
    if truth(r.get('triple')): return 'triple'
    if truth(r.get('hr')): return 'hr'
    if truth(r.get('walk')) and not truth(r.get('hbp')): return 'bb'
    return 'out'
def is_regular(g):
    s=str(g or '').strip().lower().replace('_',' ').replace('-',' ')
    return ('regular' in s) or s in {'r','rs','0'}
def fetch_year(y):
    url=f'https://www.retrosheet.org/downloads/plays/{y}plays.zip'
    req=urllib.request.Request(url,headers={'User-Agent':'mlb-model-v6 empirical research'})
    with urllib.request.urlopen(req,timeout=120) as resp: raw=resp.read()
    rows=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if n.lower().endswith('.csv'):
                with z.open(n) as f: rows.extend(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')))
    return url,rows

def ids(r):
    # Retrosheet parsed plays expose batter/pitcher ids under these common names.
    bid=r.get('batter') or r.get('batter_id') or r.get('bat_id') or r.get('batterid')
    pid=r.get('pitcher') or r.get('pitcher_id') or r.get('pit_id') or r.get('pitcherid')
    return str(bid or '').strip(),str(pid or '').strip()

def date_key(r):
    for k in ('date','game_date','game_dt'):
        v=str(r.get(k,'')).strip()
        if v: return v[:10]
    gid=str(r.get('game_id') or r.get('gameid') or '')
    # Retrosheet IDs conventionally carry MMDD in positions after team/year; lexical row order still prevents future-PA leakage.
    return gid

def build_examples(year,prior_strength):
    url,rows=fetch_year(year)
    rows=[r for r in rows if is_regular(r.get('gametype'))]
    # Preserve source order; snapshots update only after each PA, so current/future PA is never included.
    batter=defaultdict(Counter); pitcher=defaultdict(Counter); league=Counter(); out=[]
    accepted=0
    for r in rows:
        ev=classify(r); bid,pid=ids(r)
        if ev is None or not bid or not pid: continue
        total=sum(league.values())
        if total>=500:
            league_rate={e:league[e]/total for e in EVENTS}
            def rate(store,e):
                c=store[e]; n=sum(store.values())
                return (c+prior_strength*league_rate[e])/(n+prior_strength)
            ex={'year':year,'date':date_key(r),'event':ev,'batter_id':bid,'pitcher_id':pid}
            for e in EVENTS:
                ex[f'b_{e}']=rate(batter[bid],e); ex[f'p_{e}']=rate(pitcher[pid],e); ex[f'l_{e}']=league_rate[e]
            out.append(ex)
        batter[bid][ev]+=1; pitcher[pid][ev]+=1; league[ev]+=1; accepted+=1
    return url,out,accepted

def clip(p): return min(1-EPS,max(EPS,p))
def logloss(examples,event,w):
    ll=0.0
    for x in examples:
        p=clip(w*x[f'b_{event}']+(1-w)*x[f'p_{event}'])
        y=1.0 if x['event']==event else 0.0
        ll-=y*math.log(p)+(1-y)*math.log(1-p)
    return ll/max(1,len(examples))
def tune(train24_by_prior):
    grid=[]; best=None
    for prior,ex in train24_by_prior.items():
        for e in EVENTS:
            for w in WEIGHTS:
                ll=logloss(ex,e,w)
                row={'prior_strength':prior,'event':e,'batter_weight':w,'logloss_2024':ll}; grid.append(row)
    # choose one prior globally by sum of each event's best 2024 ll; then per-event weights
    score_prior={}
    for prior in PRIOR_STRENGTHS:
        score_prior[prior]=sum(min(r['logloss_2024'] for r in grid if r['prior_strength']==prior and r['event']==e) for e in EVENTS)
    chosen=min(score_prior,key=score_prior.get)
    weights={e:min((r for r in grid if r['prior_strength']==chosen and r['event']==e),key=lambda r:r['logloss_2024'])['batter_weight'] for e in EVENTS}
    return chosen,weights,grid,score_prior

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    examples={}; provenance=[]
    for prior in PRIOR_STRENGTHS:
        examples[prior]={}
        for y in YEARS:
            url,ex,n=build_examples(y,prior); examples[prior][y]=ex
            if prior==PRIOR_STRENGTHS[0]: provenance.append({'year':y,'source_url':url,'accepted_pa':n,'examples':len(ex)})
    tune_by_prior={p:examples[p][2024] for p in PRIOR_STRENGTHS}
    prior,weights,grid,prior_scores=tune(tune_by_prior)
    val=examples[prior][2025]
    validation={}
    for e in EVENTS:
        fitted=logloss(val,e,weights[e]); legacy=logloss(val,e,.68); batter_only=logloss(val,e,1.0); pitcher_only=logloss(val,e,0.0)
        validation[e]={'batter_weight':weights[e],'pitcher_weight':1-weights[e],'logloss_fitted':fitted,'logloss_legacy_68_32':legacy,'delta_vs_legacy':legacy-fitted,'logloss_batter_only':batter_only,'logloss_pitcher_only':pitcher_only}
    passes=[v['logloss_fitted']<=v['logloss_legacy_68_32'] for v in validation.values()]
    manifest={'component':'batter/pitcher event blend','governance_status':'PASS' if all(passes) else 'WARNING','development_years':[2021,2022,2023],'selection_year':[2024],'locked_validation_year':[2025],'market_inputs_used':False,'prior_strength_selected':prior,'prior_scores_2024':prior_scores,'event_weights':weights,'validation_2025':validation,'provenance':provenance,'promotion_rule':'Each fitted event weight must not worsen locked-2025 binary log loss versus legacy 0.68 batter / 0.32 pitcher.'}
    (OUT/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'production_event_blend_weights.json').write_text(json.dumps({'version':'empirical-event-blend-v1','prior_strength':prior,'weights':weights,'validation_2025':validation},indent=2))
    with (OUT/'blend_grid_2024.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['prior_strength','event','batter_weight','logloss_2024']);w.writeheader();w.writerows(grid)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
"""Locked-2025 full-game half-inning scoring gate for the joint PA model.

Uses the persisted end-2024 decayed skill state from the PA calibration, then processes
only 2025 chronologically. The validation observations, model, empirical transition table,
and promotion metrics are unchanged; this file only removes redundant historical rebuilds
and caches repeated game/pitcher/start-position run distributions.
"""
from __future__ import annotations
import csv, io, json, math, urllib.request, zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/batter_pitcher_blend'
TRANS=ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json'
CLASSES=['out','bb','single','double','triple','hr'];CAP=6;EPS=1e-9;MIN_HALF_INNINGS=5000

def truth(v):return str(v).strip().lower() in {'1','true','t','yes','y'}
def intval(v,d=0):
    try:return int(float(v))
    except:return d
def is_regular(g):
    s=str(g or '').strip().lower().replace('_',' ').replace('-',' ');return ('regular' in s) or s in {'r','rs','0'}
def classify(r):
    if not truth(r.get('pa')):return None
    if truth(r.get('single')):return 'single'
    if truth(r.get('double')):return 'double'
    if truth(r.get('triple')):return 'triple'
    if truth(r.get('hr')):return 'hr'
    if truth(r.get('walk')) and not truth(r.get('hbp')):return 'bb'
    if any(truth(r.get(k)) for k in ('hbp','xi','roe','noout','k_safe')):return None
    if truth(r.get('k')) or intval(r.get('outs_post'))>intval(r.get('outs_pre')):return 'out'
    return None
def game_day(gid):
    digits=''.join(c for c in str(gid) if c.isdigit())
    if len(digits)>=8:return datetime.strptime(digits[:8],'%Y%m%d').date().toordinal()
    raise ValueError(gid)
def fetch_rows_2025():
    url='https://www.retrosheet.org/downloads/plays/2025plays.zip';req=urllib.request.Request(url,headers={'User-Agent':'mlb-model-v6 locked 2025 validation'})
    with urllib.request.urlopen(req,timeout=120) as resp:raw=resp.read()
    out=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if not n.lower().endswith('.csv'):continue
            with z.open(n) as f:
                for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')):
                    if not is_regular(r.get('gametype')):continue
                    gid=str(r.get('gid') or r.get('game_id') or '').strip()
                    if not gid:continue
                    r['_gid']=gid;r['_day']=game_day(gid);out.append(r)
    out.sort(key=lambda r:(r['_day'],r['_gid']));return out

class Store:
    def __init__(self,h,data=None):
        self.h=float(h);self.d={}
        if data:self.d={k:(np.asarray(v['vec'],dtype=float),int(v['last_day'])) for k,v in data.items()}
    def get(self,k,day):
        v,last=self.d.get(k,(np.zeros(6),day));return (v*(0.5**((day-last)/self.h)) if day>last else v).copy()
    def add(self,k,day,idx):v=self.get(k,day);v[idx]+=1;self.d[k]=(v,day)
def norm(v):
    s=float(np.sum(v));return np.asarray(v,dtype=float)/s if s>0 else np.full(6,1/6)
def softmax(z):z=np.asarray(z,dtype=float);z-=np.max(z);e=np.exp(z);return e/e.sum()
def model_prob(br,pr,lr,params):
    feat=[]
    for k in range(1,6):feat.append(math.log(max(EPS,br[k])/max(EPS,lr[k])))
    for k in range(1,6):feat.append(math.log(max(EPS,pr[k])/max(EPS,lr[k])))
    for k in range(1,6):feat.append(math.log(max(EPS,lr[k])/max(EPS,lr[0])))
    raw=softmax(np.asarray(params['coef'])@np.asarray(feat)+np.asarray(params['intercept']));out=np.zeros(6)
    for pos,cls_idx in enumerate(params['sklearn_classes']):out[int(cls_idx)]=raw[pos]
    return out
def legacy_prob(br,pr):return norm(.68*br+.32*pr)
def run_dist(lineup_probs,start_idx,table):
    memo={}
    def rec(o,m,i,depth):
        if o>=3 or depth>=18:return np.array([1.,0,0,0,0,0,0])
        key=(o,m,i,depth)
        if key in memo:return memo[key]
        ans=np.zeros(7);ni=(i+1)%9
        for ev,p in zip(CLASSES,lineup_probs[i]):
            for t in table['states'][f'{ev}|{o}|{m}']:
                no=min(3,o+intval(t['outs_added']));nm=intval(t['post_mask']);runs=intval(t['runs']);w=float(p)*float(t['p']);nxt=rec(no,nm,ni,depth+1)
                for k,v in enumerate(nxt):ans[min(CAP,runs+k)]+=w*v
        s=ans.sum();ans=ans/s if s else np.array([1.,0,0,0,0,0,0]);memo[key]=ans;return ans
    return rec(0,0,start_idx,0)
def ll(obs,p):return -math.log(max(EPS,float(p[min(CAP,obs)])))
def brier(obs,p):
    y=np.zeros(7);y[min(CAP,obs)]=1;return float(np.sum((p-y)**2))

def main():
    params=json.loads((BASE/'joint_multinomial_pa_model.json').read_text());manifest=json.loads((BASE/'model_development_manifest.json').read_text())
    if manifest['governance_status']!='PA_GATE_PASS_HALF_INNING_PENDING':raise SystemExit(f"PA gate not passed: {manifest['governance_status']}")
    if not manifest.get('convergence',{}).get('converged'):raise SystemExit('PA model convergence gate not passed')
    table=json.loads(TRANS.read_text());state=json.loads((BASE/'end_2024_skill_state.json').read_text());hp=params['selected_hyperparameters'];h=float(hp['half_life_days']);prior=float(hp['prior_strength'])
    if float(state['half_life_days'])!=h:raise RuntimeError('End-2024 state half-life mismatch')
    rows=fetch_rows_2025();bat=Store(h,state['batter']);pit=Store(h,state['pitcher']);league=Store(h,state['league']);records=[];skip=Counter();i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid'];j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=rows[i:j];lr=norm(league.get('league',day)+1.0);bcache={};pcache={};lineups={0:{},1:{}};all_pa=[];modeled=[]
        for r in game:
            if not truth(r.get('pa')):continue
            bid=str(r.get('batter') or '').strip();pid=str(r.get('pitcher') or '').strip();tb=intval(r.get('top_bot'),-1)
            if not bid or not pid or tb not in (0,1):continue
            if bid not in bcache:bcache[bid]=norm(bat.get(bid,day)+prior*lr)
            if pid not in pcache:pcache[pid]=norm(pit.get(pid,day)+prior*lr)
            lp=intval(r.get('lp'),0)
            if 1<=lp<=9 and lp not in lineups[tb]:lineups[tb][lp]=bid
            all_pa.append(r);ev=classify(r)
            if ev is not None:modeled.append((ev,bid,pid))
        # Cache probability lineups and recursion results within this game.
        lineup_prob_cache={};dist_cache={}
        for tb in (0,1):
            if len(lineups[tb])!=9:skip['incomplete_original_lineup']+=1;continue
            order=[lineups[tb][k] for k in range(1,10)];team_pa=[r for r in all_pa if intval(r.get('top_bot'))==tb]
            for inn in range(1,10):
                half=[r for r in team_pa if intval(r.get('inning'))==inn]
                if not half:continue
                pids={str(r.get('pitcher') or '').strip() for r in half}
                if len(pids)!=1:skip['pitcher_change_within_half']+=1;continue
                pid=next(iter(pids))
                if not pid or pid not in pcache:skip['missing_pitcher_skill']+=1;continue
                before=sum(1 for r in team_pa if intval(r.get('inning'))<inn);start=before%9
                actual_batters=[str(r.get('batter') or '').strip() for r in half];expected=[order[(start+n)%9] for n in range(len(actual_batters))]
                if actual_batters!=expected:skip['batting_order_substitution_or_mismatch']+=1;continue
                key=(tb,pid)
                if key not in lineup_prob_cache:
                    pr=pcache[pid];cand=[];leg=[]
                    for bid in order:
                        br=bcache.get(bid,norm(bat.get(bid,day)+prior*lr));cand.append(model_prob(br,pr,lr,params));leg.append(legacy_prob(br,pr))
                    lineup_prob_cache[key]=(cand,leg)
                dkey=(tb,pid,start)
                if dkey not in dist_cache:
                    cand,leg=lineup_prob_cache[key];dist_cache[dkey]=(run_dist(cand,start,table),run_dist(leg,start,table))
                pd1,pd0=dist_cache[dkey];actual=sum(max(0,intval(r.get('runs'))) for r in half);unmodeled=sum(1 for r in half if classify(r) is None)
                records.append({'gid':gid,'inning':inn,'top_bot':tb,'actual_runs':actual,'unmodeled_pa':unmodeled,'joint_ll':ll(actual,pd1),'legacy_ll':ll(actual,pd0),'joint_brier':brier(actual,pd1),'legacy_brier':brier(actual,pd0),'joint_p_score':1-pd1[0],'legacy_p_score':1-pd0[0]})
        for ev,bid,pid in modeled:
            k=CLASSES.index(ev);bat.add(bid,day,k);pit.add(pid,day,k);league.add('league',day,k)
        i=j
    if len(records)<MIN_HALF_INNINGS:raise RuntimeError(f'Insufficient eligible 2025 half-innings: {len(records)} < {MIN_HALF_INNINGS}')
    inning_counts=Counter(r['inning'] for r in records)
    if any(inning_counts[k]==0 for k in range(1,10)):raise RuntimeError(f'Missing inning coverage: {dict(inning_counts)}')
    joint_ll=float(np.mean([r['joint_ll'] for r in records]));legacy_ll=float(np.mean([r['legacy_ll'] for r in records]));joint_br=float(np.mean([r['joint_brier'] for r in records]));legacy_br=float(np.mean([r['legacy_brier'] for r in records]))
    actual_score=np.asarray([r['actual_runs']>0 for r in records],dtype=float);jp=np.asarray([r['joint_p_score'] for r in records]);lp=np.asarray([r['legacy_p_score'] for r in records]);sbj=float(np.mean((jp-actual_score)**2));sbl=float(np.mean((lp-actual_score)**2));unmodeled_total=sum(r['unmodeled_pa'] for r in records)
    result={'sample_half_innings':len(records),'scope':'2025 regular-season innings 1-9; original batting order intact; one pitcher for entire half-inning','inning_counts':dict(sorted(inning_counts.items())),'skipped_half_innings':dict(skip),'unmodeled_pa_total':unmodeled_total,'unmodeled_pa_per_half_inning':float(unmodeled_total/len(records)),'joint_multinomial':{'run_bucket_logloss':joint_ll,'run_bucket_brier':joint_br,'score_any_run_brier':sbj},'legacy_68_32':{'run_bucket_logloss':legacy_ll,'run_bucket_brier':legacy_br,'score_any_run_brier':sbl},'delta_vs_legacy':{'run_bucket_logloss':legacy_ll-joint_ll,'run_bucket_brier':legacy_br-joint_br,'score_any_run_brier':sbl-sbj}}
    pass_gate=(joint_ll<legacy_ll and joint_br<=legacy_br and sbj<=sbl);result['gate_status']='PASS' if pass_gate else 'BLOCKED'
    (BASE/'half_inning_validation_2025.json').write_text(json.dumps(result,indent=2));manifest['half_inning_validation_2025']=result;manifest['governance_status']='PASS' if pass_gate else 'BLOCKED_HALF_INNING_GATE';manifest['production_eligible']=bool(pass_gate);(BASE/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2));params['half_inning_validation_2025']=result;params['production_eligible']=bool(pass_gate);(BASE/'joint_multinomial_pa_model.json').write_text(json.dumps(params,indent=2));print(json.dumps(result,indent=2))
    if not pass_gate:raise SystemExit('HALF-INNING MODEL BLOCKED: joint PA model did not clear all locked-2025 scoring gates')
if __name__=='__main__':main()

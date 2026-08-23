#!/usr/bin/env python3
"""Locked-2025 early-half-inning scoring gate for the joint PA model.

Evaluates I1/I2 run distributions using pregame-frozen player skill, the frozen joint PA
model, and the promoted empirical transition engine. HBP/ROE/interference/no-out PA classes
remain unmodeled in the six-state engine, but they are retained when reconstructing batting
order progression and actual half-inning scoring so their omission cannot be hidden.
"""
from __future__ import annotations
import csv, io, json, math, urllib.request, zipfile
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/batter_pitcher_blend'
TRANS=ROOT/'data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json'
YEARS=[2021,2022,2023,2024,2025]
CLASSES=['out','bb','single','double','triple','hr']
CAP=6
EPS=1e-9

def truth(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
def intval(v,d=0):
    try:return int(float(v))
    except:return d
def is_regular(g):
    s=str(g or '').strip().lower().replace('_',' ').replace('-',' ')
    return ('regular' in s) or s in {'r','rs','0'}
def classify(r):
    if not truth(r.get('pa')): return None
    if truth(r.get('single')): return 'single'
    if truth(r.get('double')): return 'double'
    if truth(r.get('triple')): return 'triple'
    if truth(r.get('hr')): return 'hr'
    if truth(r.get('walk')) and not truth(r.get('hbp')): return 'bb'
    if truth(r.get('hbp')) or truth(r.get('xi')) or truth(r.get('roe')) or truth(r.get('noout')) or truth(r.get('k_safe')): return None
    if truth(r.get('k')): return 'out'
    if intval(r.get('outs_post'))>intval(r.get('outs_pre')): return 'out'
    return None
def game_day(gid):
    digits=''.join(c for c in str(gid) if c.isdigit())
    if len(digits)>=8:return datetime.strptime(digits[:8],'%Y%m%d').date().toordinal()
    raise ValueError(gid)
def fetch_rows():
    out=[]
    for y in YEARS:
        url=f'https://www.retrosheet.org/downloads/plays/{y}plays.zip'
        req=urllib.request.Request(url,headers={'User-Agent':'mlb-model-v6 half inning validation'})
        with urllib.request.urlopen(req,timeout=120) as resp:raw=resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for n in z.namelist():
                if not n.lower().endswith('.csv'):continue
                with z.open(n) as f:
                    for r in csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')):
                        if not is_regular(r.get('gametype')):continue
                        gid=str(r.get('gid') or r.get('game_id') or '').strip()
                        if not gid:continue
                        r['_year']=y;r['_gid']=gid;r['_day']=game_day(gid);out.append(r)
    out.sort(key=lambda r:(r['_day'],r['_gid']))
    return out

class Store:
    def __init__(self,h):self.h=float(h);self.d={}
    def get(self,k,day):
        v,last=self.d.get(k,(np.zeros(len(CLASSES)),day))
        if day>last:v=v*(0.5**((day-last)/self.h))
        return v.copy()
    def add(self,k,day,idx):
        v=self.get(k,day);v[idx]+=1;self.d[k]=(v,day)

def norm(v):
    s=float(np.sum(v));return np.asarray(v,dtype=float)/s if s>0 else np.full(len(CLASSES),1/len(CLASSES))
def softmax(z):
    z=np.asarray(z,dtype=float);z-=np.max(z);e=np.exp(z);return e/e.sum()
def model_prob(br,pr,lr,params):
    feat=[]
    for k in range(1,6):feat.append(math.log(max(EPS,br[k])/max(EPS,lr[k])))
    for k in range(1,6):feat.append(math.log(max(EPS,pr[k])/max(EPS,lr[k])))
    for k in range(1,6):feat.append(math.log(max(EPS,lr[k])/max(EPS,lr[0])))
    return softmax(np.asarray(params['coef'])@np.asarray(feat)+np.asarray(params['intercept']))
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
    params=json.loads((BASE/'joint_multinomial_pa_model.json').read_text())
    manifest=json.loads((BASE/'model_development_manifest.json').read_text())
    if manifest['governance_status']!='PA_GATE_PASS_HALF_INNING_PENDING':raise SystemExit(f"PA gate not passed: {manifest['governance_status']}")
    table=json.loads(TRANS.read_text());hp=params['selected_hyperparameters'];h=float(hp['half_life_days']);prior=float(hp['prior_strength'])
    rows=fetch_rows();bat=Store(h);pit=Store(h);league=Store(h);records=[];i=0
    while i<len(rows):
        day,gid=rows[i]['_day'],rows[i]['_gid'];j=i
        while j<len(rows) and rows[j]['_day']==day and rows[j]['_gid']==gid:j+=1
        game=rows[i:j];year=game[0]['_year'];lr=norm(league.get('league',day)+1.0)
        bcache={};pcache={};lineups={0:{},1:{}};starter={};all_pa=[];modeled=[]
        for r in game:
            if not truth(r.get('pa')):continue
            bid=str(r.get('batter') or '').strip();pid=str(r.get('pitcher') or '').strip();tb=intval(r.get('top_bot'),-1)
            if not bid or not pid or tb not in (0,1):continue
            if bid not in bcache:bcache[bid]=norm(bat.get(bid,day)+prior*lr)
            if pid not in pcache:pcache[pid]=norm(pit.get(pid,day)+prior*lr)
            lp=intval(r.get('lp'),0)
            if 1<=lp<=9 and lp not in lineups[tb]:lineups[tb][lp]=bid
            starter.setdefault(1-tb,pid)
            all_pa.append(r)
            ev=classify(r)
            if ev is not None:modeled.append((r,ev,bid,pid))
        if year==2025 and all(len(lineups[tb])==9 for tb in (0,1)):
            for tb in (0,1):
                order=[lineups[tb][k] for k in range(1,10)];sp=starter.get(1-tb)
                if not sp or sp not in pcache:continue
                i1pas=[r for r in all_pa if intval(r.get('inning'))==1 and intval(r.get('top_bot'))==tb]
                start2=len(i1pas)%9
                for inn,start in ((1,0),(2,start2)):
                    half=[r for r in all_pa if intval(r.get('inning'))==inn and intval(r.get('top_bot'))==tb]
                    if not half:continue
                    if any(str(r.get('pitcher') or '').strip()!=sp for r in half):continue
                    cand=[];leg=[];pr=pcache[sp]
                    for bid in order:
                        br=bcache.get(bid,norm(bat.get(bid,day)+prior*lr));cand.append(model_prob(br,pr,lr,params));leg.append(legacy_prob(br,pr))
                    pd1=run_dist(cand,start,table);pd0=run_dist(leg,start,table)
                    actual=sum(max(0,intval(r.get('runs'))) for r in game if intval(r.get('inning'))==inn and intval(r.get('top_bot'))==tb)
                    unmodeled=sum(1 for r in half if classify(r) is None)
                    records.append({'gid':gid,'inning':inn,'top_bot':tb,'actual_runs':actual,'unmodeled_pa':unmodeled,'joint_ll':ll(actual,pd1),'legacy_ll':ll(actual,pd0),'joint_brier':brier(actual,pd1),'legacy_brier':brier(actual,pd0),'joint_p_score':1-pd1[0],'legacy_p_score':1-pd0[0]})
        for r,ev,bid,pid in modeled:
            k=CLASSES.index(ev);bat.add(bid,day,k);pit.add(pid,day,k);league.add('league',day,k)
        i=j
    if not records:raise RuntimeError('No eligible 2025 I1/I2 half-innings')
    joint_ll=float(np.mean([r['joint_ll'] for r in records]));legacy_ll=float(np.mean([r['legacy_ll'] for r in records]));joint_br=float(np.mean([r['joint_brier'] for r in records]));legacy_br=float(np.mean([r['legacy_brier'] for r in records]))
    actual_score=np.asarray([r['actual_runs']>0 for r in records],dtype=float);jp=np.asarray([r['joint_p_score'] for r in records]);lp=np.asarray([r['legacy_p_score'] for r in records])
    sbj=float(np.mean((jp-actual_score)**2));sbl=float(np.mean((lp-actual_score)**2));unmodeled_rate=float(sum(r['unmodeled_pa'] for r in records)/max(1,sum(1 for _ in records)))
    result={'sample_half_innings':len(records),'scope':'2025 regular-season I1/I2, starting pitcher remains for all PAs','unmodeled_pa_per_half_inning':unmodeled_rate,'joint_multinomial':{'run_bucket_logloss':joint_ll,'run_bucket_brier':joint_br,'score_any_run_brier':sbj},'legacy_68_32':{'run_bucket_logloss':legacy_ll,'run_bucket_brier':legacy_br,'score_any_run_brier':sbl},'delta_vs_legacy':{'run_bucket_logloss':legacy_ll-joint_ll,'run_bucket_brier':legacy_br-joint_br,'score_any_run_brier':sbl-sbj}}
    pass_gate=(joint_ll<legacy_ll and joint_br<=legacy_br and sbj<=sbl);result['gate_status']='PASS' if pass_gate else 'BLOCKED'
    (BASE/'half_inning_validation_2025.json').write_text(json.dumps(result,indent=2));manifest['half_inning_validation_2025']=result;manifest['governance_status']='PASS' if pass_gate else 'BLOCKED_HALF_INNING_GATE';manifest['production_eligible']=bool(pass_gate);(BASE/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2));params['half_inning_validation_2025']=result;params['production_eligible']=bool(pass_gate);(BASE/'joint_multinomial_pa_model.json').write_text(json.dumps(params,indent=2));print(json.dumps(result,indent=2))
    if not pass_gate:raise SystemExit('HALF-INNING MODEL BLOCKED: joint PA model did not clear all locked-2025 scoring gates')
if __name__=='__main__':main()

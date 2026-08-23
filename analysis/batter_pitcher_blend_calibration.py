#!/usr/bin/env python3
"""Leakage-safe joint multinomial batter/pitcher PA calibration.

Architecture
------------
* Six mutually-exclusive PA outcomes are fitted jointly: out, BB, 1B, 2B, 3B, HR.
* Player skill is represented as batter/pitcher log-rate ratios versus the rolling league rate.
* Historical counts roll across seasons with exponential time decay.
* Player snapshots are frozen BEFORE each game; PAs from the current game are only added after
  all game predictions are materialized.
* 2021-2023 = development/training, 2024 = hyperparameter selection,
  2025 = locked chronological validation.
* No sportsbook or market inputs are used.

The locked-2025 PA gate compares the joint model against league-only, legacy 68/32,
batter-only, and pitcher-only probability vectors using multiclass log loss and Brier score.
Production promotion remains blocked until a separate half-inning scoring calibration gate passes.
"""
from __future__ import annotations
import csv, io, json, math, urllib.request, zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/batter_pitcher_blend'
YEARS=[2021,2022,2023,2024,2025]
CLASSES=['out','bb','single','double','triple','hr']
NONOUT=CLASSES[1:]
HALF_LIVES=[180.0,365.0,730.0]
PRIORS=[50.0,100.0,200.0]
CS=[0.1,0.3,1.0]
EPS=1e-7

def truth(v): return str(v).strip().lower() in {'1','true','t','yes','y'}
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
    return 'out'
def ids(r):
    bid=r.get('batter') or r.get('batter_id') or r.get('bat_id') or r.get('batterid')
    pid=r.get('pitcher') or r.get('pitcher_id') or r.get('pit_id') or r.get('pitcherid')
    return str(bid or '').strip(),str(pid or '').strip()
def game_id(r): return str(r.get('gid') or r.get('game_id') or r.get('gameid') or r.get('game') or '').strip()
def game_date(r):
    for k in ('date','game_date','game_dt'):
        v=str(r.get(k,'')).strip()
        if v:
            try: return datetime.fromisoformat(v[:10]).date().toordinal()
            except Exception: pass
    gid=game_id(r)
    digits=''.join(c for c in gid if c.isdigit())
    if len(digits)>=8:
        try: return datetime.strptime(digits[:8],'%Y%m%d').date().toordinal()
        except Exception: pass
    raise ValueError(f'Unable to determine game date for {gid!r}')

def fetch_plays():
    plays=[]; provenance=[]
    for y in YEARS:
        url=f'https://www.retrosheet.org/downloads/plays/{y}plays.zip'
        req=urllib.request.Request(url,headers={'User-Agent':'mlb-model-v6 joint empirical research'})
        with urllib.request.urlopen(req,timeout=120) as resp: raw=resp.read()
        rows=[]
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for n in z.namelist():
                if n.lower().endswith('.csv'):
                    with z.open(n) as f: rows.extend(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')))
        accepted=0
        for r in rows:
            if not is_regular(r.get('gametype')): continue
            ev=classify(r); bid,pid=ids(r); gid=game_id(r)
            if ev is None or not bid or not pid or not gid: continue
            plays.append((game_date(r),gid,y,bid,pid,ev)); accepted+=1
        provenance.append({'year':y,'source_url':url,'rows_total':len(rows),'accepted_pa':accepted})
    plays.sort(key=lambda x:(x[0],x[1]))
    return plays,provenance

class DecayedStore:
    def __init__(self,half_life): self.h=float(half_life); self.data={}
    def _decayed(self,key,day):
        vec,last=self.data.get(key,(np.zeros(len(CLASSES),dtype=float),day))
        if day>last: vec=vec*(0.5**((day-last)/self.h))
        return vec.copy()
    def get(self,key,day): return self._decayed(key,day)
    def add(self,key,day,class_idx):
        vec=self._decayed(key,day);vec[class_idx]+=1.0;self.data[key]=(vec,day)

def normalize(v):
    s=float(np.sum(v))
    if s<=0: return np.full(len(CLASSES),1/len(CLASSES))
    return np.asarray(v,dtype=float)/s

def build_features(plays,half_life,prior):
    bat=DecayedStore(half_life); pit=DecayedStore(half_life); league=DecayedStore(half_life)
    X=[]; y=[]; years=[]; probs=[]; meta=[]
    i=0
    while i<len(plays):
        day,gid=plays[i][0],plays[i][1]; j=i
        while j<len(plays) and plays[j][0]==day and plays[j][1]==gid: j+=1
        game=plays[i:j]
        lv=league.get('league',day); lrate=normalize(lv+1.0)
        bcache={};pcache={}
        for _,_,year,bid,pid,ev in game:
            if bid not in bcache:
                bv=bat.get(bid,day); bcache[bid]=normalize(bv+prior*lrate)
            if pid not in pcache:
                pv=pit.get(pid,day); pcache[pid]=normalize(pv+prior*lrate)
            br,pr=bcache[bid],pcache[pid]
            feat=[]
            for k in range(1,len(CLASSES)): feat.append(math.log(max(EPS,br[k])/max(EPS,lrate[k])))
            for k in range(1,len(CLASSES)): feat.append(math.log(max(EPS,pr[k])/max(EPS,lrate[k])))
            for k in range(1,len(CLASSES)): feat.append(math.log(max(EPS,lrate[k])/max(EPS,lrate[0])))
            X.append(feat); y.append(CLASSES.index(ev)); years.append(year)
            legacy=normalize(.68*br+.32*pr)
            probs.append({'league':lrate.copy(),'legacy':legacy,'batter':br.copy(),'pitcher':pr.copy()})
            meta.append({'year':year,'day':day,'game_id':gid,'batter_id':bid,'pitcher_id':pid})
        for _,_,year,bid,pid,ev in game:
            k=CLASSES.index(ev);bat.add(bid,day,k);pit.add(pid,day,k);league.add('league',day,k)
        i=j
    return np.asarray(X,dtype=np.float32),np.asarray(y,dtype=np.int8),np.asarray(years,dtype=np.int16),probs,meta

def multiclass_brier(y,p):
    one=np.eye(len(CLASSES),dtype=float)[y]
    return float(np.mean(np.sum((p-one)**2,axis=1)))
def metrics(y,p):
    return {'logloss':float(log_loss(y,p,labels=list(range(len(CLASSES))))),'brier':multiclass_brier(y,p)}
def baseline_matrix(probs,key): return np.vstack([x[key] for x in probs])
def fit_model(X,y,C):
    m=LogisticRegression(C=C,solver='lbfgs',max_iter=350)
    m.fit(X,y);return m

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plays,provenance=fetch_plays()
    if not plays: raise RuntimeError('No Retrosheet PA rows materialized')
    grid=[];best=None
    for hl in HALF_LIVES:
        for prior in PRIORS:
            X,y,yrs,pr,meta=build_features(plays,hl,prior)
            tr=np.isin(yrs,[2021,2022,2023]); sel=yrs==2024
            if not tr.any() or not sel.any(): raise RuntimeError(f'Empty chronological split for hl={hl} prior={prior}')
            for C in CS:
                model=fit_model(X[tr],y[tr],C); pp=model.predict_proba(X[sel]); m=metrics(y[sel],pp)
                row={'half_life_days':hl,'prior_strength':prior,'C':C,'logloss_2024':m['logloss'],'brier_2024':m['brier']};grid.append(row)
                if best is None or (m['logloss'],m['brier'])<(best[0],best[1]): best=(m['logloss'],m['brier'],hl,prior,C)
    _,_,hl,prior,C=best
    X,y,yrs,pr,meta=build_features(plays,hl,prior)
    train=np.isin(yrs,[2021,2022,2023,2024]); val=yrs==2025
    model=fit_model(X[train],y[train],C); fitted=model.predict_proba(X[val])
    val_probs=[p for p,z in zip(pr,val) if z]
    base={k:baseline_matrix(val_probs,k) for k in ('league','legacy','batter','pitcher')}
    validation={'joint_multinomial':metrics(y[val],fitted)}
    for k,p in base.items(): validation[k]=metrics(y[val],p)
    by_class={};one=np.eye(len(CLASSES))[y[val]]
    for idx,c in enumerate(CLASSES):
        by_class[c]={'mean_pred':float(np.mean(fitted[:,idx])),'observed_rate':float(np.mean(one[:,idx])),'brier':float(np.mean((fitted[:,idx]-one[:,idx])**2)),'legacy_brier':float(np.mean((base['legacy'][:,idx]-one[:,idx])**2))}
    pa_pass=(validation['joint_multinomial']['logloss']<validation['legacy']['logloss'] and validation['joint_multinomial']['brier']<=validation['legacy']['brier'])
    manifest={'component':'joint multinomial batter/pitcher PA model','architecture':'regularized multinomial logistic model on batter/pitcher relative log rates and league logits','governance_status':'PA_GATE_PASS_HALF_INNING_PENDING' if pa_pass else 'BLOCKED_PA_GATE','production_eligible':False,'development_years':[2021,2022,2023],'selection_year':[2024],'locked_validation_year':[2025],'market_inputs_used':False,'same_game_updates_used_in_features':False,'rolling_history_crosses_season_boundaries':True,'selected_hyperparameters':{'half_life_days':hl,'prior_strength':prior,'C':C},'validation_2025':validation,'validation_by_class':by_class,'provenance':provenance,'promotion_rule':'Must beat legacy 68/32 on locked-2025 multiclass log loss and not worsen multiclass Brier, then pass separate half-inning scoring calibration gate.'}
    params={'version':'joint-multinomial-pa-v1','classes':CLASSES,'feature_order':[f'batter_log_ratio_{e}' for e in NONOUT]+[f'pitcher_log_ratio_{e}' for e in NONOUT]+[f'league_logit_{e}_vs_out' for e in NONOUT],'selected_hyperparameters':manifest['selected_hyperparameters'],'coef':model.coef_.tolist(),'intercept':model.intercept_.tolist(),'sklearn_classes':model.classes_.tolist(),'validation_2025':validation,'production_eligible':False}
    (OUT/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2))
    (OUT/'joint_multinomial_pa_model.json').write_text(json.dumps(params,indent=2))
    with (OUT/'multinomial_grid_2024.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(grid[0]));w.writeheader();w.writerows(grid)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

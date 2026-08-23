#!/usr/bin/env python3
"""Leakage-safe joint multinomial batter/pitcher PA calibration.

Six modeled outcomes (out, BB, 1B, 2B, 3B, HR) are fitted jointly. HBP,
interference, reach-on-error and other no-out PAs are excluded because the current
production transition engine does not yet model those classes. Player skill rolls
across seasons with time decay and is frozen before each game.

Governance: 2021-23 train, 2024 select decay/prior/regularization, 2025 locked test.
"""
from __future__ import annotations
import csv, io, json, math, urllib.request, zipfile, gc
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/derived/model_calibration/batter_pitcher_blend'
YEARS=[2021,2022,2023,2024,2025]
CLASSES=['out','bb','single','double','triple','hr'];NONOUT=CLASSES[1:]
HALF_LIVES=[180.0,365.0,730.0];PRIORS=[50.0,100.0,200.0];CS=[0.1,0.3,1.0];EPS=1e-7

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
def ids(r):
    return str(r.get('batter') or r.get('batter_id') or '').strip(),str(r.get('pitcher') or r.get('pitcher_id') or '').strip()
def game_id(r):return str(r.get('gid') or r.get('game_id') or r.get('gameid') or r.get('game') or '').strip()
def game_date(r):
    for k in ('date','game_date','game_dt'):
        v=str(r.get(k,'')).strip()
        if v:
            try:return datetime.fromisoformat(v[:10]).date().toordinal()
            except:pass
    digits=''.join(c for c in game_id(r) if c.isdigit())
    if len(digits)>=8:return datetime.strptime(digits[:8],'%Y%m%d').date().toordinal()
    raise ValueError(game_id(r))
def fetch_plays():
    plays=[];prov=[]
    for y in YEARS:
        url=f'https://www.retrosheet.org/downloads/plays/{y}plays.zip';req=urllib.request.Request(url,headers={'User-Agent':'mlb-model-v6 joint empirical research'})
        with urllib.request.urlopen(req,timeout=120) as resp:raw=resp.read()
        rows=[]
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for n in z.namelist():
                if n.lower().endswith('.csv'):
                    with z.open(n) as f:rows.extend(csv.DictReader(io.TextIOWrapper(f,encoding='utf-8-sig',newline='')))
        acc=exc=0
        for r in rows:
            if not is_regular(r.get('gametype')):continue
            ev=classify(r);bid,pid=ids(r);gid=game_id(r)
            if truth(r.get('pa')) and ev is None:exc+=1
            if ev is None or not bid or not pid or not gid:continue
            plays.append((game_date(r),gid,y,bid,pid,ev));acc+=1
        prov.append({'year':y,'source_url':url,'rows_total':len(rows),'accepted_modeled_pa':acc,'excluded_unmodeled_pa':exc})
    plays.sort(key=lambda x:(x[0],x[1]));return plays,prov

class Store:
    def __init__(self,h):self.h=float(h);self.d={}
    def get(self,k,day):
        v,last=self.d.get(k,(np.zeros(6),day));return (v*(0.5**((day-last)/self.h)) if day>last else v).copy()
    def add(self,k,day,idx):v=self.get(k,day);v[idx]+=1;self.d[k]=(v,day)
def norm(v):
    s=float(np.sum(v));return np.asarray(v,dtype=float)/s if s>0 else np.full(6,1/6)

def build_features(plays,h,prior,collect_baselines_year=None):
    bat=Store(h);pit=Store(h);league=Store(h);X=[];Y=[];yrs=[];bases=[];i=0
    while i<len(plays):
        day,gid=plays[i][0],plays[i][1];j=i
        while j<len(plays) and plays[j][0]==day and plays[j][1]==gid:j+=1
        game=plays[i:j];lr=norm(league.get('league',day)+1.0);bc={};pc={}
        for _,_,year,bid,pid,ev in game:
            if bid not in bc:bc[bid]=norm(bat.get(bid,day)+prior*lr)
            if pid not in pc:pc[pid]=norm(pit.get(pid,day)+prior*lr)
            br,pr=bc[bid],pc[pid];feat=[]
            feat.extend(math.log(max(EPS,br[k])/max(EPS,lr[k])) for k in range(1,6));feat.extend(math.log(max(EPS,pr[k])/max(EPS,lr[k])) for k in range(1,6));feat.extend(math.log(max(EPS,lr[k])/max(EPS,lr[0])) for k in range(1,6))
            X.append(feat);Y.append(CLASSES.index(ev));yrs.append(year)
            if collect_baselines_year==year:bases.append(np.stack([lr,norm(.68*br+.32*pr),br,pr]).astype(np.float32))
        for _,_,_,bid,pid,ev in game:
            k=CLASSES.index(ev);bat.add(bid,day,k);pit.add(pid,day,k);league.add('league',day,k)
        i=j
    return np.asarray(X,dtype=np.float32),np.asarray(Y,dtype=np.int8),np.asarray(yrs,dtype=np.int16),(np.stack(bases) if bases else None)

def brier(y,p):return float(np.mean(np.sum((p-np.eye(6)[y])**2,axis=1)))
def metrics(y,p):return {'logloss':float(log_loss(y,p,labels=list(range(6)))),'brier':brier(y,p)}
def fit(X,y,C):
    m=LogisticRegression(C=C,solver='lbfgs',max_iter=350);m.fit(X,y);return m

def main():
    OUT.mkdir(parents=True,exist_ok=True);plays,prov=fetch_plays()
    if not plays:raise RuntimeError('No Retrosheet modeled PAs materialized')
    grid=[];best=None
    for h in HALF_LIVES:
        for prior in PRIORS:
            X,y,yrs,_=build_features(plays,h,prior);tr=np.isin(yrs,[2021,2022,2023]);sel=yrs==2024
            if not tr.any() or not sel.any():raise RuntimeError('Empty chronological split')
            for C in CS:
                m=fit(X[tr],y[tr],C);met=metrics(y[sel],m.predict_proba(X[sel]));row={'half_life_days':h,'prior_strength':prior,'C':C,'logloss_2024':met['logloss'],'brier_2024':met['brier']};grid.append(row)
                if best is None or (met['logloss'],met['brier'])<(best[0],best[1]):best=(met['logloss'],met['brier'],h,prior,C)
            del X,y,yrs;gc.collect()
    _,_,h,prior,C=best
    X,y,yrs,bases=build_features(plays,h,prior,collect_baselines_year=2025);train=np.isin(yrs,[2021,2022,2023,2024]);val=yrs==2025;m=fit(X[train],y[train],C);pred=m.predict_proba(X[val])
    if bases is None or len(bases)!=int(val.sum()):raise RuntimeError('Locked-2025 baseline alignment failure')
    names=['league','legacy','batter','pitcher'];validation={'joint_multinomial':metrics(y[val],pred)}
    for idx,n in enumerate(names):validation[n]=metrics(y[val],bases[:,idx,:])
    one=np.eye(6)[y[val]];by_class={c:{'mean_pred':float(pred[:,i].mean()),'observed_rate':float(one[:,i].mean()),'brier':float(np.mean((pred[:,i]-one[:,i])**2)),'legacy_brier':float(np.mean((bases[:,1,i]-one[:,i])**2))} for i,c in enumerate(CLASSES)}
    pa_pass=validation['joint_multinomial']['logloss']<validation['legacy']['logloss'] and validation['joint_multinomial']['brier']<=validation['legacy']['brier']
    manifest={'component':'joint multinomial batter/pitcher PA model','architecture':'regularized multinomial logistic model on batter/pitcher relative log rates and league logits','modeled_outcomes':CLASSES,'unmodeled_pa_handling':'HBP/interference/reach-on-error/no-out PAs excluded from PA fit; materiality tested by half-inning gate','governance_status':'PA_GATE_PASS_HALF_INNING_PENDING' if pa_pass else 'BLOCKED_PA_GATE','production_eligible':False,'development_years':[2021,2022,2023],'selection_year':[2024],'locked_validation_year':[2025],'market_inputs_used':False,'same_game_updates_used_in_features':False,'rolling_history_crosses_season_boundaries':True,'selected_hyperparameters':{'half_life_days':h,'prior_strength':prior,'C':C},'validation_2025':validation,'validation_by_class':by_class,'provenance':prov,'promotion_rule':'Beat legacy 68/32 on locked-2025 multiclass log loss without worsening multiclass Brier, then pass locked-2025 half-inning scoring gate.'}
    params={'version':'joint-multinomial-pa-v1','classes':CLASSES,'feature_order':[f'batter_log_ratio_{e}' for e in NONOUT]+[f'pitcher_log_ratio_{e}' for e in NONOUT]+[f'league_logit_{e}_vs_out' for e in NONOUT],'selected_hyperparameters':manifest['selected_hyperparameters'],'coef':m.coef_.tolist(),'intercept':m.intercept_.tolist(),'sklearn_classes':m.classes_.tolist(),'validation_2025':validation,'production_eligible':False}
    (OUT/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2));(OUT/'joint_multinomial_pa_model.json').write_text(json.dumps(params,indent=2))
    with (OUT/'multinomial_grid_2024.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(grid[0]));w.writeheader();w.writerows(grid)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Development-only conditional reliever selection model.

Consumes exact-date active-roster candidate sets. Fits a regularized candidate-level
utility model, then normalizes utilities within each pitching-change choice set.
No team identity, pitcher identity, market data, or future-game information is used
as a predictive feature. September 2025 is diagnostic only; production promotion
requires a fresh 2026 forward test.
"""
from __future__ import annotations
import csv,json,math,time
from pathlib import Path
from datetime import date
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions'
CAND=BASE/'bullpen_reliever_candidate_sets_2025.csv'
OUT=BASE/'bullpen_reliever_selection_development_2025.json'
EPS=1e-12
NUM=['prior_apps_30d','prior_bf_30d','prior_rest_days']
CAT=['transition_kind','inning_band']

def parse_date(s): return date.fromisoformat(s)
def inning_band(i):
    i=int(i)
    if i<=4:return '1-4'
    if i<=6:return '5-6'
    if i<=8:return '7-8'
    return '9+'

def load():
    groups=[]
    with CAND.open() as f:
        for r in csv.DictReader(f):
            if int(r['actual_next_in_candidate_pool'])!=1: continue
            cs=json.loads(r['candidates_json'])
            if len(cs)<2: continue
            g={'date':parse_date(r['date']),'actual':r['actual_next_pitcher_id'],'rows':[]}
            for c in cs:
                g['rows'].append({
                    'pitcher_id':c['pitcher_id'],
                    'prior_apps_30d':float(c.get('prior_apps_30d') or 0),
                    'prior_bf_30d':float(c.get('prior_bf_30d') or 0),
                    'prior_rest_days':None if c.get('prior_rest_days') in (None,'') else float(c['prior_rest_days']),
                    'transition_kind':r['transition_kind'],
                    'inning_band':inning_band(r['inning']),
                    'y':int(c['pitcher_id']==r['actual_next_pitcher_id'])})
            if sum(x['y'] for x in g['rows'])==1:groups.append(g)
    return groups

def split(groups):
    tr=[];va=[];te=[]
    for g in groups:
        if g['date']<=date(2025,7,31):tr.append(g)
        elif g['date']<=date(2025,8,31):va.append(g)
        else:te.append(g)
    return tr,va,te

def flatten(gs):
    X=[];y=[]
    for g in gs:
        for r in g['rows']:
            X.append([r[k] for k in NUM+CAT]);y.append(r['y'])
    return np.asarray(X,dtype=object),np.asarray(y,dtype=int)

def make_model(C):
    pre=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler())]),list(range(len(NUM)))),
        ('cat',OneHotEncoder(handle_unknown='ignore'),list(range(len(NUM),len(NUM)+len(CAT))))])
    return Pipeline([('pre',pre),('lr',LogisticRegression(C=C,max_iter=2000,class_weight='balanced'))])

def probs_for_group(model,g):
    X=np.asarray([[r[k] for k in NUM+CAT] for r in g['rows']],dtype=object)
    p=model.predict_proba(X)[:,1]
    u=np.log(np.clip(p,EPS,1-EPS))-np.log(np.clip(1-p,EPS,1-EPS));u-=u.max();e=np.exp(u);return e/e.sum()

def evaluate(model,gs):
    n=len(gs); ll=0.;top1=0;top3=0;mrr=0.;base_u=0.;base_usage=0.;usage_ll=0.;lat=[]
    for g in gs:
        t=time.perf_counter_ns();p=probs_for_group(model,g);lat.append((time.perf_counter_ns()-t)/1e6)
        yi=next(i for i,r in enumerate(g['rows']) if r['y']==1); ll-=math.log(max(EPS,p[yi]))
        order=np.argsort(-p);rank=int(np.where(order==yi)[0][0])+1;top1+=rank==1;top3+=rank<=3;mrr+=1/rank
        k=len(g['rows']);base_u+=math.log(k)
        usage=np.asarray([max(0.,r['prior_apps_30d'])+0.25 for r in g['rows']],float);usage/=usage.sum();usage_ll-=math.log(max(EPS,usage[yi]));base_usage+=int(np.argmax(usage)==yi)
    return {'choice_sets':n,'log_loss':ll/n,'top1_accuracy':top1/n,'top3_accuracy':top3/n,'mrr':mrr/n,
            'uniform_log_loss':base_u/n,'prior_usage_log_loss':usage_ll/n,'prior_usage_top1_accuracy':base_usage/n,
            'p99_inference_ms':float(np.percentile(lat,99)) if lat else None}

def main():
    groups=load();tr,va,te=split(groups)
    if min(map(len,(tr,va,te)))<100: raise SystemExit(f'insufficient split sizes: {[len(tr),len(va),len(te)]}')
    best=None
    for C in (0.03,0.1,0.3,1.0):
        X,y=flatten(tr);m=make_model(C);m.fit(X,y);ev=evaluate(m,va)
        if best is None or ev['log_loss']<best[0]:best=(ev['log_loss'],C,m,ev)
    _,C,m,val=best
    test=evaluate(m,te)
    rep={'status':'DEVELOPMENT_ONLY_2025','market_blind':True,
         'architecture':'candidate-level regularized utility model normalized within exact-date active-roster choice set',
         'features':NUM+CAT,'excluded_features':['team identity','pitcher identity','sportsbook/market inputs','future current-game usage'],
         'split':{'train':'through 2025-07-31','validation':'2025-08-01..2025-08-31','diagnostic_only':'2025-09-01 onward'},
         'sample_choice_sets':{'train':len(tr),'validation':len(va),'diagnostic':len(te)},'selected_C':C,
         'validation':val,'september_diagnostic':test,
         'development_gate':{'beats_uniform_logloss':test['log_loss']<test['uniform_log_loss'],
                             'beats_prior_usage_logloss':test['log_loss']<test['prior_usage_log_loss'],
                             'beats_prior_usage_top1':test['top1_accuracy']>test['prior_usage_top1_accuracy'],
                             'p99_lt_1000ms':test['p99_inference_ms']<1000},
         'promotion_rule':'No production promotion from 2025 diagnostics. Final promotion requires fresh 2026 forward test plus downstream run-distribution improvement.'}
    OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
    if not (rep['development_gate']['beats_uniform_logloss'] and rep['development_gate']['beats_prior_usage_logloss'] and rep['development_gate']['p99_lt_1000ms']):
        raise SystemExit('Reliever-selection development gate blocked')
if __name__=='__main__':main()
# CI sync marker: reliever-choice fit wired into bullpen validation.

#!/usr/bin/env python3
"""Development-only conditional reliever selection model with controlled feature expansion."""
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
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions'; CAND=BASE/'bullpen_reliever_candidate_sets_2025.csv'; OUT=BASE/'bullpen_reliever_selection_development_2025.json'; EPS=1e-12
BASE_NUM=['prior_apps_30d','prior_bf_30d','prior_rest_days']; BASE_CAT=['transition_kind','inning_band']
EXP_NUM=BASE_NUM+['prior_late_inning_share_30d','prior_save_like_share_30d','already_used_this_game','defensive_score_diff']
EXP_CAT=BASE_CAT+['throws','score_band']
def parse_date(s):return date.fromisoformat(s)
def inning_band(i):
 i=int(i); return '1-4' if i<=4 else '5-6' if i<=6 else '7-8' if i<=8 else '9+'
def score_band(x):
 x=float(x); return '<=-4' if x<=-4 else '-3..-1' if x<0 else '0' if x==0 else '1..3' if x<=3 else '4+'
def load():
 groups=[]
 with CAND.open() as f:
  for r in csv.DictReader(f):
   if int(r['actual_next_in_candidate_pool'])!=1:continue
   cs=json.loads(r['candidates_json']);
   if len(cs)<2:continue
   sd=float(r['defensive_score_diff']) if r.get('defensive_score_diff') not in (None,'') else 0.0
   g={'date':parse_date(r['date']),'rows':[]}
   for c in cs:
    g['rows'].append({'pitcher_id':c['pitcher_id'],'prior_apps_30d':float(c.get('prior_apps_30d') or 0),'prior_bf_30d':float(c.get('prior_bf_30d') or 0),'prior_rest_days':None if c.get('prior_rest_days') in (None,'') else float(c['prior_rest_days']),'prior_late_inning_share_30d':float(c.get('prior_late_inning_share_30d') or 0),'prior_save_like_share_30d':float(c.get('prior_save_like_share_30d') or 0),'already_used_this_game':float(c.get('already_used_this_game') or 0),'defensive_score_diff':sd,'transition_kind':r['transition_kind'],'inning_band':inning_band(r.get('decision_inning') or r['inning']),'throws':c.get('throws') or 'UNK','score_band':score_band(sd),'y':int(c['pitcher_id']==r['actual_next_pitcher_id'])})
   if sum(x['y'] for x in g['rows'])==1:groups.append(g)
 return groups
def split(gs):
 tr=[];va=[];te=[]
 for g in gs:(tr if g['date']<=date(2025,7,31) else va if g['date']<=date(2025,8,31) else te).append(g)
 return tr,va,te
def make_model(C,num,cat):
 pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler())]),list(range(len(num)))),('cat',OneHotEncoder(handle_unknown='ignore'),list(range(len(num),len(num)+len(cat))))])
 return Pipeline([('pre',pre),('lr',LogisticRegression(C=C,max_iter=2000,class_weight='balanced'))])
def flat(gs,num,cat):
 X=[];y=[]
 for g in gs:
  for r in g['rows']:X.append([r[k] for k in num+cat]);y.append(r['y'])
 return np.asarray(X,dtype=object),np.asarray(y,dtype=int)
def probs(m,g,num,cat):
 X=np.asarray([[r[k] for k in num+cat] for r in g['rows']],dtype=object);p=m.predict_proba(X)[:,1];u=np.log(np.clip(p,EPS,1-EPS))-np.log(np.clip(1-p,EPS,1-EPS));u-=u.max();e=np.exp(u);return e/e.sum()
def evalm(m,gs,num,cat):
 ll=top1=top3=mrr=ul=ull=ut=0.;lat=[]
 for g in gs:
  t=time.perf_counter_ns();p=probs(m,g,num,cat);lat.append((time.perf_counter_ns()-t)/1e6);yi=next(i for i,r in enumerate(g['rows']) if r['y']);ll-=math.log(max(EPS,p[yi]));o=np.argsort(-p);rank=int(np.where(o==yi)[0][0])+1;top1+=rank==1;top3+=rank<=3;mrr+=1/rank;k=len(g['rows']);ul+=math.log(k);u=np.asarray([max(0.,r['prior_apps_30d'])+.25 for r in g['rows']],float);u/=u.sum();ull-=math.log(max(EPS,u[yi]));ut+=np.argmax(u)==yi
 n=len(gs);return {'choice_sets':n,'log_loss':ll/n,'top1_accuracy':top1/n,'top3_accuracy':top3/n,'mrr':mrr/n,'uniform_log_loss':ul/n,'prior_usage_log_loss':ull/n,'prior_usage_top1_accuracy':ut/n,'p99_inference_ms':float(np.percentile(lat,99))}
def fit_spec(tr,va,te,num,cat):
 X,y=flat(tr,num,cat);best=None
 for C in (.03,.1,.3,1.0):
  m=make_model(C,num,cat);m.fit(X,y);v=evalm(m,va,num,cat)
  if best is None or v['log_loss']<best[0]:best=(v['log_loss'],C,m,v)
 _,C,m,v=best;return {'C':C,'validation':v,'september_diagnostic':evalm(m,te,num,cat)}
def main():
 gs=load();tr,va,te=split(gs)
 if min(map(len,(tr,va,te)))<100:raise SystemExit('insufficient split')
 b=fit_spec(tr,va,te,BASE_NUM,BASE_CAT);e=fit_spec(tr,va,te,EXP_NUM,EXP_CAT);bd=b['september_diagnostic'];ed=e['september_diagnostic']
 rep={'status':'DEVELOPMENT_ONLY_2025','market_blind':True,'sample_choice_sets':{'train':len(tr),'validation':len(va),'diagnostic':len(te)},'base_spec':b,'expanded_spec':e,'expanded_features':EXP_NUM+EXP_CAT,'excluded_features':['team identity','pitcher identity','sportsbook/market inputs','future current-game usage'],'development_gate':{'expanded_beats_base_logloss':ed['log_loss']<bd['log_loss'],'expanded_beats_uniform':ed['log_loss']<ed['uniform_log_loss'],'expanded_beats_prior_usage':ed['log_loss']<ed['prior_usage_log_loss'],'p99_lt_1000ms':ed['p99_inference_ms']<1000},'promotion_rule':'No production promotion from 2025 diagnostics. Require fresh 2026 forward test and downstream run-distribution improvement.'}
 OUT.write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2))
 if not (rep['development_gate']['expanded_beats_uniform'] and rep['development_gate']['expanded_beats_prior_usage'] and rep['development_gate']['p99_lt_1000ms']):raise SystemExit('Expanded reliever-selection development gate blocked')
if __name__=='__main__':main()

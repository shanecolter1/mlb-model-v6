#!/usr/bin/env python3
"""Fit market-blind between-inning pitcher-removal hazard.

2025 September has already been inspected for this component, so it is now development
evidence rather than a pristine final holdout. This specification improves structural
representation (categorical inning/BF state and starter×inning interaction) without
loosening calibration or latency standards. Final production promotion requires a
fresh 2026 forward test after this specification is frozen.
"""
from __future__ import annotations
import csv,json,pickle,time
from datetime import date
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions'
CSV_PATH=BASE/'bullpen_between_inning_boundaries_2025.csv'
OUT=BASE/'between_inning_hazard'
NUM=['batters_in_completed_half_inning','defensive_score_diff']
CAT=['completed_inning_cat','bf_band_cat','top_bot_defense','next_batter_lineup_position','pitcher_is_starter','starter_inning']
CGRID=[0.01,0.03,0.1,0.3,1.0]
MIN_CONTEXT_N=200

def parse_date(s):
 raw=''.join(ch for ch in str(s) if ch.isdigit())
 if len(raw)<=6:
  d=date.fromordinal(int(raw));return d.year*10000+d.month*100+d.day
 return int(raw[:8])

def bf_band(v):
 x=float(v); return '0-9' if x<10 else '10-17' if x<18 else '18-24' if x<25 else '25+'

def load():
 out=[]
 with CSV_PATH.open() as f:
  for r in csv.DictReader(f):
   try:
    d=parse_date(r['date']); y=int(r['pitcher_changed_before_next_defensive_inning'])
    inn=str(int(float(r['completed_inning']))); st=str(r.get('pitcher_is_starter','0'))
    vals={
      'batters_in_completed_half_inning':float(r['batters_in_completed_half_inning']),
      'defensive_score_diff':float(r['defensive_score_diff']),
      'completed_inning_cat':inn,
      'bf_band_cat':bf_band(r['pitcher_batters_faced_game_to_date']),
      'top_bot_defense':str(r.get('top_bot_defense','')),
      'next_batter_lineup_position':str(r.get('next_batter_lineup_position','')),
      'pitcher_is_starter':st,
      'starter_inning':f'{st}:{inn}',
      '_raw_bf':float(r['pitcher_batters_faced_game_to_date']),
    }
    out.append((d,vals,y))
   except Exception: pass
 return out

def mat(rows):
 cols=NUM+CAT; a=np.empty((len(rows),len(cols)),dtype=object); y=np.array([z[2] for z in rows],int)
 for i,(_,x,_) in enumerate(rows):
  for j,c in enumerate(cols): a[i,j]=x[c]
 return a,y

def pipe(C):
 pre=ColumnTransformer([
  ('n',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler())]),list(range(len(NUM)))),
  ('c',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),list(range(len(NUM),len(NUM)+len(CAT))))
 ])
 return Pipeline([('pre',pre),('lr',LogisticRegression(C=C,max_iter=1000))])

def met(y,p):
 p=np.clip(np.asarray(p),1e-15,1-1e-15)
 return {'n':int(len(y)),'positive_rate':float(np.mean(y)),'predicted_rate':float(np.mean(p)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None}

def score_band(x):
 x=float(x);return '<=-4' if x<=-4 else '-3:-2' if x<=-2 else '-1:1' if x<=1 else '2:3' if x<=3 else '>=4'

def cal(rows,p):
 specs=[('inning','completed_inning_cat'),('starter','pitcher_is_starter'),('score_band','defensive_score_diff'),('bf_band','bf_band_cat')]
 out={}; worst=0.0; worst_supported=0.0
 for label,key in specs:
  g={}
  for i,(_,x,y) in enumerate(rows):
   try:k=score_band(x[key]) if label=='score_band' else str(x[key])
   except:continue
   g.setdefault(k,[]).append(i)
  out[label]={}
  for k,idx in g.items():
   yy=np.array([rows[i][2] for i in idx]);pp=np.array([p[i] for i in idx]);gap=abs(float(np.mean(yy)-np.mean(pp)))
   worst=max(worst,gap)
   supported=len(idx)>=MIN_CONTEXT_N
   if supported: worst_supported=max(worst_supported,gap)
   out[label][k]={'n':len(idx),'observed_rate':float(np.mean(yy)),'predicted_rate':float(np.mean(pp)),'abs_gap':gap,'governance_supported':supported}
 return out,worst,worst_supported

def main():
 rows=load();tr=[r for r in rows if r[0]<=20250731];va=[r for r in rows if 20250801<=r[0]<=20250831];diag=[r for r in rows if r[0]>=20250901]
 if min(map(len,(tr,va,diag)))<500:raise RuntimeError('temporal split too small')
 Xt,yt=mat(tr);Xv,yv=mat(va);Xd,yd=mat(diag);tune=[]
 for C in CGRID:
  m=pipe(C);m.fit(Xt,yt);mm=met(yv,m.predict_proba(Xv)[:,1]);mm['C']=C;tune.append(mm)
 best=min(tune,key=lambda z:(z['log_loss'],z['brier']))
 dev=tr+va;Xdev,ydev=mat(dev);m=pipe(best['C']);m.fit(Xdev,ydev);pd=m.predict_proba(Xd)[:,1]
 dm=met(yd,pd);base=np.full_like(yd,float(np.mean(ydev)),dtype=float);bm=met(yd,base);contexts,worst,worst_supported=cal(diag,pd)
 lat=[]
 for i in range(min(5000,len(Xd))):
  t=time.perf_counter_ns();m.predict_proba(Xd[i:i+1]);lat.append((time.perf_counter_ns()-t)/1e6)
 latency={'n':len(lat),'median_ms':float(np.median(lat)),'p95_ms':float(np.percentile(lat,95)),'p99_ms':float(np.percentile(lat,99)),'max_ms':float(np.max(lat))}
 gates={'beats_constant_brier':dm['brier']<bm['brier'],'beats_constant_log_loss':dm['log_loss']<bm['log_loss'],'supported_context_abs_gap_le_0_05':worst_supported<=0.05,'latency_p99_lt_1000ms':latency['p99_ms']<1000}
 dev_status='PASS' if all(gates.values()) else 'BLOCKED'
 rep={'market_blind':True,'task':'P(pitcher change before next defensive inning)','feature_columns':NUM+CAT,'anti_overfit_note':'September 2025 has been inspected and is now development diagnostic evidence, not pristine final holdout. Final production promotion requires frozen-spec 2026 forward evaluation.','split':{'train_n':len(tr),'validation_n':len(va),'september_2025_diagnostic_n':len(diag)},'regularization_tuning':tune,'selected_C':best['C'],'september_2025_diagnostic_model':dm,'diagnostic_constant_rate_baseline':bm,'context_calibration':contexts,'worst_context_abs_gap_all_groups':worst,'worst_context_abs_gap_supported_groups':worst_supported,'minimum_context_n_for_gate':MIN_CONTEXT_N,'latency':latency,'development_gates':gates,'development_status':dev_status,'production_promotion_status':'FORWARD_TEST_REQUIRED' if dev_status=='PASS' else 'BLOCKED','required_next_test':'fresh 2026 forward/out-of-time evaluation with frozen specification'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'between_inning_removal_hazard_validation.json').write_text(json.dumps(rep,indent=2));pickle.dump(m,(OUT/'between_inning_removal_hazard_model.pkl').open('wb'));print(json.dumps(rep,indent=2))
 if dev_status!='PASS':raise SystemExit('Between-inning development gate blocked')
if __name__=='__main__':main()

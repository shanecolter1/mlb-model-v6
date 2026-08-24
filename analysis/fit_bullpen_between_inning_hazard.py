#!/usr/bin/env python3
"""Fit market-blind between-inning pitcher-removal hazard with locked temporal validation."""
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
NUM=['completed_inning','pitcher_batters_faced_game_to_date','batters_in_completed_half_inning','defensive_score_diff']
CAT=['top_bot_defense','next_batter_lineup_position','pitcher_is_starter']
CGRID=[0.01,0.03,0.1,0.3,1.0]
def parse_date(s):
 raw=''.join(ch for ch in str(s) if ch.isdigit())
 if len(raw)<=6:
  d=date.fromordinal(int(raw));return d.year*10000+d.month*100+d.day
 return int(raw[:8])
def load():
 out=[]
 with CSV_PATH.open() as f:
  for r in csv.DictReader(f):
   try:
    d=parse_date(r['date']); y=int(r['pitcher_changed_before_next_defensive_inning'])
    vals={k:r.get(k,'') for k in NUM+CAT}
    for k in NUM: vals[k]=float(vals[k]) if vals[k] not in ('',None,'None') else np.nan
    for k in CAT: vals[k]=str(vals[k])
    out.append((d,vals,y))
   except Exception: pass
 return out
def mat(rows):
 cols=NUM+CAT; a=np.empty((len(rows),len(cols)),dtype=object); y=np.array([z[2] for z in rows],int)
 for i,(_,x,_) in enumerate(rows):
  for j,c in enumerate(cols): a[i,j]=x[c]
 return a,y
def pipe(C):
 pre=ColumnTransformer([('n',Pipeline([('imp',SimpleImputer(strategy='median',add_indicator=True)),('sc',StandardScaler())]),list(range(len(NUM)))),('c',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),list(range(len(NUM),len(NUM)+len(CAT))))])
 return Pipeline([('pre',pre),('lr',LogisticRegression(C=C,max_iter=1000))])
def met(y,p):
 p=np.clip(np.asarray(p),1e-15,1-1e-15)
 return {'n':int(len(y)),'positive_rate':float(np.mean(y)),'predicted_rate':float(np.mean(p)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None}
def band(v,kind):
 if kind=='inning': return str(int(v))
 if kind=='starter': return str(v)
 if kind=='score':
  x=float(v);return '<=-4' if x<=-4 else '-3:-2' if x<=-2 else '-1:1' if x<=1 else '2:3' if x<=3 else '>=4'
 x=float(v);return '0-9' if x<10 else '10-17' if x<18 else '18-24' if x<25 else '25+'
def cal(rows,p):
 specs=[('inning','completed_inning'),('starter','pitcher_is_starter'),('score_band','defensive_score_diff'),('bf_band','pitcher_batters_faced_game_to_date')]; out={}; worst=0.0
 for label,key in specs:
  g={}
  for i,(_,x,y) in enumerate(rows):
   try:k=band(x[key],'score' if label=='score_band' else 'starter' if label=='starter' else 'inning' if label=='inning' else 'bf')
   except:continue
   g.setdefault(k,[]).append(i)
  out[label]={}
  for k,idx in g.items():
   yy=np.array([rows[i][2] for i in idx]); pp=np.array([p[i] for i in idx]); gap=abs(float(np.mean(yy)-np.mean(pp)));worst=max(worst,gap)
   out[label][k]={'n':len(idx),'observed_rate':float(np.mean(yy)),'predicted_rate':float(np.mean(pp)),'abs_gap':gap}
 return out,worst
def main():
 rows=load(); tr=[r for r in rows if r[0]<=20250731]; va=[r for r in rows if 20250801<=r[0]<=20250831]; te=[r for r in rows if r[0]>=20250901]
 if min(map(len,(tr,va,te)))<500: raise RuntimeError('temporal split too small')
 Xt,yt=mat(tr);Xv,yv=mat(va);Xq,yq=mat(te); tune=[]
 for C in CGRID:
  m=pipe(C);m.fit(Xt,yt);mm=met(yv,m.predict_proba(Xv)[:,1]);mm['C']=C;tune.append(mm)
 best=min(tune,key=lambda z:(z['log_loss'],z['brier'])); dev=tr+va;Xd,yd=mat(dev);m=pipe(best['C']);m.fit(Xd,yd);pq=m.predict_proba(Xq)[:,1]
 tm=met(yq,pq); base=np.full_like(yq,float(np.mean(yd)),dtype=float); bm=met(yq,base); contexts,worst=cal(te,pq)
 lat=[]
 for i in range(min(5000,len(Xq))):
  t=time.perf_counter_ns();m.predict_proba(Xq[i:i+1]);lat.append((time.perf_counter_ns()-t)/1e6)
 latency={'n':len(lat),'median_ms':float(np.median(lat)),'p95_ms':float(np.percentile(lat,95)),'p99_ms':float(np.percentile(lat,99)),'max_ms':float(np.max(lat))}
 gates={'beats_constant_brier':tm['brier']<bm['brier'],'beats_constant_log_loss':tm['log_loss']<bm['log_loss'],'worst_context_abs_gap_le_0_05':worst<=0.05,'latency_p99_lt_1000ms':latency['p99_ms']<1000}
 rep={'market_blind':True,'task':'P(pitcher change before next defensive inning)','feature_columns':NUM+CAT,'split':{'train_n':len(tr),'validation_n':len(va),'locked_test_n':len(te)},'regularization_tuning':tune,'selected_C':best['C'],'locked_test_model':tm,'locked_test_constant_rate_baseline':bm,'context_calibration':contexts,'worst_context_abs_gap':worst,'latency':latency,'promotion_gates':gates,'promotion_status':'PASS' if all(gates.values()) else 'BLOCKED'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'between_inning_removal_hazard_validation.json').write_text(json.dumps(rep,indent=2));pickle.dump(m,(OUT/'between_inning_removal_hazard_model.pkl').open('wb'));print(json.dumps(rep,indent=2))
 if rep['promotion_status']!='PASS': raise SystemExit('Between-inning hazard gate blocked')
if __name__=='__main__':main()

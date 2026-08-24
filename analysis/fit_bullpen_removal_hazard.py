#!/usr/bin/env python3
"""Fit and validate the market-blind in-inning pitcher-removal hazard."""
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

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/derived/model_calibration/bullpen_transitions';CSV_PATH=BASE/'bullpen_transition_pa_boundaries_2025.csv';OUT=BASE/'removal_hazard'
NUM=['inning','pitcher_batters_faced_game_to_date','defensive_score_diff'];CAT=['top_bot','outs_pre','bases_mask','lineup_position','pitcher_is_starter'];CANDIDATE_C=[0.01,0.03,0.1,0.3,1.0]

def parse_date(s):
    raw=str(s).strip();digits=''.join(ch for ch in raw if ch.isdigit())
    if digits and len(digits)<=6:
        d=date.fromordinal(int(digits));return d.year*10000+d.month*100+d.day
    if len(digits)>=8:return int(digits[:8])
    raise ValueError(f'unrecognized date value: {s!r}')

def load_rows():
    rows=[]
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            try:
                d=parse_date(r['date']);y=int(r['pitcher_changed_before_next_pa']);vals={k:r.get(k,'') for k in NUM+CAT}
                for k in NUM:vals[k]=float(vals[k]) if vals[k] not in ('',None,'None') else np.nan
                for k in CAT:vals[k]=str(vals[k])
                rows.append((d,vals,y))
            except Exception:continue
    return rows

def matrix(rows):
    X=[x for _,x,_ in rows];y=np.array([y for *_,y in rows],dtype=int);cols=NUM+CAT;arr=np.empty((len(X),len(cols)),dtype=object)
    for i,x in enumerate(X):
        for j,c in enumerate(cols):arr[i,j]=x[c]
    return arr,y,cols

def make_pipe(C):
    ni=list(range(len(NUM)));ci=list(range(len(NUM),len(NUM)+len(CAT)))
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),ni),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),ci)])
    return Pipeline([('pre',pre),('lr',LogisticRegression(C=C,solver='lbfgs',max_iter=1000))])

def metrics(y,p):
    eps=1e-15;p=np.clip(np.asarray(p),eps,1-eps)
    return {'n':int(len(y)),'positive_rate':float(np.mean(y)),'predicted_rate':float(np.mean(p)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None}

def band(v,cuts):
    for lo,hi,name in cuts:
        if lo<=v<=hi:return name
    return 'other'

def context_calibration(test,p):
    groups={'inning':{},'starter':{},'score_band':{},'bf_band':{}}
    defs={
      'inning':lambda x:str(int(float(x['inning']))),
      'starter':lambda x:str(x['pitcher_is_starter']),
      'score_band':lambda x:band(float(x['defensive_score_diff']),[(-99,-4,'<=-4'),(-3,-2,'-3:-2'),(-1,1,'-1:1'),(2,3,'2:3'),(4,99,'>=4')]),
      'bf_band':lambda x:band(float(x['pitcher_batters_faced_game_to_date']),[(0,9,'0-9'),(10,17,'10-17'),(18,24,'18-24'),(25,99,'25+')])}
    worst=0.0
    for name,fn in defs.items():
        tmp={}
        for i,(_,x,y) in enumerate(test):
            try:k=fn(x)
            except Exception:continue
            tmp.setdefault(k,[]).append(i)
        for k,idx in tmp.items():
            if len(idx)<100:continue
            obs=float(np.mean([test[i][2] for i in idx]));pred=float(np.mean(p[idx]));gap=abs(obs-pred);worst=max(worst,gap)
            groups[name][k]={'n':len(idx),'observed_rate':obs,'predicted_rate':pred,'abs_gap':gap}
    return groups,worst

def latency(model,X):
    sample=X[:min(len(X),5000)];times=[]
    for i in range(len(sample)):
        t=time.perf_counter_ns();model.predict_proba(sample[i:i+1]);times.append((time.perf_counter_ns()-t)/1e6)
    a=np.asarray(times)
    return {'n':int(len(a)),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95)),'p99_ms':float(np.percentile(a,99)),'max_ms':float(a.max())}

def main():
    rows=load_rows();train=[r for r in rows if r[0]<=20250731];valid=[r for r in rows if 20250801<=r[0]<=20250831];test=[r for r in rows if r[0]>=20250901]
    if min(map(len,(train,valid,test)))<1000:raise RuntimeError('temporal split too small')
    Xt,yt,cols=matrix(train);Xv,yv,_=matrix(valid);Xq,yq,_=matrix(test);tune=[]
    for C in CANDIDATE_C:
        m=make_pipe(C);m.fit(Xt,yt);pv=m.predict_proba(Xv)[:,1];mm=metrics(yv,pv);mm['C']=C;tune.append(mm)
    best=min(tune,key=lambda z:(z['log_loss'],z['brier']));dev=train+valid;Xd,yd,_=matrix(dev);model=make_pipe(best['C']);model.fit(Xd,yd);pq=model.predict_proba(Xq)[:,1]
    test_m=metrics(yq,pq);base_rate=float(np.mean(yd));base=np.full_like(yq,base_rate,dtype=float);base_m=metrics(yq,base)
    ctx,worst_gap=context_calibration(test,pq);lat=latency(model,Xq)
    # Context gap threshold is intentionally moderate for rare-event subgroups; if violated, block promotion and recalibrate rather than tune on locked test.
    gates={'beats_constant_brier':test_m['brier']<base_m['brier'],'beats_constant_log_loss':test_m['log_loss']<base_m['log_loss'],'worst_context_abs_gap_le_0_05':worst_gap<=0.05,'latency_p99_lt_1000ms':lat['p99_ms']<1000.0}
    report={'market_blind':True,'task':'P(pitcher change before next PA) for in-inning PA boundaries','feature_columns':cols,
      'anti_overfit_controls':['chronological train/validation/locked-test split','small regularization grid','no pitcher identity','locked test not used for tuning','missingness handled inside development pipeline'],
      'split':{'train_n':len(train),'validation_n':len(valid),'locked_test_n':len(test)},'regularization_tuning':tune,'selected_C':best['C'],
      'locked_test_model':test_m,'locked_test_constant_rate_baseline':base_m,'context_calibration':ctx,'worst_context_abs_gap':worst_gap,'latency':lat,'promotion_gates':gates,
      'promotion_status':'PASS' if all(gates.values()) else 'BLOCKED','scope_warning':'Between-inning removal and reliever selection remain separate required models.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'in_inning_removal_hazard_validation.json').write_text(json.dumps(report,indent=2))
    with (OUT/'in_inning_removal_hazard_model.pkl').open('wb') as f:pickle.dump(model,f)
    print(json.dumps(report,indent=2))
    if report['promotion_status']!='PASS':raise SystemExit('Removal hazard locked-test gate blocked')
if __name__=='__main__':main()

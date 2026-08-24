#!/usr/bin/env python3
"""Fit an empirical market-blind in-inning pitcher-removal hazard model.

Input is the persisted 2025 PA-boundary artifact from bullpen_transition_history.
Governance:
- train: through 2025-07-31
- tune regularization: 2025-08-01..2025-08-31
- locked test: 2025-09-01 onward
- no sportsbook/market inputs
- no pitcher identity feature in this first structural model (anti-overfit safeguard)

The model predicts P(current pitcher is replaced before the next PA | current baseball state).
"""
from __future__ import annotations
import csv, json, pickle
from datetime import date
from pathlib import Path
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions'
CSV_PATH=BASE/'bullpen_transition_pa_boundaries_2025.csv'
OUT=BASE/'removal_hazard'

NUM=['inning','pitcher_batters_faced_game_to_date','defensive_score_diff']
CAT=['top_bot','outs_pre','bases_mask','lineup_position','pitcher_is_game_first_for_defense']
CANDIDATE_C=[0.01,0.03,0.1,0.3,1.0]


def parse_date(s):
    """Return YYYYMMDD from either ISO-ish date text or Python date ordinal."""
    raw=str(s).strip()
    digits=''.join(ch for ch in raw if ch.isdigit())
    if digits and len(digits)<=6:
        d=date.fromordinal(int(digits))
        return d.year*10000+d.month*100+d.day
    if len(digits)>=8:
        return int(digits[:8])
    raise ValueError(f'unrecognized date value: {s!r}')


def load_rows():
    rows=[]
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            try:
                d=parse_date(r['date']); y=int(r['pitcher_changed_before_next_pa'])
                vals={k:r.get(k,'') for k in NUM+CAT}
                for k in NUM:
                    v=vals[k]
                    vals[k]=float(v) if v not in ('',None,'None') else np.nan
                for k in CAT:
                    vals[k]=str(vals[k])
                rows.append((d,vals,y))
            except Exception:
                continue
    return rows


def matrix(rows):
    X=[x for _,x,_ in rows]; y=np.array([y for *_,y in rows],dtype=int)
    cols=NUM+CAT
    arr=np.empty((len(X),len(cols)),dtype=object)
    for i,x in enumerate(X):
        for j,c in enumerate(cols):arr[i,j]=x[c]
    return arr,y,cols


def make_pipe(C):
    num_idx=list(range(len(NUM))); cat_idx=list(range(len(NUM),len(NUM)+len(CAT)))
    pre=ColumnTransformer([
        ('num',Pipeline([
            ('impute',SimpleImputer(strategy='median',add_indicator=True)),
            ('scale',StandardScaler()),
        ]),num_idx),
        ('cat',Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore')),
        ]),cat_idx),
    ])
    return Pipeline([('pre',pre),('lr',LogisticRegression(C=C,solver='lbfgs',max_iter=1000))])


def metrics(y,p):
    eps=1e-15;p=np.clip(np.asarray(p),eps,1-eps)
    return {'n':int(len(y)),'positive_rate':float(np.mean(y)),'brier':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None}


def main():
    rows=load_rows()
    train=[r for r in rows if r[0]<=20250731]
    valid=[r for r in rows if 20250801<=r[0]<=20250831]
    test=[r for r in rows if r[0]>=20250901]
    split_sizes={'train':len(train),'validation':len(valid),'locked_test':len(test)}
    if min(split_sizes.values())<1000:raise RuntimeError(f'temporal split too small: {split_sizes}')
    Xt,yt,cols=matrix(train); Xv,yv,_=matrix(valid); Xq,yq,_=matrix(test)
    tune=[]
    for C in CANDIDATE_C:
        m=make_pipe(C);m.fit(Xt,yt);pv=m.predict_proba(Xv)[:,1]
        mm=metrics(yv,pv);mm['C']=C;tune.append(mm)
    best=min(tune,key=lambda z:(z['log_loss'],z['brier']))
    dev=train+valid;Xd,yd,_=matrix(dev)
    model=make_pipe(best['C']);model.fit(Xd,yd);pq=model.predict_proba(Xq)[:,1]
    test_m=metrics(yq,pq)
    base_rate=float(np.mean(yd));base=np.full_like(yq,base_rate,dtype=float);base_m=metrics(yq,base)
    report={
        'market_blind':True,
        'task':'P(pitcher change before next PA) for in-inning PA boundaries',
        'data_source':'persisted 2025 Retrosheet bullpen transition artifact',
        'anti_overfit_controls':['chronological train/validation/locked-test split','small regularization grid','no pitcher identity in first structural model','locked test not used for tuning','numeric missingness imputed from development data only'],
        'feature_columns':cols,
        'split':{'train_through':'2025-07-31','validation':'2025-08-01..2025-08-31','locked_test':'2025-09-01 onward','train_n':len(train),'validation_n':len(valid),'locked_test_n':len(test)},
        'regularization_tuning':tune,
        'selected_C':best['C'],
        'locked_test_model':test_m,
        'locked_test_constant_rate_baseline':base_m,
        'improvement_vs_constant':{'brier':base_m['brier']-test_m['brier'],'log_loss':base_m['log_loss']-test_m['log_loss']},
        'promotion_status':'PASS' if test_m['brier']<base_m['brier'] and test_m['log_loss']<base_m['log_loss'] else 'BLOCKED',
        'scope_warning':'This model covers in-inning removal only. Between-inning removal is a separate hazard and remains required for full-game promotion.',
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'in_inning_removal_hazard_validation.json').write_text(json.dumps(report,indent=2))
    with (OUT/'in_inning_removal_hazard_model.pkl').open('wb') as f:pickle.dump(model,f)
    print(json.dumps(report,indent=2))
    if report['promotion_status']!='PASS':raise SystemExit('Removal hazard locked-test gate blocked')

if __name__=='__main__':main()

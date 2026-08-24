#!/usr/bin/env python3
"""Export fitted sklearn bullpen models into browser-safe JSON.
Runs only after the existing validation scripts have produced their pickle files.
No fitting, retuning, or market data is performed here.
"""
from __future__ import annotations
import json,pickle
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/derived/model_calibration/bullpen_transitions'
OUT=BASE/'bullpen_production_models.json'

MODELS={
 'in_inning_removal':BASE/'removal_hazard/in_inning_removal_hazard_model.pkl',
 'between_inning_removal':BASE/'between_inning_hazard/between_inning_removal_hazard_model.pkl',
}

def py(v):
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(list,tuple)): return [py(x) for x in v]
    if isinstance(v,dict): return {str(k):py(x) for k,x in v.items()}
    return v

def export_pipe(path):
    with path.open('rb') as f:m=pickle.load(f)
    pre=m.named_steps['pre'];lr=m.named_steps['lr']
    out={'intercept':float(lr.intercept_[0]),'coef':[float(x) for x in lr.coef_[0]],'transformers':[]}
    for name,pipe,cols in pre.transformers_:
        if name=='remainder':continue
        rec={'name':name,'columns':[int(x) for x in cols]}
        if name=='num':
            imp=pipe.named_steps['impute'];sc=pipe.named_steps['scale']
            rec.update({'statistics':py(imp.statistics_),'add_indicator':bool(getattr(imp,'add_indicator',False)),
                        'indicator_features':py(getattr(getattr(imp,'indicator_',None),'features_',[])),
                        'mean':py(sc.mean_),'scale':py(sc.scale_)})
        elif name=='cat':
            imp=pipe.named_steps['impute'];oh=pipe.named_steps['onehot']
            rec.update({'statistics':py(imp.statistics_),'categories':py(oh.categories_)})
        out['transformers'].append(rec)
    return out

def main():
    missing=[str(p) for p in MODELS.values() if not p.exists()]
    if missing: raise SystemExit('Missing fitted models: '+', '.join(missing))
    payload={'market_blind':True,'frozen_export':True,'models':{k:export_pipe(p) for k,p in MODELS.items()}}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'status':'PASS','out':str(OUT),'models':list(payload['models'])},indent=2))
if __name__=='__main__':main()

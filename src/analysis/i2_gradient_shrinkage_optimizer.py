#!/usr/bin/env python3
"""Empirically fit a monotone probability-dependent shrinkage gradient for I2.

Price-blind. Uses only chronological OOS replay predictions from fixed shrinkage
variants. The gradient is selected on 2022-2024 and evaluated once on sealed 2025.

Method
------
1. Anchor games by the current p100 baseline P(Under).
2. Place five knots at development-sample p100 quantiles 10/30/50/70/90%.
3. Search monotone non-increasing shrinkage values as P(Under) rises.
4. For each game, interpolate shrinkage across probability knots, then interpolate
   that game's predicted P(Under) across the discrete replay shrinkage variants.
5. Select by development log loss; break near-ties toward smoother gradients.
6. Report development and sealed-2025 log loss/Brier/calibration by overall and
   fixed p100 quartiles.

This diagnoses the strict historical replay, not the byte-for-byte live simulator.
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['game_id','season']
SHRINK_GRID=np.array([0.,10.,25.,50.,75.,100.,200.])
KNOT_Q=[0.10,0.30,0.50,0.70,0.90]


def clip(p): return np.clip(np.asarray(p,float),1e-9,1-1e-9)
def logloss(y,p):
    y=np.asarray(y,float); p=clip(p)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((y-p)**2))
def cal_gap(y,p): return float((np.mean(y)-np.mean(p))*100)

def load_variant(path:Path, name:str):
    x=pd.read_csv(path)
    x=x[KEY+['baseline_prediction','actual_over']].copy()
    x[f'p_under_{name}']=1-pd.to_numeric(x['baseline_prediction'],errors='coerce')
    x['actual_under']=1-pd.to_numeric(x['actual_over'],errors='coerce')
    return x.drop(columns=['baseline_prediction','actual_over'])

def build_panel(root:Path, variants:list[dict]):
    merged=None
    for v in variants:
        q=load_variant(root/v['name']/'strict_oos_predictions.csv',v['name'])
        if merged is None: merged=q
        else: merged=merged.merge(q.drop(columns=['actual_under']),on=KEY,how='inner',validate='one_to_one')
    return merged

def interp_predictions(panel:pd.DataFrame, target_s:np.ndarray, variants:list[dict]):
    pairs=sorted((float(v['prior_strength']),v['name']) for v in variants)
    sx=np.array([a for a,_ in pairs],float)
    mat=np.column_stack([panel[f'p_under_{n}'].to_numpy(float) for _,n in pairs])
    out=np.empty(len(panel),float)
    for i,s in enumerate(target_s): out[i]=np.interp(s,sx,mat[i])
    return out

def gradient_s(anchor_p:np.ndarray, knots_p:np.ndarray, knot_s:tuple[float,...]):
    return np.interp(anchor_p,knots_p,np.asarray(knot_s,float),left=knot_s[0],right=knot_s[-1])

def cohort_labels(p):
    q=np.quantile(p,[.25,.50,.75])
    return np.select([p<=q[0],p<=q[1],p<=q[2]],['Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL'],default='Q4_HIGH_UNDER')

def eval_rows(panel,pred,split,labels):
    rows=[]
    idx=np.ones(len(panel),dtype=bool) if split=='ALL' else (panel.season.to_numpy()==int(split))
    for cohort in ['ALL','Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL','Q4_HIGH_UNDER']:
        m=idx if cohort=='ALL' else idx & (labels==cohort)
        y=panel.actual_under.to_numpy(float)[m]; p=pred[m]
        if len(y)==0: continue
        rows.append({'split':str(split),'cohort':cohort,'n':int(len(y)),'log_loss':logloss(y,p),'brier':brier(y,p),
                     'mean_p_under':float(np.mean(p)),'actual_under_rate':float(np.mean(y)),'calibration_gap_pp':cal_gap(y,p)})
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); variants=json.loads(Path(a.manifest).read_text())
    panel=build_panel(Path(a.variants_dir),variants).sort_values(KEY).reset_index(drop=True)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if 'p100' not in [v['name'] for v in variants]: raise SystemExit('p100 variant required')
    panel['anchor_p_under']=panel['p_under_p100']
    dev=panel.season.isin([2022,2023,2024]).to_numpy(); hold=panel.season.eq(2025).to_numpy()
    knots_p=panel.loc[dev,'anchor_p_under'].quantile(KNOT_Q).to_numpy(float)

    # Monotone non-increasing shrinkage as Under probability rises.
    vals=list(SHRINK_GRID)
    candidates=[]
    ydev=panel.actual_under.to_numpy(float)[dev]
    anchor=panel.anchor_p_under.to_numpy(float)
    for ks in itertools.product(vals, repeat=len(knots_p)):
        if not all(ks[i]>=ks[i+1] for i in range(len(ks)-1)): continue
        ts=gradient_s(anchor,knots_p,ks)
        pred=interp_predictions(panel,ts,variants)
        ll=logloss(ydev,pred[dev])
        rough=sum(abs(ks[i+1]-ks[i]) for i in range(len(ks)-1))
        candidates.append((ll,rough,ks,pred,ts))
    candidates.sort(key=lambda z:(round(z[0],6),z[1],z[0]))
    best=candidates[0]
    best_ll,best_rough,best_ks,best_pred,best_s=best

    # Constant benchmarks from available variants.
    bench=[]
    for v in variants:
        p=panel[f"p_under_{v['name']}"].to_numpy(float)
        bench.append({'variant':v['name'],'prior_strength':v['prior_strength'],'dev_log_loss':logloss(ydev,p[dev]),
                      'holdout_2025_log_loss':logloss(panel.actual_under.to_numpy(float)[hold],p[hold]),
                      'dev_brier':brier(ydev,p[dev]),'holdout_2025_brier':brier(panel.actual_under.to_numpy(float)[hold],p[hold])})
    pd.DataFrame(bench).sort_values('dev_log_loss').to_csv(out/'constant_benchmarks.csv',index=False)

    # Frozen p100 quartiles from development anchor only, applied via cutpoints to all rows.
    qc=panel.loc[dev,'anchor_p_under'].quantile([.25,.50,.75]).to_numpy(float)
    labels=np.select([anchor<=qc[0],anchor<=qc[1],anchor<=qc[2]],['Q1_HIGH_OVER','Q2_MODERATE_OVER_NEUTRAL','Q3_MODERATE_UNDER_NEUTRAL'],default='Q4_HIGH_UNDER')
    rows=[]
    for split in ['ALL',2022,2023,2024,2025]: rows+=eval_rows(panel,best_pred,split,labels)
    pd.DataFrame(rows).to_csv(out/'gradient_performance.csv',index=False)

    pd.DataFrame({'game_id':panel.game_id,'season':panel.season,'anchor_p_under':anchor,'effective_shrinkage':best_s,
                  'gradient_p_under':best_pred,'actual_under':panel.actual_under,'fixed_cohort':labels}).to_csv(out/'gradient_oos_predictions.csv',index=False)

    top=[]
    for ll,rough,ks,_,_ in candidates[:50]: top.append({'dev_log_loss':ll,'roughness':rough,**{f'knot{i+1}_shrinkage':ks[i] for i in range(len(ks))}})
    pd.DataFrame(top).to_csv(out/'top_gradient_candidates.csv',index=False)

    manifest={'status':'PASS','price_used':False,'market_fields_used':[],
              'development_seasons':[2022,2023,2024],'sealed_validation_season':2025,
              'anchor':'p100 baseline P(Under)','knot_quantiles':KNOT_Q,'knot_probabilities':[float(x) for x in knots_p],
              'shrinkage_grid':[float(x) for x in SHRINK_GRID], 'monotonic_constraint':'non-increasing shrinkage as P(Under) rises',
              'selected_knot_shrinkage':[float(x) for x in best_ks],'development_log_loss':best_ll,'roughness':best_rough,
              'selection_rule':'minimum development log loss; six-decimal near-ties prefer smoother gradient',
              'caveat':'Strict chronological OOS historical replay; not byte-for-byte live simulator replay.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    print('\nCONSTANT BENCHMARKS')
    print(pd.DataFrame(bench).sort_values('dev_log_loss').to_string(index=False))
    print('\nGRADIENT PERFORMANCE')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()

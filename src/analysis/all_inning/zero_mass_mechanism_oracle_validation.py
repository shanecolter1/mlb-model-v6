#!/usr/bin/env python3
"""Mechanism screening for persistent full-inning U0.5 zero-mass residuals.

Research-only oracle layer. It uses the first three realized batters in each half inning
and the realized first pitcher in that half to ask *which pre-date skill dimensions*
explain persistent scoreless-inning departures after conditioning on the opening game
total and inning. Participant identities are therefore NOT production-pregame features.

2025 remains sealed. All player rates are strictly prior-date from the existing M1
artifact. Outcome-dependent PA-count leakage is prevented by using exactly the first
three PAs per played half (and only the first PA for pitcher identity/rates).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EPS=1e-9
TARGETS=[(8.5,8),(8.5,6),(9.0,6),(8.5,3),(8.0,3)]
FAMILIES={
    'strikeout':['batter_365d_k_rate','pitcher_365d_k_rate'],
    'baserunner':['batter_365d_baserunner_rate','pitcher_365d_baserunner_rate'],
    'home_run':['batter_365d_hr_rate','pitcher_365d_hr_rate'],
    'nonhr_hit':['batter_365d_nonhr_hit_rate','pitcher_365d_nonhr_hit_rate'],
    'platoon':['platoon_same_hand'],
}
FEATURES=[c for v in FAMILIES.values() for c in v]


def sigmoid(x): return 1/(1+np.exp(-np.clip(np.asarray(x,float),-35,35)))
def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def ll(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def brier(y,p): return float(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))

def fit_ridge(X,y,offset,lam=8.0,max_iter=100):
    b=np.zeros(X.shape[1]); I=np.eye(X.shape[1])*lam
    for _ in range(max_iter):
        p=sigmoid(offset+X@b); w=np.clip(p*(1-p),1e-6,None)
        g=X.T@(y-p)-I@b; H=X.T@(w[:,None]*X)+I
        step=np.linalg.solve(H,g); nb=b+step
        if np.max(np.abs(nb-b))<1e-8: b=nb; break
        b=nb
    return b

def standardize_fit(a):
    mu=np.nanmean(a,axis=0); sd=np.nanstd(a,axis=0)
    sd[(~np.isfinite(sd))|(sd<1e-9)]=1.0
    return mu,sd

def standardize(a,mu,sd):
    z=(a-mu)/sd
    return np.where(np.isfinite(z),z,0.0)

def build_game_features(pa, half):
    pa=pa.copy(); pa['half']=pa['half'].astype(str).str.lower()
    pa=pa.sort_values(['game_date','game_id','inning','half','play_index'],kind='mergesort')
    # Exactly first 3 PAs: fixed exposure avoids using inning length as a feature.
    pa['pa_rank']=pa.groupby(['game_id','inning','half']).cumcount()+1
    first3=pa[pa.pa_rank<=3].copy()
    # Batter dimensions are mean of first three scheduled/realized hitters.
    bat_cols=[c for c in FEATURES if c.startswith('batter_')] + ['platoon_same_hand']
    bat=(first3.groupby(['game_id','inning','half'],as_index=False)[bat_cols].mean())
    # Pitcher state is the pitcher facing the first PA only; do not average over changes.
    pit_cols=[c for c in FEATURES if c.startswith('pitcher_')]
    pit=(first3[first3.pa_rank==1][['game_id','inning','half']+pit_cols].copy())
    h=bat.merge(pit,on=['game_id','inning','half'],how='inner',validate='one_to_one')
    # Collapse top+bottom to full-inning feature signature. Mean captures aggregate pressure;
    # absolute half difference captures asymmetry without observing runs.
    wide=[]
    for (gid,inn),g in h.groupby(['game_id','inning']):
        if set(g.half)!={'top','bottom'}: continue
        row={'game_id':gid,'inning':int(inn)}
        for c in FEATURES:
            vals=pd.to_numeric(g[c],errors='coerce').to_numpy(float)
            row[c]=float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[c+'_half_absdiff']=float(np.abs(vals[0]-vals[1])) if len(vals)==2 and np.isfinite(vals).all() else np.nan
        wide.append(row)
    return pd.DataFrame(wide)

def build_outcomes(half):
    h=half[(half.half_played==True)&(half.inning<=8)].copy()
    g=(h.groupby(['game_id','game_date','season','inning','dk_total_open_total'],as_index=False)
       .agg(halves=('half','nunique'),runs=('runs_half','sum')))
    g=g[g.halves==2].copy(); g['under05']=(g.runs==0).astype(int)
    return g

def fold_eval(d,feature_cols,lam=8.0):
    rows=[]; coefs=[]; preds=[]
    years=sorted(d.season.unique())
    for yr in years[1:]:
        tr=d[d.season<yr].copy(); te=d[d.season==yr].copy()
        if len(tr)<150 or len(te)<80: continue
        base=float(tr.under05.mean()); p0=np.full(len(te),base)
        A=tr[feature_cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        B=te[feature_cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        mu,sd=standardize_fit(A); X=standardize(A,mu,sd); Z=standardize(B,mu,sd)
        b=fit_ridge(X,tr.under05.to_numpy(float),np.full(len(tr),logit(base)),lam)
        p=sigmoid(logit(base)+Z@b); y=te.under05.to_numpy(float)
        rows.append({'test_season':int(yr),'n_train':len(tr),'n_test':len(te),'baseline_p0':base,
                     'baseline_logloss':ll(y,p0),'model_logloss':ll(y,p),
                     'logloss_improvement':ll(y,p0)-ll(y,p),
                     'baseline_brier':brier(y,p0),'model_brier':brier(y,p),
                     'brier_improvement':brier(y,p0)-brier(y,p)})
        for c,v in zip(feature_cols,b): coefs.append({'test_season':int(yr),'feature':c,'beta':float(v)})
        q=te[['game_id','season','under05']].copy(); q['baseline']=p0; q['pred']=p; q['delta']=p-p0; preds.append(q)
    return pd.DataFrame(rows),pd.DataFrame(coefs),pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pa-matrix',type=Path,required=True); ap.add_argument('--half-matrix',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    pa=pd.read_parquet(a.pa_matrix); half=pd.read_parquet(a.half_matrix)
    if pa.season.max()>2024 or half.season.max()>2024: raise RuntimeError('2025 holdout leakage')
    missing=[c for c in FEATURES if c not in pa.columns]
    if missing: raise RuntimeError(f'M1 matrix missing {missing}')
    feat=build_game_features(pa,half); out=build_outcomes(half)
    d=out.merge(feat,on=['game_id','inning'],how='inner',validate='one_to_one')
    all_features=FEATURES+[c+'_half_absdiff' for c in FEATURES]
    results=[]; coef_all=[]; ablations=[]; strat=[]
    for total,inn in TARGETS:
        z=d[(d.dk_total_open_total==total)&(d.inning==inn)].copy()
        full,coef,pred=fold_eval(z,all_features)
        if full.empty: continue
        full['pregame_total']=total; full['inning']=inn; results.append(full)
        coef['pregame_total']=total; coef['inning']=inn; coef_all.append(coef)
        full_ll=float(full.model_logloss.mean()); full_br=float(full.model_brier.mean())
        for fam,cols in FAMILIES.items():
            remove=set(cols+[c+'_half_absdiff' for c in cols]); kept=[c for c in all_features if c not in remove]
            f,_,_=fold_eval(z,kept)
            if f.empty: continue
            ablations.append({'pregame_total':total,'inning':inn,'removed_family':fam,
                              'full_mean_logloss':full_ll,'ablated_mean_logloss':float(f.model_logloss.mean()),
                              'delta_logloss_if_removed':float(f.model_logloss.mean()-full_ll),
                              'full_mean_brier':full_br,'ablated_mean_brier':float(f.model_brier.mean()),
                              'delta_brier_if_removed':float(f.model_brier.mean()-full_br)})
        # Out-of-fold predicted-delta quintiles: does the oracle isolate meaningful P0 dispersion?
        if not pred.empty:
            try: pred['q']=pd.qcut(pred.delta,5,labels=False,duplicates='drop')
            except ValueError: pred['q']=0
            for q,g in pred.groupby('q'):
                strat.append({'pregame_total':total,'inning':inn,'delta_quintile':int(q)+1,'n':len(g),
                              'mean_model_delta_pp':100*float(g.delta.mean()),'actual_under_pct':100*float(g.under05.mean()),
                              'mean_pred_under_pct':100*float(g.pred.mean())})
    R=pd.concat(results,ignore_index=True); C=pd.concat(coef_all,ignore_index=True); A=pd.DataFrame(ablations); S=pd.DataFrame(strat)
    R.to_csv(a.output_dir/'walk_forward_metrics.csv',index=False); C.to_csv(a.output_dir/'coefficients.csv',index=False); A.to_csv(a.output_dir/'family_ablation.csv',index=False); S.to_csv(a.output_dir/'oof_delta_quintiles.csv',index=False)
    summary=(R.groupby(['pregame_total','inning'],as_index=False).agg(n_test=('n_test','sum'),mean_baseline_logloss=('baseline_logloss','mean'),mean_model_logloss=('model_logloss','mean'),mean_logloss_improvement=('logloss_improvement','mean'),mean_brier_improvement=('brier_improvement','mean')))
    summary.to_csv(a.output_dir/'target_summary.csv',index=False)
    fam=(A.sort_values(['pregame_total','inning','delta_logloss_if_removed'],ascending=[True,True,False]).groupby(['pregame_total','inning'],as_index=False).head(1)) if len(A) else A
    fam.to_csv(a.output_dir/'strongest_family_by_target.csv',index=False)
    manifest={'status':'PASS','architecture':'zero_mass_mechanism_oracle_validation','development_seasons':sorted(int(x) for x in d.season.unique()),'holdout_season':2025,'holdout_opened':False,'targets':[{'pregame_total':t,'inning':i} for t,i in TARGETS],'participant_identity_class':'retrospective_realized_first-three-batters_and_first-pitcher_oracle','statistics_timing':'strictly_prior_date','outcome_length_leakage_control':'exactly_first_three_PA_per_half; first-pitcher from PA1 only','market_inputs':['opening full-game total only'],'not_production_betting_model':True,'summary':summary.to_dict('records'),'strongest_family_by_target':fam.to_dict('records') if len(fam) else []}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

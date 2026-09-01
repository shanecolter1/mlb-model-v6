#!/usr/bin/env python3
"""Chronological incremental Statcast family screen beyond validated M1 core.

Hyperparameter/window selection uses 2022 only, 2023 is untouched development
validation, and 2024 is final development confirmation. 2025 is never loaded.
Each challenger and its core comparator are scored on identical complete-case
rows. No missing-value imputation is introduced.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS=['k','baserunner','hr','nonhr_hit']
WINDOWS=['30d','90d','365d','season']
RIDGES=[0.0,0.1,1.0,10.0,100.0,1000.0]
EPS=1e-10
PLATOON_EVENTS={'k','baserunner','hr'}
CORE_SPECS={
 'k':{'window':'365d','spec':'additive_interaction'},
 'baserunner':{'window':'365d','spec':'additive_interaction'},
 'hr':{'window':'365d','spec':'additive_interaction'},
 'nonhr_hit':{'window':'365d','spec':'additive'},
}
FAMILY_TEMPLATES={
 'discipline_contact':['batter_{w}_swing_rate','batter_{w}_whiff_rate','batter_{w}_chase_rate','batter_{w}_contact_rate','batter_{w}_zone_rate','pitcher_{w}_swing_rate','pitcher_{w}_whiff_rate','pitcher_{w}_chase_rate','pitcher_{w}_contact_rate','pitcher_{w}_zone_rate'],
 'pitch_mix_matchup':['pitcher_{w}_mix_fb','pitcher_{w}_mix_breaking','pitcher_{w}_mix_offspeed','batter_{w}_fb_whiff_rate','batter_{w}_breaking_whiff_rate','batter_{w}_offspeed_whiff_rate','batter_{w}_fb_xwoba_contact','batter_{w}_breaking_xwoba_contact','batter_{w}_offspeed_xwoba_contact'],
 'velocity_spin_extension':['pitcher_{w}_fb_velo','pitcher_{w}_fb_spin','pitcher_{w}_extension','batter_{w}_fb_velo','batter_{w}_fb_spin','batter_{w}_extension'],
 'contact_quality_expected':['batter_{w}_avg_ev','batter_{w}_hardhit_rate','batter_{w}_barrel_rate','batter_{w}_avg_launch_angle','batter_{w}_xwoba_contact','batter_{w}_xba_contact','pitcher_{w}_avg_ev','pitcher_{w}_hardhit_rate','pitcher_{w}_barrel_rate','pitcher_{w}_avg_launch_angle','pitcher_{w}_xwoba_contact','pitcher_{w}_xba_contact']
}

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-35,35)))
def ll(y,p):
    p=np.clip(p,EPS,1-EPS); return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def br(y,p): return float(np.mean((y-p)**2))
def fit_logistic(X,y,ridge=0.0,penalty_start=None,max_iter=25):
    b=np.zeros(X.shape[1]); pen=np.zeros((X.shape[1],X.shape[1]))
    if penalty_start is not None and ridge>0:
        for j in range(int(penalty_start),X.shape[1]): pen[j,j]=1.0
    for _ in range(max_iter):
        p=sigmoid(X@b); w=np.maximum(p*(1-p),1e-7)
        g=X.T@(p-y)+ridge*(pen@b); H=X.T@(X*w[:,None])+ridge*pen+1e-8*np.eye(X.shape[1])
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: step=np.linalg.pinv(H)@g
        b2=b-step
        if np.max(np.abs(b2-b))<1e-7: b=b2; break
        b=b2
    return b

def standardize(train,test,cols):
    A=train[cols].to_numpy(float); B=test[cols].to_numpy(float)
    mu=A.mean(0); sd=A.std(0); sd[~np.isfinite(sd)|(sd<1e-10)]=1
    return (A-mu)/sd,(B-mu)/sd

def core_rate_cols(event):
    window=CORE_SPECS[event]['window']
    return [f'batter_{window}_{event}_rate',f'pitcher_{window}_{event}_rate']

def core_matrix(df,event):
    b,p=core_rate_cols(event); bv=df[b].to_numpy(float); pv=df[p].to_numpy(float); inn=df.inning.to_numpy(int)
    cols=[np.ones(len(df))]+[(inn==i).astype(float) for i in range(2,10)]+[bv,pv]
    if CORE_SPECS[event]['spec']=='additive_interaction': cols.append(bv*pv)
    if event in PLATOON_EVENTS:
        bl=(df.batter_side.astype(str).str.upper()=='L').astype(float).to_numpy(); pl=(df.pitcher_hand.astype(str).str.upper()=='L').astype(float).to_numpy(); cols += [bl,pl,bl*pl]
    return np.column_stack(cols)
def family_cols(family,window): return [s.format(w=window) for s in FAMILY_TEMPLATES[family]]
def add_pitch_mix_interactions(tr,te,window,Ftr,Fte):
    for side in ['fb','breaking','offspeed']:
        mix=tr[f'pitcher_{window}_mix_{side}'].to_numpy(float); mixte=te[f'pitcher_{window}_mix_{side}'].to_numpy(float)
        for response in ['whiff_rate','xwoba_contact']:
            bv=tr[f'batter_{window}_{side}_{response}'].to_numpy(float); bvte=te[f'batter_{window}_{side}_{response}'].to_numpy(float); v=mix*bv; vt=mixte*bvte; mu=v.mean(); sd=v.std(); sd=sd if np.isfinite(sd) and sd>=1e-10 else 1.0
            Ftr=np.column_stack([Ftr,(v-mu)/sd]); Fte=np.column_stack([Fte,(vt-mu)/sd])
    return Ftr,Fte

def evaluate(x,event,family,window,year,ridge):
    ycol='y_'+event; ratecols=core_rate_cols(event); fcols=family_cols(family,window)
    needed=list(dict.fromkeys(['season','inning','batter_side','pitcher_hand',ycol]+ratecols+fcols))
    tr=x[x.season<year][needed].dropna().copy(); te=x[x.season==year][needed].dropna().copy()
    if len(tr)<5000 or len(te)<1000: return None
    ytr=tr[ycol].to_numpy(float); yte=te[ycol].to_numpy(float); Ctr=core_matrix(tr,event); Cte=core_matrix(te,event); Ftr,Fte=standardize(tr,te,fcols)
    if family=='pitch_mix_matchup': Ftr,Fte=add_pitch_mix_interactions(tr,te,window,Ftr,Fte)
    b0=fit_logistic(Ctr,ytr); p0=sigmoid(Cte@b0); Xtr=np.column_stack([Ctr,Ftr]); Xte=np.column_stack([Cte,Fte]); b1=fit_logistic(Xtr,ytr,ridge,Ctr.shape[1]); p1=sigmoid(Xte@b1)
    return {'event':event,'family':family,'window':window,'ridge':ridge,'test_year':year,'n_train':len(tr),'n_test':len(te),'core_logloss':ll(yte,p0),'challenger_logloss':ll(yte,p1),'logloss_improvement':ll(yte,p0)-ll(yte,p1),'core_brier':br(yte,p0),'challenger_brier':br(yte,p1),'brier_improvement':br(yte,p0)-br(yte,p1)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--m1-matrix',type=Path,required=True); ap.add_argument('--batter-asof',type=Path,required=True); ap.add_argument('--pitcher-asof',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.m1_matrix); b=pd.read_parquet(a.batter_asof); p=pd.read_parquet(a.pitcher_asof)
    for d in [x,b,p]: d['game_date']=pd.to_datetime(d.game_date,errors='coerce').dt.normalize()
    x['season']=pd.to_numeric(x.season,errors='raise').astype(int)
    if set(x.season.unique())!={2021,2022,2023,2024} or (x.season>=2025).any(): raise RuntimeError('development seasons must be exactly 2021-2024')
    x=x.merge(b,on=['batter_id','game_date'],how='left',validate='many_to_one').merge(p,on=['pitcher_id','game_date'],how='left',validate='many_to_one')
    selection=[]
    for e in EVENTS:
      for fam in FAMILY_TEMPLATES:
       for w in WINDOWS:
        for r in RIDGES:
          q=evaluate(x,e,fam,w,2022,r)
          if q is not None: selection.append(q)
    selgrid=pd.DataFrame(selection)
    if selgrid.empty: raise RuntimeError('no evaluable selection candidates')
    selected=selgrid.sort_values(['event','family','logloss_improvement','brier_improvement'],ascending=[True,True,False,False]).groupby(['event','family'],as_index=False).head(1).copy()
    val=[]; conf=[]
    for r in selected.itertuples():
        q=evaluate(x,r.event,r.family,r.window,2023,float(r.ridge)); c=evaluate(x,r.event,r.family,r.window,2024,float(r.ridge))
        if q is not None: val.append(q)
        if c is not None: conf.append(c)
    validation=pd.DataFrame(val); confirmation=pd.DataFrame(conf)
    status=selected[['event','family','window','ridge','logloss_improvement','brier_improvement']].rename(columns={'logloss_improvement':'selection_2022_logloss_improvement','brier_improvement':'selection_2022_brier_improvement'}).merge(validation[['event','family','logloss_improvement','brier_improvement','n_test']].rename(columns={'logloss_improvement':'validation_2023_logloss_improvement','brier_improvement':'validation_2023_brier_improvement','n_test':'validation_2023_n'}),on=['event','family'],how='left').merge(confirmation[['event','family','logloss_improvement','brier_improvement','n_test']].rename(columns={'logloss_improvement':'confirmation_2024_logloss_improvement','brier_improvement':'confirmation_2024_brier_improvement','n_test':'confirmation_2024_n'}),on=['event','family'],how='left')
    status['earns_incremental_candidate_status']=(status.selection_2022_logloss_improvement>0)&(status.validation_2023_logloss_improvement>0)&(status.confirmation_2024_logloss_improvement>0)&(status.confirmation_2024_brier_improvement>0)
    selgrid.to_csv(a.output_dir/'m1_statcast_family_selection_2022_grid.csv',index=False); selected.to_csv(a.output_dir/'m1_statcast_family_selected_specs.csv',index=False); validation.to_csv(a.output_dir/'m1_statcast_family_validation_2023.csv',index=False); confirmation.to_csv(a.output_dir/'m1_statcast_family_confirmation_2024.csv',index=False); status.to_csv(a.output_dir/'m1_statcast_family_gate_status.csv',index=False)
    manifest={'status':'PASS','architecture':'M1_incremental_statcast_feature_family_screen','development_seasons':[2021,2022,2023,2024],'selection_year':2022,'development_validation_year':2023,'confirmation_year':2024,'holdout_season':2025,'holdout_opened':False,'market_data_used':False,'core':'validated 365d M1 event-rate specifications + inning effects; confirmed handedness terms additionally retained for K/baserunner/HR','core_specs':CORE_SPECS,'families':list(FAMILY_TEMPLATES),'windows':WINDOWS,'ridge_candidates':RIDGES,'ridge_penalty_scope':'incremental Statcast coefficients only; validated core remains unpenalized','comparison_rows':'identical complete-case rows within each family/window; no imputation','automatic_production_promotion':False,'candidate_rule':'selected on 2022 only; positive logloss on 2022, untouched 2023 validation, and 2024 confirmation, with positive 2024 Brier','gate_status':status.to_dict('records')}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(status.to_string(index=False)); print(json.dumps({k:v for k,v in manifest.items() if k!='gate_status'},indent=2))
if __name__=='__main__': main()

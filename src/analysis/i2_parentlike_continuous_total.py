#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.analysis.i2_strict_asof_replay import build_game_key_from_normalized, join_master_to_games, team_feature_table, add_side_context, normalize_code, Standardizer

EPS=1e-9

def logloss(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.mean((y-p)**2))

def fit_ridge_regression(X,y,lam):
    X=np.asarray(X,float); y=np.asarray(y,float)
    Z=np.column_stack([np.ones(len(X)),X]); P=np.eye(Z.shape[1])*float(lam); P[0,0]=0
    return np.linalg.solve(Z.T@Z+P,Z.T@y)

def predict_ridge(beta,X):
    return np.column_stack([np.ones(len(X)),np.asarray(X,float)])@beta

def fit_under_map(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); xm=float(np.mean(x)); xs=float(np.std(x)) or 1.0; z=(x-xm)/xs
    def obj(b):
        eta=b[0]+b[1]*z; p=1/(1+np.exp(-np.clip(eta,-40,40)))
        return -np.sum(y*np.log(np.clip(p,EPS,1-EPS))+(1-y)*np.log(np.clip(1-p,EPS,1-EPS)))
    yb=np.clip(np.mean(y),EPS,1-EPS); init=[np.log(yb/(1-yb)),-0.2]
    r=minimize(obj,init,method='L-BFGS-B',bounds=[(None,None),(None,0.0)])
    return float(r.x[0]),float(r.x[1]),xm,xs

def pred_under_map(m,x):
    a,b,xm,xs=m; z=(np.asarray(x,float)-xm)/xs; return 1/(1+np.exp(-np.clip(a+b*z,-40,40)))

def find_score_cols(df):
    for a,h in [('away_final_runs','home_final_runs'),('away_score','home_score'),('away_runs','home_runs'),('away_team_runs','home_team_runs'),('visitor_score','home_score')]:
        if a in df.columns and h in df.columns: return a,h
    raise ValueError('final score columns not found')

def prepare(phase,v04):
    phase=Path(phase)
    raw=pd.read_csv(next(phase.rglob('MLB_Game_Stats_Joined_2021_2025.csv.gz')),low_memory=False)
    if 'benchmark_matched' in raw.columns: raw=raw[raw.benchmark_matched==True].copy()
    a,h=find_score_cols(raw)
    raw['season']=pd.to_numeric(raw.season,errors='coerce'); raw['game_date']=pd.to_datetime(raw.game_date,errors='coerce').dt.normalize()
    raw['away_team_code']=raw.away_team_code.map(normalize_code); raw['home_team_code']=raw.home_team_code.map(normalize_code)
    raw['i2_runs']=pd.to_numeric(raw.inning2_total_runs,errors='coerce'); raw['actual_under']=(raw.i2_runs==0).astype(float)
    raw['actual_total_runs']=pd.to_numeric(raw[a],errors='coerce')+pd.to_numeric(raw[h],errors='coerce')
    keep=['season','game_date','game_number','away_team_code','home_team_code','i2_runs','actual_under','actual_total_runs']
    if 'retro_game_id' in raw.columns: keep.append('retro_game_id')
    m=raw[keep].dropna(subset=['season','game_date','i2_runs','actual_total_runs']).copy(); m['season']=m.season.astype(int)
    games=build_game_key_from_normalized(next(phase.rglob('games.parquet'))); m=join_master_to_games(m,games)
    team,rate_cols=team_feature_table(next(phase.rglob('team_asof.parquet'))); m,features=add_side_context(m,team,rate_cols)
    v=pd.read_csv(next(Path(v04).rglob('v04_oos_predictions.csv')))
    cols=['game_id','season','p_under_local_cv','actual_under']
    vv=v[cols].drop_duplicates('game_id')
    m=m.merge(vv,on=['game_id','season'],how='inner',suffixes=('_hist','_v04'))
    if 'actual_under_v04' in m.columns: m['actual_under']=m.actual_under_v04
    elif 'actual_under_hist' in m.columns: m['actual_under']=m.actual_under_hist
    return m,features,len(v)

def fit_parent(train,features,lam):
    sc=Standardizer().fit(train[features].apply(pd.to_numeric,errors='coerce').to_numpy(float))
    X=sc.transform(train[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)); b=fit_ridge_regression(X,train.actual_total_runs.to_numpy(float),lam)
    return sc,b

def parent_predict(model,df,features):
    sc,b=model; X=sc.transform(df[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)); return np.clip(predict_ridge(b,X),4.0,15.0)

def choose_lambda(train,features):
    vals=[]
    for lam in [1,10,100,1000]:
        mse=[]
        for s in sorted(train.season.unique())[1:]:
            tt=train[train.season<s]; vv=train[train.season==s]
            if len(tt)<500 or vv.empty: continue
            p=parent_predict(fit_parent(tt,features,lam),vv,features); mse.append(float(np.mean((vv.actual_total_runs-p)**2)))
        vals.append((np.mean(mse) if mse else 999,lam))
    return min(vals)[1]

def crossfit_totals(train,features,lam):
    parts=[]
    for s in sorted(train.season.unique())[1:]:
        tt=train[train.season<s]; vv=train[train.season==s].copy()
        if len(tt)<500 or vv.empty: continue
        vv['parent_expected_total_cf']=parent_predict(fit_parent(tt,features,lam),vv,features); parts.append(vv)
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

def moments(s):
    s=pd.Series(s).dropna(); return {'mean':float(s.mean()),'sd':float(s.std()),'skew':float(s.skew()),'excess_kurtosis':float(s.kurt()),'p01':float(s.quantile(.01)),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase1',required=True); ap.add_argument('--v04',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    m,features,fulln=prepare(a.phase1,a.v04); print('joined_n',len(m),'v04_n',fulln,'features',len(features))
    rows=[]; annual=[]
    for season in [2023,2024,2025]:
        tr=m[m.season<season].copy(); te=m[m.season==season].copy()
        lam=choose_lambda(tr,features); cf=crossfit_totals(tr,features,lam)
        if cf.empty: continue
        mapper=fit_under_map(cf.parent_expected_total_cf,cf.actual_under)
        parent=fit_parent(tr,features,lam); te['parent_expected_total']=parent_predict(parent,te,features); te['p_under_parent_continuous']=pred_under_map(mapper,te.parent_expected_total); te['selected_parent_lambda']=lam; rows.append(te)
        annual.append({'season':season,'n':len(te),'parent_lambda':lam,'parent_total_rmse':float(np.sqrt(np.mean((te.actual_total_runs-te.parent_expected_total)**2))),'logloss':logloss(te.actual_under,te.p_under_parent_continuous),'brier':brier(te.actual_under,te.p_under_parent_continuous),'mean_p_under':float(te.p_under_parent_continuous.mean()),'actual_under':float(te.actual_under.mean())})
    o=pd.concat(rows,ignore_index=True); comp=[]
    for split,x in [('ALL',o),('DEV',o[o.season<=2024]),('2025',o[o.season==2025])]:
        for name,col in [('PARENTLIKE_CONTINUOUS','p_under_parent_continuous'),('V04_LOCAL_CV','p_under_local_cv')]:
            comp.append({'split':split,'model':name,'n':len(x),'logloss':logloss(x.actual_under,x[col]),'brier':brier(x.actual_under,x[col]),'mean_p_under':float(x[col].mean()),'actual_under':float(x.actual_under.mean())})
    pd.DataFrame(comp).to_csv(out/'performance_comparison.csv',index=False); pd.DataFrame(annual).to_csv(out/'annual_metrics.csv',index=False)
    o[['game_id','season','parent_expected_total','actual_total_runs','p_under_parent_continuous','p_under_local_cv','actual_under','selected_parent_lambda']].to_csv(out/'oos_predictions.csv',index=False)
    shape={'parentlike_continuous':moments(o.p_under_parent_continuous),'v04':moments(o.p_under_local_cv),'parent_expected_total':moments(o.parent_expected_total),'joined_n':len(m),'evaluated_n':len(o),'full_v04_n':fulln,'features':features,'price_used':False,'participant_identity_used':False}
    (out/'shape.json').write_text(json.dumps(shape,indent=2)); print(pd.DataFrame(comp).to_string(index=False)); print(pd.DataFrame(annual).to_string(index=False)); print(json.dumps(shape,indent=2))
if __name__=='__main__': main()

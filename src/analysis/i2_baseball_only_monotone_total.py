#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from src.analysis.i2_baseball_only_continuous_total import add_baseball_total, norm, score_cols, logloss, brier

EPS=1e-9

def fit_linear(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    xm=float(np.mean(x)); xs=float(np.std(x)) or 1.0; z=(x-xm)/xs
    def obj(b):
        eta=b[0]+b[1]*z; p=1/(1+np.exp(-np.clip(eta,-40,40)))
        return -np.sum(y*np.log(np.clip(p,EPS,1-EPS))+(1-y)*np.log(np.clip(1-p,EPS,1-EPS)))
    ybar=np.clip(np.mean(y),EPS,1-EPS); init=[np.log(ybar/(1-ybar)),-0.1]
    r=minimize(obj,init,method='L-BFGS-B',bounds=[(None,None),(None,0.0)])
    return (float(r.x[0]),float(r.x[1]),xm,xs)

def pred_linear(model,x):
    a,b,xm,xs=model; z=(np.asarray(x,float)-xm)/xs; return 1/(1+np.exp(-np.clip(a+b*z,-40,40)))

def fit_iso(x,y):
    ir=IsotonicRegression(increasing=False,out_of_bounds='clip',y_min=0.02,y_max=0.98)
    ir.fit(np.asarray(x,float),np.asarray(y,float)); return ir

def fit_models(train):
    lin=fit_linear(train.baseball_expected_total,train.y)
    iso=fit_iso(train.baseball_expected_total,train.y)
    return lin,iso

def predict_mix(lin,iso,x,alpha):
    pl=pred_linear(lin,x); pi=np.asarray(iso.predict(np.asarray(x,float)),float)
    return (1-alpha)*pl+alpha*pi

def build_common(phase_root,v04_root):
    phase=Path(phase_root)
    d=pd.read_csv(next(phase.rglob('MLB_Game_Stats_Joined_2021_2025.csv.gz')),low_memory=False)
    if 'benchmark_matched' in d.columns: d=d[d.benchmark_matched==True].copy()
    d['game_date']=pd.to_datetime(d.game_date).dt.normalize(); sa,sh=score_cols(d)
    d['away_score_num']=pd.to_numeric(d[sa]); d['home_score_num']=pd.to_numeric(d[sh]); d['away_team_code']=d.away_team_code.map(norm); d['home_team_code']=d.home_team_code.map(norm); d=add_baseball_total(d)
    games=pd.read_parquet(next(phase.rglob('games.parquet'))); games['game_date']=pd.to_datetime(games.game_date).dt.normalize(); games['away_team_code']=games.away_team.map(norm); games['home_team_code']=games.home_team.map(norm)
    games=games.sort_values(['game_date','away_team_code','home_team_code','game_id']); games['_seq']=games.groupby(['game_date','away_team_code','home_team_code']).cumcount()
    d=d.sort_values(['game_date','away_team_code','home_team_code','retro_game_id']); d['_seq']=d.groupby(['game_date','away_team_code','home_team_code']).cumcount(); d=d.merge(games[['game_id','game_date','away_team_code','home_team_code','_seq']],on=['game_date','away_team_code','home_team_code','_seq'],how='left')
    v=pd.read_csv(next(Path(v04_root).rglob('v04_oos_predictions.csv')))
    m=v.merge(d[d.game_id.notna()].drop_duplicates('game_id')[['game_id','baseball_expected_total']],on='game_id',how='left',validate='one_to_one')
    jr=float(m.baseball_expected_total.notna().mean()); m=m[m.baseball_expected_total.notna()].copy(); m['y']=m.actual_under.astype(float)
    return m,jr,len(v)

def moments(s):
    s=pd.Series(s).dropna(); return {'mean':float(s.mean()),'sd':float(s.std()),'skew':float(s.skew()),'excess_kurtosis':float(s.kurt()),'p01':float(s.quantile(.01)),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase1',required=True); ap.add_argument('--v04',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    m,jr,fulln=build_common(a.phase1,a.v04); print('join_rate',jr,'common_n',len(m),'full_n',fulln)
    if jr<.98: raise SystemExit('join below 98%')
    alphas=[0.0,0.25,0.5,0.75,1.0]; rows=[]; annual=[]; cvrows=[]
    for season in [2023,2024,2025]:
        tr=m[m.season<season].copy(); te=m[m.season==season].copy(); prior=sorted(tr.season.unique())
        scores=[]
        for alpha in alphas:
            vals=[]
            for vs in prior[1:]:
                tt=tr[tr.season<vs]; vv=tr[tr.season==vs]
                if tt.empty or vv.empty: continue
                lin,iso=fit_models(tt); pp=predict_mix(lin,iso,vv.baseball_expected_total,alpha); vals.append(logloss(vv.y,pp)); cvrows.append({'target_season':season,'validation_season':vs,'alpha':alpha,'logloss':vals[-1]})
            scores.append((np.mean(vals) if vals else (999+alpha),alpha))
        _,alpha=min(scores); lin,iso=fit_models(tr); p=predict_mix(lin,iso,te.baseball_expected_total,alpha); q=te.copy(); q['p_under_monotone']=p; q['selected_alpha_iso']=alpha; rows.append(q)
        annual.append({'season':season,'n':len(q),'alpha_iso':alpha,'logloss':logloss(q.y,p),'brier':brier(q.y,p),'mean_p':float(np.mean(p)),'actual_under':float(q.y.mean())})
    o=pd.concat(rows,ignore_index=True); comp=[]
    for split,x in [('ALL',o),('DEV',o[o.season<=2024]),('2025',o[o.season==2025])]:
        for name,col in [('BASEBALL_MONOTONE','p_under_monotone'),('V04_LOCAL_CV','p_under_local_cv')]:
            comp.append({'split':split,'model':name,'n':len(x),'logloss':logloss(x.y,x[col]),'brier':brier(x.y,x[col]),'mean_p_under':float(x[col].mean()),'actual_under':float(x.y.mean())})
    pd.DataFrame(comp).to_csv(out/'performance_comparison.csv',index=False); pd.DataFrame(annual).to_csv(out/'annual_metrics.csv',index=False); pd.DataFrame(cvrows).to_csv(out/'cv_metrics.csv',index=False); o[['game_id','season','baseball_expected_total','p_under_monotone','p_under_local_cv','actual_under','selected_alpha_iso']].to_csv(out/'oos_predictions.csv',index=False)
    shape={'baseball_monotone':moments(o.p_under_monotone),'v04':moments(o.p_under_local_cv),'baseball_expected_total':moments(o.baseball_expected_total),'join_rate':jr,'common_n':len(m),'evaluated_n':len(o),'full_v04_n':fulln,'price_used':False}; (out/'shape.json').write_text(json.dumps(shape,indent=2)); print(pd.DataFrame(comp).to_string(index=False)); print(pd.DataFrame(annual).to_string(index=False)); print(json.dumps(shape,indent=2))
if __name__=='__main__': main()

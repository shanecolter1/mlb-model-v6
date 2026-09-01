#!/usr/bin/env python3
"""Translate validated probabilistic matchup event signal into inning run residuals.

The raw 11-feature M4 run challenger failed, while the proximal state-to-PA-event
gate confirmed all four M1 event dimensions out of sample. This test therefore
compresses the matchup state through those validated event models before asking
it to move M0.

For each half inning and each event (K, BB/HBP, HR, non-HR hit), we compute the
matchup-only PA logit residual delta from the event model. Top and bottom deltas
are summed symmetrically. These four deltas are the only run-residual features:

  logit P(any run in inning) = logit M0(total,inning) + gamma' delta_event

Event-model ridge strengths are read from the prior event-fidelity result. 2023
event deltas are produced by event models fit on 2022; 2024 deltas are produced
by event models fit on 2022-2023. Run-model ridge is selected with chronological
monthly expanding folds inside 2023, then refit on all 2023 and confirmed once
on 2024. 2025 remains untouched.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS={
 'k':({'strikeout'},['expected_batter_k','expected_pitcher_k','expected_interaction_k']),
 'baserunner':({'walk','hit_by_pitch'},['expected_batter_baserunner','expected_pitcher_baserunner','expected_interaction_baserunner']),
 'hr':({'home_run'},['expected_batter_hr','expected_pitcher_hr','expected_interaction_hr']),
 'nonhr_hit':({'single','double','triple'},['expected_batter_nonhr_hit','expected_pitcher_nonhr_hit']),
}
RUN_RIDGES=[0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0]
EPS=1e-8

def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def sigmoid(z): return 1/(1+np.exp(-np.clip(np.asarray(z,float),-35,35)))
def ll(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def br(y,p): return float(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))
def fit_ridge(X,y,offset,ridge):
    b=np.zeros(X.shape[1]); eye=np.eye(X.shape[1])
    for _ in range(60):
        p=sigmoid(offset+X@b); w=np.maximum(p*(1-p),1e-7)
        g=X.T@(y-p)-ridge*b; H=(X.T*w)@X+ridge*eye+1e-9*eye
        step=np.linalg.solve(H,g); b2=b+step
        if np.max(np.abs(step))<1e-8: b=b2; break
        b=b2
    return b

def pa_merge(pa,hf):
    pa=pa.copy(); pa['game_date']=pd.to_datetime(pa.game_date,errors='coerce').dt.normalize(); pa['season']=pa.game_date.dt.year.astype(int); pa['inning']=pd.to_numeric(pa.inning).astype(int)
    pa=pa[pa.season.isin([2022,2023,2024])&pa.inning.between(1,9)].copy()
    halfcol='half_inning' if 'half_inning' in pa.columns else 'half'; pa['half']=pa[halfcol].astype(str).str.lower()
    ev=pa.event.astype(str)
    for e,(vals,_) in EVENTS.items(): pa['y_'+e]=ev.isin(vals).astype('int8')
    keys=['game_id','season','inning','half','batting_team_id','pitching_team_id']
    fcols=sorted({f for _,fs in EVENTS.values() for f in fs})
    h=hf[keys+['game_date','dk_total_open_total','runs_half']+fcols].copy(); h['season']=pd.to_numeric(h.season).astype(int); h['inning']=pd.to_numeric(h.inning).astype(int); h['half']=h.half.astype(str).str.lower(); h['game_date']=pd.to_datetime(h.game_date).dt.normalize()
    x=pa.merge(h[keys+fcols],on=keys,how='inner',validate='many_to_one')
    return x,h

def fit_event_model(train,event,ridge):
    _,fs=EVENTS[event]; ycol='y_'+event
    z=train.dropna(subset=fs+[ycol]).copy()
    mu=z[fs].mean().to_numpy(float).copy(); sd=z[fs].std(ddof=0).to_numpy(float).copy(); sd[sd<1e-9]=1
    X=(z[fs].to_numpy(float)-mu)/sd; y=z[ycol].to_numpy(float)
    rates=z.groupby('inning')[ycol].mean().to_dict(); overall=float(z[ycol].mean())
    p0=np.array([rates.get(int(i),overall) for i in z.inning],float)
    beta=fit_ridge(X,y,logit(p0),ridge)
    return {'features':fs,'mu':mu,'sd':sd,'beta':beta,'rates':rates,'overall':overall}

def event_delta_for_halves(h,model,event):
    fs=model['features']; z=h.dropna(subset=fs).copy(); X=(z[fs].to_numpy(float)-model['mu'])/model['sd']
    z['delta_'+event]=X@model['beta']
    return z[['game_id','season','inning','half','delta_'+event]]

def make_full_event_delta(pa,h,event_ridges):
    parts=[]
    for target_year,train_years in [(2023,[2022]),(2024,[2022,2023])]:
        hp=h[h.season==target_year].copy(); deltas=None
        for event in EVENTS:
            model=fit_event_model(pa[pa.season.isin(train_years)],event,float(event_ridges[event]))
            d=event_delta_for_halves(hp,model,event)
            deltas=d if deltas is None else deltas.merge(d,on=['game_id','season','inning','half'],how='inner',validate='one_to_one')
        basecols=['game_id','game_date','season','inning','half','dk_total_open_total','runs_half']
        q=hp[basecols].merge(deltas,on=['game_id','season','inning','half'],how='inner',validate='one_to_one')
        top=q[q.half=='top']; bot=q[q.half=='bottom']
        keys=['game_id','game_date','season','inning','dk_total_open_total']
        x=top.merge(bot,on=keys,suffixes=('_top','_bottom'),how='inner',validate='one_to_one')
        x['full_inning_runs']=x.runs_half_top+x.runs_half_bottom; x['any_run']=(x.full_inning_runs>=1).astype(int)
        for e in EVENTS: x['delta_'+e]=x['delta_'+e+'_top']+x['delta_'+e+'_bottom']
        parts.append(x[keys+['full_inning_runs','any_run']+['delta_'+e for e in EVENTS]])
    return pd.concat(parts,ignore_index=True)

def attach_m0(full,all_half):
    out=[]
    for year in [2023,2024]:
        hist=all_half[(all_half.season<year)&all_half.half_played.astype(bool)].copy()
        p=hist.pivot(index=['game_id','inning','dk_total_open_total'],columns='half',values='runs_half').reset_index().dropna(subset=['top','bottom'])
        p['any_run']=((p.top+p.bottom)>=1).astype(int)
        b=p.groupby(['dk_total_open_total','inning'],as_index=False).any_run.mean().rename(columns={'any_run':'m0_p_any'})
        z=full[full.season==year].merge(b,on=['dk_total_open_total','inning'],how='left',validate='many_to_one'); out.append(z)
    return pd.concat(out,ignore_index=True)

def prep_run(train,test,features):
    tr=train.dropna(subset=features+['m0_p_any']).copy(); te=test.dropna(subset=features+['m0_p_any']).copy()
    mu=tr[features].mean().to_numpy(float).copy(); sd=tr[features].std(ddof=0).to_numpy(float).copy(); sd[sd<1e-9]=1
    return tr,te,(tr[features].to_numpy(float)-mu)/sd,(te[features].to_numpy(float)-mu)/sd,mu,sd

def eval_run(train,test,features,ridge):
    tr,te,Xtr,Xte,mu,sd=prep_run(train,test,features); ytr=tr.any_run.to_numpy(float); yte=te.any_run.to_numpy(float)
    beta=fit_ridge(Xtr,ytr,logit(tr.m0_p_any),ridge); p0=np.asarray(te.m0_p_any,float); p=sigmoid(logit(p0)+Xte@beta)
    return {'n_train':len(tr),'n_test':len(te),'m0_logloss':ll(yte,p0),'model_logloss':ll(yte,p),'logloss_improvement':ll(yte,p0)-ll(yte,p),
            'm0_brier':br(yte,p0),'model_brier':br(yte,p),'brier_improvement':br(yte,p0)-br(yte,p),'beta':beta}

def select_ridge_2023(x,features):
    z=x[x.season==2023].copy(); z['month']=pd.to_datetime(z.game_date).dt.month
    months=sorted(int(m) for m in z.month.unique()); rows=[]
    # Calendar-month expanding validation; first observed month supplies foundation.
    for ridge in RUN_RIDGES:
        vals=[]
        for m in months[1:]:
            tr=z[z.month<m]; te=z[z.month==m]
            if len(tr)<500 or len(te)<100: continue
            r=eval_run(tr,te,features,ridge); vals.append(r)
            rows.append({'ridge':ridge,'test_month':m,'n_train':r['n_train'],'n_test':r['n_test'],'logloss_improvement':r['logloss_improvement'],'brier_improvement':r['brier_improvement']})
    f=pd.DataFrame(rows)
    s=f.groupby('ridge',as_index=False).agg(mean_logloss_improvement=('logloss_improvement','mean'),worst_month_logloss_improvement=('logloss_improvement','min'),mean_brier_improvement=('brier_improvement','mean'))
    best=s.sort_values(['mean_logloss_improvement','worst_month_logloss_improvement'],ascending=False).iloc[0]
    return f,s,float(best.ridge)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plate-appearances',type=Path,required=True); ap.add_argument('--half-features',type=Path,required=True); ap.add_argument('--all-half-matrix',type=Path,required=True); ap.add_argument('--event-confirmation',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa0=pd.read_parquet(a.plate_appearances); hf=pd.read_parquet(a.half_features); ah=pd.read_parquet(a.all_half_matrix); ec=pd.read_csv(a.event_confirmation)
    if (pd.to_datetime(pa0.game_date).dt.year>=2025).any() or (pd.to_numeric(hf.season)>=2025).any() or (pd.to_numeric(ah.season)>=2025).any(): raise RuntimeError('2025 leakage')
    pa,h=pa_merge(pa0,hf); event_ridges={r.event:float(r.selected_ridge_2023) for r in ec.itertuples()}
    full=make_full_event_delta(pa,h,event_ridges); full=attach_m0(full,ah)
    features=['delta_'+e for e in EVENTS]
    folds,summary,ridge=select_ridge_2023(full,features)
    confirm=eval_run(full[full.season==2023],full[full.season==2024],features,ridge)
    coeff=pd.DataFrame({'feature':features,'standardized_coefficient_2023_fit':confirm['beta']})
    folds.to_csv(a.output_dir/'m4_event_to_run_monthly_ridge_folds_2023.csv',index=False); summary.to_csv(a.output_dir/'m4_event_to_run_ridge_summary_2023.csv',index=False); coeff.to_csv(a.output_dir/'m4_event_to_run_coefficients.csv',index=False); full.to_parquet(a.output_dir/'m4_event_residual_full_inning_matrix.parquet',index=False)
    manifest={'status':'PASS','architecture':'M4_event_residual_translation_to_M0_any_run','development_seasons_loaded':[2022,2023,2024],
              'event_model_ridges_source':'2023 selections from M4 state-to-event fidelity','event_model_ridges':event_ridges,
              'event_delta_construction':'PA-event logit matchup residual only; top+bottom symmetric sum','run_features':features,
              'run_ridge_selection':'2023 chronological calendar-month expanding validation','selected_run_ridge':ridge,
              'confirmation_year':2024,'holdout_season':2025,'holdout_opened':False,'market_data_used':False,
              'only_market_context':'isolated full-game opening total in M0 probability','full_inning_rows':int(len(full)),
              'confirmation_2024':{k:(float(v) if isinstance(v,(np.floating,float)) else int(v) if isinstance(v,(np.integer,int)) else v) for k,v in confirm.items() if k!='beta'},
              'confirm_logloss_positive':bool(confirm['logloss_improvement']>0),'confirm_brier_positive':bool(confirm['brier_improvement']>0),
              'automatic_production_promotion':False,'note':'Tests event-to-run translation after matchup-state fidelity was confirmed. Positive confirmation supports moving toward discrete 0/1/2/3/4+ modeling; negative confirmation requires richer event/count/run-state translation.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

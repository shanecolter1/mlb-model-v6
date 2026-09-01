#!/usr/bin/env python3
"""Validate whether the integrated probabilistic matchup state preserves M1 event signal.

The direct M4 run-residual challengers failed overall. Before changing the state
models, this audit asks the more proximal question: after replacing realized
pitcher/batter identities with the pregame joint state mixture, do the assembled
matchup features still predict the same PA event dimensions validated in M1?

For K, BB/HBP baserunner, HR, and non-HR hit outcomes, PA rows inherit their
half-inning's pregame expected matchup features. A prior-training inning-specific
league event rate is the offset baseline. Ridge is selected on 2023 using only
2022 training, then frozen and re-fit on 2022-2023 for 2024 confirmation.

This is matchup-state fidelity, not a run model. No 2025 data, inning markets,
or new feature assumptions are used.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

RIDGES=[0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0]
EVENTS={
 'k':{'strikeout'},
 'baserunner':{'walk','hit_by_pitch'},
 'hr':{'home_run'},
 'nonhr_hit':{'single','double','triple'},
}
FEATURES={
 'k':['expected_batter_k','expected_pitcher_k','expected_interaction_k'],
 'baserunner':['expected_batter_baserunner','expected_pitcher_baserunner','expected_interaction_baserunner'],
 'hr':['expected_batter_hr','expected_pitcher_hr','expected_interaction_hr'],
 'nonhr_hit':['expected_batter_nonhr_hit','expected_pitcher_nonhr_hit'],
}
EPS=1e-8


def logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def sigmoid(z):
    return 1/(1+np.exp(-np.clip(np.asarray(z,float),-35,35)))
def ll(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def br(y,p): return float(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))

def fit(X,y,offset,ridge):
    b=np.zeros(X.shape[1]); eye=np.eye(X.shape[1])
    for _ in range(60):
        p=sigmoid(offset+X@b); w=np.maximum(p*(1-p),1e-7)
        g=X.T@(y-p)-ridge*b; H=(X.T*w)@X+ridge*eye+1e-9*eye
        step=np.linalg.solve(H,g); b2=b+step
        if np.max(np.abs(step))<1e-8: b=b2; break
        b=b2
    return b

def baseline_rates(train,ycol):
    g=train.groupby('inning')[ycol].mean().to_dict()
    overall=float(train[ycol].mean())
    return g,overall

def evaluate(train,test,event,ridge):
    fs=FEATURES[event]; ycol='y_'+event
    tr=train.dropna(subset=fs+[ycol]).copy(); te=test.dropna(subset=fs+[ycol]).copy()
    mu=tr[fs].mean().to_numpy(float).copy(); sd=tr[fs].std(ddof=0).to_numpy(float).copy(); sd[sd<1e-9]=1
    Xtr=(tr[fs].to_numpy(float)-mu)/sd; Xte=(te[fs].to_numpy(float)-mu)/sd
    rates,overall=baseline_rates(tr,ycol)
    ptr=np.array([rates.get(int(i),overall) for i in tr.inning],float)
    pte=np.array([rates.get(int(i),overall) for i in te.inning],float)
    ytr=tr[ycol].to_numpy(float); yte=te[ycol].to_numpy(float)
    beta=fit(Xtr,ytr,logit(ptr),ridge); p=sigmoid(logit(pte)+Xte@beta)
    return {'n_train':len(tr),'n_test':len(te),'baseline_logloss':ll(yte,pte),'model_logloss':ll(yte,p),
            'logloss_improvement':ll(yte,pte)-ll(yte,p),'baseline_brier':br(yte,pte),'model_brier':br(yte,p),
            'brier_improvement':br(yte,pte)-br(yte,p),'beta':beta,'y':yte,'p0':pte,'p':p}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plate-appearances',type=Path,required=True); ap.add_argument('--half-features',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa=pd.read_parquet(a.plate_appearances); hf=pd.read_parquet(a.half_features)
    pa['game_date']=pd.to_datetime(pa.game_date,errors='coerce').dt.normalize(); pa['season']=pa.game_date.dt.year.astype(int); pa['inning']=pd.to_numeric(pa.inning).astype(int)
    pa=pa[pa.inning.between(1,9)&pa.season.isin([2022,2023,2024])].copy()
    if (pa.season>=2025).any() or (pd.to_numeric(hf.season)>=2025).any(): raise RuntimeError('2025 leakage')
    ev=pa.event.astype(str)
    for e,vals in EVENTS.items(): pa['y_'+e]=ev.isin(vals).astype('int8')
    # Normalize half label to the same top/bottom convention used by integrated features.
    halfcol='half_inning' if 'half_inning' in pa.columns else 'half'
    pa['half']=pa[halfcol].astype(str).str.lower().replace({'top':'top','bottom':'bottom'})
    keys=['game_id','season','inning','half','batting_team_id','pitching_team_id']
    fcols=sorted({c for v in FEATURES.values() for c in v})
    h=hf[keys+fcols].copy(); h['season']=pd.to_numeric(h.season).astype(int); h['inning']=pd.to_numeric(h.inning).astype(int); h['half']=h.half.astype(str).str.lower()
    x=pa.merge(h,on=keys,how='inner',validate='many_to_one')

    select=[]; confirm=[]; coeff=[]; agg={}
    for event in EVENTS:
        cand=[]
        tr22=x[x.season==2022]; te23=x[x.season==2023]
        for ridge in RIDGES:
            r=evaluate(tr22,te23,event,ridge)
            cand.append({'event':event,'selection_year':2023,'ridge':ridge,'n_train':r['n_train'],'n_test':r['n_test'],'logloss_improvement':r['logloss_improvement'],'brier_improvement':r['brier_improvement']})
        c=pd.DataFrame(cand); select.extend(c.to_dict('records'))
        best=c.sort_values(['logloss_improvement','brier_improvement'],ascending=False).iloc[0]
        ridge=float(best.ridge)
        r=evaluate(x[x.season<=2023],x[x.season==2024],event,ridge)
        confirm.append({'event':event,'selected_ridge_2023':ridge,'n_train_2024':r['n_train'],'n_test_2024':r['n_test'],
                        'baseline_logloss_2024':r['baseline_logloss'],'model_logloss_2024':r['model_logloss'],'logloss_improvement_2024':r['logloss_improvement'],
                        'baseline_brier_2024':r['baseline_brier'],'model_brier_2024':r['model_brier'],'brier_improvement_2024':r['brier_improvement'],
                        'confirm_logloss_positive':r['logloss_improvement']>0,'confirm_brier_positive':r['brier_improvement']>0})
        for f,b in zip(FEATURES[event],r['beta']): coeff.append({'event':event,'selected_ridge':ridge,'feature':f,'standardized_coefficient_2022_2023_fit':float(b)})
        agg[event]={'logloss_improvement_2024':r['logloss_improvement'],'brier_improvement_2024':r['brier_improvement']}
    pd.DataFrame(select).to_csv(a.output_dir/'m4_event_fidelity_ridge_selection_2023.csv',index=False)
    con=pd.DataFrame(confirm); con.to_csv(a.output_dir/'m4_event_fidelity_confirmation_2024.csv',index=False)
    pd.DataFrame(coeff).to_csv(a.output_dir/'m4_event_fidelity_coefficients.csv',index=False)
    manifest={'status':'PASS','architecture':'M4_probabilistic_matchup_state_to_PA_event_fidelity','development_seasons_loaded':[2022,2023,2024],
              'selection_year':2023,'confirmation_year':2024,'holdout_season':2025,'holdout_opened':False,'events':list(EVENTS),
              'features_by_event':FEATURES,'baseline':'training-fold inning-specific league event rate','market_data_used':False,
              'pa_rows_matched':int(len(x)),'confirmation':con.to_dict('records'),'all_events_logloss_positive_2024':bool(con.confirm_logloss_positive.all()),
              'all_events_brier_positive_2024':bool(con.confirm_brier_positive.all()),'automatic_production_promotion':False,
              'note':'Proximal fidelity gate. If event signal survives but run residual does not, the next research target is event-to-run translation rather than pitcher/batter state identification.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

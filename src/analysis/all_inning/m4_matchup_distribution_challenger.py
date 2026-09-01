#!/usr/bin/env python3
"""Compare full latent matchup distribution against collapsed point-vector M4.

Development-only (2021-2024 source; 2023 selection, 2024 confirmation). 2025 is
never loaded.  The same M2/M3 joint slot x starter/bullpen probabilities are used
for both branches.  The point branch transforms the probability-weighted mean
matchup vector.  The distribution branch transforms every latent matchup state
first and mixes probabilities afterwards, preserving Jensen/nonlinear effects.

Bullpen uncertainty is retained at reliever-identity level using the already
selected inning-specific usage-window/alpha specification; candidate reliever
weights are not collapsed before the nonlinear event transform.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analysis.all_inning import m4_integrated_matchup_residual_validation as base

EVENTS = base.EVENTS
METRICS = base.METRICS
INTER = base.INTERACTION_METRICS
RIDGES = [0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0,100.0,200.0,500.0,1000.0]
EPS=1e-10


def sigmoid(x):
    x=np.clip(x,-30,30); return 1/(1+np.exp(-x))
def logit(p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8); return np.log(p/(1-p))
def ll(y,p):
    p=np.clip(p,EPS,1-EPS); return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def br(y,p): return float(np.mean((y-p)**2))

def fit_ridge(X,y,offset,ridge):
    b=np.zeros(X.shape[1])
    for _ in range(40):
        p=sigmoid(offset+X@b); w=np.maximum(p*(1-p),1e-6)
        H=X.T@(w[:,None]*X)+ridge*np.eye(X.shape[1]); g=X.T@(y-p)-ridge*b
        try: step=np.linalg.solve(H,g)
        except np.linalg.LinAlgError: step=np.linalg.pinv(H)@g
        b += step
        if np.max(np.abs(step))<1e-8: break
    return b

def actual_pa_features(pa):
    bidx=base.make_rate_index(base.daily_player_rates(pa,'batter_id'),'batter_id')
    pidx=base.make_rate_index(base.daily_player_rates(pa,'pitcher_id'),'pitcher_id')
    rows=[]
    for r in pa[pa.season.isin([2022,2023,2024])].itertuples():
        d=pd.Timestamp(r.game_date); bv=base.rates_asof(bidx,r.batter_id,d); pv=base.rates_asof(pidx,r.pitcher_id,d)
        if bv is None or pv is None: continue
        z={'game_id':r.game_id,'batting_team_id':r.batting_team_id,'pitching_team_id':r.pitching_team_id,
           'inning':int(r.inning),'season':int(r.season),'event':str(r.event)}
        for k,m in enumerate(METRICS):
            z[f'b_{m}']=bv[k]; z[f'p_{m}']=pv[k]
            if m in INTER: z[f'x_{m}']=bv[k]*pv[k]
        rows.append(z)
    return pd.DataFrame(rows), bidx, pidx

def event_cols(m):
    return [f'b_{m}',f'p_{m}']+([f'x_{m}'] if m in INTER else [])

def fit_event_models(ap):
    out={}
    for m,vals in EVENTS.items():
        cols=event_cols(m); y=ap.event.isin(vals).astype(float).to_numpy()
        tr=ap[ap.season==2022].copy(); sel=ap[ap.season==2023].copy()
        ytr=tr.event.isin(vals).astype(float).to_numpy(); ysel=sel.event.isin(vals).astype(float).to_numpy()
        mu=tr[cols].mean().to_numpy(float); sd=tr[cols].std(ddof=0).to_numpy(float); sd[sd<1e-9]=1
        Xtr=(tr[cols].to_numpy(float)-mu)/sd; Xsel=(sel[cols].to_numpy(float)-mu)/sd
        p0=float(ytr.mean()); off=np.full(len(tr),logit(p0)); osel=np.full(len(sel),logit(p0))
        cand=[]
        for r in RIDGES:
            b=fit_ridge(Xtr,ytr,off,r); p=sigmoid(osel+Xsel@b)
            cand.append((ll(ysel,p),br(ysel,p),r,b))
        _,_,ridge,b=min(cand,key=lambda q:(q[0],q[1]))
        # refit 2022+2023 with selected ridge; standardization refit strictly before confirmation 2024
        tr2=ap[ap.season.isin([2022,2023])].copy(); y2=tr2.event.isin(vals).astype(float).to_numpy()
        mu=tr2[cols].mean().to_numpy(float); sd=tr2[cols].std(ddof=0).to_numpy(float); sd[sd<1e-9]=1
        X2=(tr2[cols].to_numpy(float)-mu)/sd; p0=float(y2.mean()); b=fit_ridge(X2,y2,np.full(len(tr2),logit(p0)),ridge)
        out[m]={'cols':cols,'mu':mu,'sd':sd,'p0':p0,'beta':b,'ridge':ridge}
    return out

def bullpen_candidates(team,date,inning,window,alpha,bf_groups,fi_groups,pidx):
    hb=bf_groups.get(team)
    if hb is None: return []
    lo=date-pd.Timedelta(days=int(window)); h=hb[(hb.game_date>=lo)&(hb.game_date<date)]
    if h.empty: return []
    g=h.groupby('pitcher_id',as_index=False).relief_pa.sum().rename(columns={'relief_pa':'global_bf'})
    hf=fi_groups.get(team)
    if hf is not None:
        q=hf[(hf.game_date>=lo)&(hf.game_date<date)&(hf.inning==inning)]
    else: q=None
    if q is not None and len(q):
        q=q.groupby('pitcher_id',as_index=False).first_count.sum().rename(columns={'first_count':'inning_first'})
        g=g.merge(q,on='pitcher_id',how='left')
    else: g['inning_first']=0.0
    g['inning_first']=g.inning_first.fillna(0.0)
    glob=g.global_bf.to_numpy(float); glob=glob/glob.sum()
    inn=g.inning_first.to_numpy(float); inn=inn/inn.sum() if inn.sum()>0 else glob
    w=(1-alpha)*glob+alpha*inn
    out=[]
    for ww,pid in zip(w,g.pitcher_id):
        v=base.rates_asof(pidx,pid,date)
        if ww>0 and v is not None and np.isfinite(v).all(): out.append((float(ww),v))
    s=sum(x[0] for x in out)
    return [(w/s,v) for w,v in out] if s>0 else []

def transform_state(bv,pv,models):
    ans={}
    for m,mo in models.items():
        raw=[bv[METRICS.index(m)],pv[METRICS.index(m)]]
        if m in INTER: raw.append(raw[0]*raw[1])
        X=(np.asarray(raw)-mo['mu'])/mo['sd']
        ans[m]=float(sigmoid(logit(mo['p0'])+X@mo['beta']))
    return ans

def build_predictions(half,joint,pa,starters,bpbest,bidx,pidx,models):
    h=half[half.half_played.astype(bool)&half.season.isin([2024])].copy()
    jcols=[c for c in joint.columns if c.startswith('p_slot') and (c.endswith('_starter') or c.endswith('_bullpen'))]
    keys=['game_id','season','inning','batting_team_id','pitching_team_id']
    h=h.merge(joint[keys+jcols],on=keys,how='left',validate='one_to_one')
    line=base.lineup_identity(pa); exposures=base.exposure_tables(pa); smap=base.starter_map(starters)
    lineup_skill={(r.game_id,r.batting_team_id,int(r.batting_order_slot)):base.rates_asof(bidx,r.batter_id,r.game_date) for r in line.itertuples()}
    bf,fi=base.bullpen_history(pa,starters); bfg={k:g.copy() for k,g in bf.groupby('pitching_team_id',sort=False)}; fig={k:g.copy() for k,g in fi.groupby('pitching_team_id',sort=False)}
    specs={int(r.inning):(int(r.window),float(r.alpha_inning_usage)) for r in bpbest.itertuples()}
    rows=[]
    for r in h.itertuples():
        inn=int(r.inning); d=pd.Timestamp(r.game_date); sid=smap.get((r.game_id,r.pitching_team_id)); sv=base.rates_asof(pidx,sid,d)
        bwin,balpha=specs.get(inn,specs.get(2,(30,.25))); bpc=bullpen_candidates(r.pitching_team_id,d,inn,bwin,balpha,bfg,fig,pidx)
        states=[]
        for start in range(1,10):
            ex=exposures[(2024,inn,start)]; bs=[]; bw=[]
            for slot in range(1,10):
                v=lineup_skill.get((r.game_id,r.batting_team_id,slot))
                if v is not None and np.isfinite(v).all() and ex[slot-1]>0: bs.append(v); bw.append(ex[slot-1])
            if not bs: continue
            bw=np.asarray(bw,float); bw/=bw.sum(); bv=(np.vstack(bs)*bw[:,None]).sum(axis=0)
            ps=float(getattr(r,f'p_slot{start}_starter') or 0); pb=float(getattr(r,f'p_slot{start}_bullpen') or 0)
            if ps>0 and sv is not None: states.append((ps,bv,sv))
            if pb>0:
                for rw,rv in bpc: states.append((pb*rw,bv,rv))
        mass=sum(s[0] for s in states)
        if mass<=0: continue
        states=[(w/mass,bv,pv) for w,bv,pv in states]
        mean_b=sum(w*bv for w,bv,pv in states); mean_p=sum(w*pv for w,bv,pv in states)
        point=transform_state(mean_b,mean_p,models)
        dist={m:0.0 for m in METRICS}
        for w,bv,pv in states:
            q=transform_state(bv,pv,models)
            for m in METRICS: dist[m]+=w*q[m]
        z={'game_id':r.game_id,'batting_team_id':r.batting_team_id,'pitching_team_id':r.pitching_team_id,'inning':inn,'season':2024,'state_count':len(states)}
        for m in METRICS: z[f'point_{m}']=point[m]; z[f'dist_{m}']=dist[m]
        rows.append(z)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();
    ap.add_argument('--plate-appearances',type=Path,required=True); ap.add_argument('--starters',type=Path,required=True); ap.add_argument('--half-matrix',type=Path,required=True); ap.add_argument('--joint-state',type=Path,required=True); ap.add_argument('--bullpen-best',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa=base.prep_pa(base.read(a.plate_appearances)); starters=base.read(a.starters); half=base.read(a.half_matrix); joint=base.read(a.joint_state); bp=base.read(a.bullpen_best)
    actual,bidx,pidx=actual_pa_features(pa); models=fit_event_models(actual); pred=build_predictions(half,joint,pa,starters,bp,bidx,pidx,models)
    # Join predictions to every realized 2024 PA in the matching half-inning.
    te=actual[actual.season==2024].merge(pred,on=['game_id','batting_team_id','pitching_team_id','inning','season'],how='inner')
    res=[]
    for m,vals in EVENTS.items():
        y=te.event.isin(vals).astype(float).to_numpy(); pp=te[f'point_{m}'].to_numpy(float); pdist=te[f'dist_{m}'].to_numpy(float)
        res.append({'event':m,'n_pa':len(y),'selected_ridge':models[m]['ridge'],'point_logloss':ll(y,pp),'distribution_logloss':ll(y,pdist),'distribution_minus_point_logloss_improvement':ll(y,pp)-ll(y,pdist),'point_brier':br(y,pp),'distribution_brier':br(y,pdist),'distribution_minus_point_brier_improvement':br(y,pp)-br(y,pdist)})
    out=pd.DataFrame(res); out.to_csv(a.output_dir/'m4_distribution_vs_point_2024.csv',index=False); pred.to_parquet(a.output_dir/'m4_distribution_half_predictions_2024.parquet',index=False)
    manifest={'status':'PASS','architecture':'M4_full_latent_matchup_distribution_vs_point_vector','development_source_seasons':[2021,2022,2023,2024],'selection_year':2023,'confirmation_year':2024,'holdout_season':2025,'holdout_opened':False,'distribution_definition':'transform each joint slot x starter/reliever-identity state then probability-mix; no pre-transform collapse','point_definition':'probability-weight matchup skill first, then nonlinear transform','market_data_used':False,'confirmation':out.to_dict('records'),'distribution_wins_all_events_logloss':bool((out.distribution_minus_point_logloss_improvement>0).all()),'distribution_wins_all_events_brier':bool((out.distribution_minus_point_brier_improvement>0).all()),'automatic_production_promotion':False}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.isotonic import IsotonicRegression

EPS=1e-9
RATE_KEYS={
    'k':'365d_ev_strikeout_rate_shrunk',
    'bb':'365d_ev_walk_rate_shrunk',
    'hr':'365d_ev_home_run_rate_shrunk',
    'hit':'365d_ev_hit_rate_shrunk',
    'xbh':'365d_ev_xbh_rate_shrunk',
    'onbase':'365d_ev_onbase_rate_shrunk',
}

def ll(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def br(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); return float(np.mean((y-p)**2))

def one(root,name):
    hits=list(Path(root).rglob(name))
    if len(hits)!=1: raise RuntimeError(f'{name}: expected exactly one, found {hits}')
    return hits[0]

def weighted_lineup(line):
    x=line.copy(); x['batting_order_slot']=pd.to_numeric(x.batting_order_slot,errors='coerce')
    x=x[x.batting_order_slot.between(1,9)].copy()
    raw=np.array([max(.75,1.08-.04*(i-1)) for i in range(1,10)],float); raw/=raw.sum()
    rows=[]
    for (gid,side),g in x.groupby(['game_id','team_side'],sort=False):
        rec={'game_id':gid,'team_side':side}
        for k,c in RATE_KEYS.items():
            vals=[]; ws=[]
            for _,r in g.iterrows():
                v=pd.to_numeric(r.get(c),errors='coerce'); s=int(r.batting_order_slot)
                if pd.notna(v): vals.append(float(v)); ws.append(raw[s-1])
            rec[f'lineup_{k}']=float(np.average(vals,weights=ws)) if vals else np.nan
        rec['lineup_coverage']=len(g)/9.0
        rows.append(rec)
    return pd.DataFrame(rows)

def starter_features(st):
    keep=['game_id','team_side']+[c for c in RATE_KEYS.values() if c in st.columns]
    z=st[keep].copy()
    return z.rename(columns={v:f'starter_{k}' for k,v in RATE_KEYS.items() if v in z.columns})

def team_features(gt):
    z=gt.copy(); rows=[]
    for _,r in z.iterrows():
        rec={'game_id':r.game_id,'team_side':r.side}
        for k,base in RATE_KEYS.items():
            bats=[c for c in z.columns if c.startswith(base) and 'batting' in c]
            pits=[c for c in z.columns if c.startswith(base) and 'pitching_allowed' in c]
            if not bats and base in z.columns: bats=[base]
            rec[f'team_bat_{k}']=pd.to_numeric(r[bats[0]],errors='coerce') if bats else np.nan
            rec[f'team_pitch_{k}']=pd.to_numeric(r[pits[0]],errors='coerce') if pits else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def build_side_table(root):
    gi=pd.read_parquet(one(root,'game_index.parquet')).copy(); gi['game_date']=pd.to_datetime(gi.game_date).dt.normalize(); gi['season']=gi.game_date.dt.year; gi['month']=gi.game_date.dt.month
    la=pd.read_parquet(one(root,'lineup_asof.parquet')); st=pd.read_parquet(one(root,'starter_asof.parquet')); gt=pd.read_parquet(one(root,'game_team_asof.parquet')); oc=pd.read_parquet(one(root,'inning_outcomes.parquet'))
    lf=weighted_lineup(la); sf=starter_features(st); tf=team_features(gt)
    res=oc.groupby('game_id',as_index=False).agg(total_runs=('total_runs','first'),home_runs=('home_runs','first'),away_runs=('away_runs','first'))
    sides=[]
    for side,opp in [('away','home'),('home','away')]:
        b=gi[['game_id','game_date','season','month','venue_id','venue_name']].copy(); b['team_side']=side; b['opp_side']=opp
        b=b.merge(lf[lf.team_side==side].drop(columns='team_side'),on='game_id',how='left')
        b=b.merge(tf[tf.team_side==side].drop(columns='team_side'),on='game_id',how='left')
        osf=sf[sf.team_side==opp].drop(columns='team_side').rename(columns={c:c.replace('starter_','opp_starter_') for c in sf.columns if c.startswith('starter_')})
        otf=tf[tf.team_side==opp].drop(columns='team_side').rename(columns={c:c.replace('team_pitch_','opp_team_pitch_') for c in tf.columns if c.startswith('team_pitch_')})
        b=b.merge(osf,on='game_id',how='left').merge(otf[['game_id']+[c for c in otf.columns if c.startswith('opp_team_pitch_')]],on='game_id',how='left').merge(res,on='game_id',how='left')
        b['runs']=b['away_runs'] if side=='away' else b['home_runs']; b['is_home']=1 if side=='home' else 0
        # Matchup blends echo the live engine's 50/50 batter-pitcher weighting while retaining bullpen/team context.
        for k in RATE_KEYS:
            a=b.get(f'lineup_{k}'); p=b.get(f'opp_starter_{k}'); tp=b.get(f'opp_team_pitch_{k}')
            if a is not None and p is not None: b[f'matchup_{k}']=.5*pd.to_numeric(a,errors='coerce')+.5*pd.to_numeric(p,errors='coerce')
            if a is not None and tp is not None: b[f'staff_matchup_{k}']=.5*pd.to_numeric(a,errors='coerce')+.5*pd.to_numeric(tp,errors='coerce')
        sides.append(b)
    return pd.concat(sides,ignore_index=True)

def run_model(train,test,alpha):
    drop={'game_id','game_date','runs','total_runs','home_runs','away_runs','team_side','opp_side','venue_name'}
    num=[c for c in train.columns if c not in drop and c!='venue_id' and pd.api.types.is_numeric_dtype(train[c])]
    cat=['venue_id','team_side']
    pre=ColumnTransformer([
      ('n',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),
      ('c',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat),
    ])
    pipe=Pipeline([('pre',pre),('m',PoissonRegressor(alpha=alpha,max_iter=1000))])
    pipe.fit(train[num+cat],train.runs)
    return np.clip(pipe.predict(test[num+cat]),.15,15.0),num

def fit_logistic(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); xm=x.mean(); xs=x.std() or 1.; z=(x-xm)/xs
    def obj(b):
        p=1/(1+np.exp(-np.clip(b[0]+b[1]*z,-40,40))); return -np.sum(y*np.log(np.clip(p,EPS,1-EPS))+(1-y)*np.log(np.clip(1-p,EPS,1-EPS)))
    yb=np.clip(y.mean(),EPS,1-EPS); r=minimize(obj,[np.log(yb/(1-yb)),-.2],method='L-BFGS-B',bounds=[(None,None),(None,0)])
    return r.x[0],r.x[1],xm,xs

def map_p(model,x):
    a,b,xm,xs=model; z=(np.asarray(x,float)-xm)/xs; return 1/(1+np.exp(-np.clip(a+b*z,-40,40)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase1',required=True); ap.add_argument('--v04',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    sides=build_side_table(a.phase1); v=pd.read_csv(one(a.v04,'v04_oos_predictions.csv')); v['y']=v.actual_under.astype(float)
    # Restrict to games with observed outcomes and usable historical feature coverage.
    sides=sides[sides.runs.notna()].copy(); alphas=[.01,.1,1.,10.]; pred=[]; tuning=[]
    for season in [2022,2023,2024,2025]:
        tr=sides[sides.season<season].copy(); te=sides[sides.season==season].copy()
        if tr.empty or te.empty: continue
        scores=[]
        for al in alphas:
            vals=[]
            for vs in sorted(tr.season.unique())[1:]:
                tt=tr[tr.season<vs]; vv=tr[tr.season==vs]
                if tt.empty or vv.empty: continue
                pp,_=run_model(tt,vv,al); vals.append(float(np.mean((vv.runs.to_numpy()-pp)**2)))
            scores.append((np.mean(vals) if vals else 999.,al))
        _,al=min(scores); pp,features=run_model(tr,te,al); q=te[['game_id','team_side','season']].copy(); q['expected_runs']=pp; q['alpha']=al; pred.append(q); tuning.append({'season':season,'alpha':al,'cv_mse':min(scores)[0],'feature_count':len(features)})
    sp=pd.concat(pred,ignore_index=True); gp=sp.groupby(['game_id','season'],as_index=False).agg(parent_expected_total=('expected_runs','sum'),parent_expected_away=('expected_runs','first'),selected_alpha=('alpha','first'))
    common=v.merge(gp,on=['game_id','season'],how='inner',validate='one_to_one'); join=len(common)/len(v); print('join_rate',join,'common',len(common),'of',len(v))
    if join<.90: raise SystemExit('parent-style join below 90%')
    # Chronological monotone mapping from the continuous baseball total to I2 Under.
    mapped=[]; annual=[]
    for season in [2022,2023,2024,2025]:
        tr=common[common.season<season]; te=common[common.season==season]
        if tr.empty or te.empty: continue
        lin=fit_logistic(tr.parent_expected_total,tr.y)
        iso=IsotonicRegression(increasing=False,out_of_bounds='clip',y_min=.02,y_max=.98).fit(tr.parent_expected_total,tr.y)
        # nested selection of light isotonic blend; default linear when history is thin
        cand=[0.,.25,.5]; scores=[]
        for mix in cand:
            vals=[]
            for vs in sorted(tr.season.unique())[1:]:
                tt=tr[tr.season<vs]; vv=tr[tr.season==vs]
                if tt.empty or vv.empty: continue
                l=fit_logistic(tt.parent_expected_total,tt.y); ii=IsotonicRegression(increasing=False,out_of_bounds='clip',y_min=.02,y_max=.98).fit(tt.parent_expected_total,tt.y)
                p=(1-mix)*map_p(l,vv.parent_expected_total)+mix*ii.predict(vv.parent_expected_total); vals.append(ll(vv.y,p))
            scores.append((np.mean(vals) if vals else 999+mix,mix))
        _,mix=min(scores); p=(1-mix)*map_p(lin,te.parent_expected_total)+mix*iso.predict(te.parent_expected_total)
        q=te.copy(); q['p_under_parent_continuous']=p; q['iso_mix']=mix; mapped.append(q)
        annual.append({'season':season,'n':len(q),'iso_mix':mix,'logloss':ll(q.y,p),'brier':br(q.y,p),'mean_p':float(np.mean(p)),'actual_under':float(q.y.mean())})
    o=pd.concat(mapped,ignore_index=True); comp=[]
    for split,x in [('ALL',o),('DEV',o[o.season<=2024]),('2025',o[o.season==2025])]:
        for name,col in [('PARENT_CONTINUOUS','p_under_parent_continuous'),('V04_LOCAL_CV','p_under_local_cv')]:
            comp.append({'split':split,'model':name,'n':len(x),'logloss':ll(x.y,x[col]),'brier':br(x.y,x[col]),'mean_p_under':float(x[col].mean()),'actual_under':float(x.y.mean())})
    def mom(s):
        s=pd.Series(s).dropna(); return {'mean':float(s.mean()),'sd':float(s.std()),'skew':float(s.skew()),'excess_kurtosis':float(s.kurt()),'p01':float(s.quantile(.01)),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99))}
    manifest={'experiment':'historical parent-style decimal expected-run total -> monotone I2 Under mapping','price_used':False,'market_total_used':False,'statistics_strictly_prior_date':True,'lineup_identity_timing':'retrospective_final_feed_unverified_pregame','starter_identity_timing':'retrospective_actual_first_pitcher_unverified_pregame','therefore_validation_class':'RETROSPECTIVE_ORACLE_PARENT_STYLE','join_rate':join,'shape_parent_expected_total':mom(o.parent_expected_total),'shape_parent_probability':mom(o.p_under_parent_continuous),'shape_v04':mom(o.p_under_local_cv)}
    pd.DataFrame(comp).to_csv(out/'performance_comparison.csv',index=False); pd.DataFrame(annual).to_csv(out/'annual_metrics.csv',index=False); pd.DataFrame(tuning).to_csv(out/'run_total_tuning.csv',index=False); o[['game_id','season','parent_expected_total','p_under_parent_continuous','p_under_local_cv','actual_under','iso_mix']].to_csv(out/'oos_predictions.csv',index=False); (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(comp).to_string(index=False)); print(pd.DataFrame(annual).to_string(index=False)); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

EPS=1e-9

def logloss(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def brier(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((y-p)**2))

def score_cols(df):
    for a,h in [('away_final_runs','home_final_runs'),('away_score','home_score'),('away_runs','home_runs')]:
        if a in df.columns and h in df.columns: return a,h
    raise ValueError('Final score columns not found')

def norm(x):
    m={'AZ':'AZ','ARI':'AZ','ARIZONA DIAMONDBACKS':'AZ','ATH':'ATH','OAK':'ATH','ATHLETICS':'ATH','OAKLAND ATHLETICS':'ATH','ATL':'ATL','ATLANTA BRAVES':'ATL','BAL':'BAL','BALTIMORE ORIOLES':'BAL','BOS':'BOS','BOSTON RED SOX':'BOS','CHC':'CHC','CHICAGO CUBS':'CHC','CWS':'CHW','CHW':'CHW','CHICAGO WHITE SOX':'CHW','CIN':'CIN','CINCINNATI REDS':'CIN','CLE':'CLE','CLEVELAND GUARDIANS':'CLE','CLEVELAND INDIANS':'CLE','COL':'COL','COLORADO ROCKIES':'COL','DET':'DET','DETROIT TIGERS':'DET','HOU':'HOU','HOUSTON ASTROS':'HOU','KC':'KC','KCR':'KC','KANSAS CITY ROYALS':'KC','LAA':'LAA','LOS ANGELES ANGELS':'LAA','LAD':'LAD','LOS ANGELES DODGERS':'LAD','MIA':'MIA','MIAMI MARLINS':'MIA','MIL':'MIL','MILWAUKEE BREWERS':'MIL','MIN':'MIN','MINNESOTA TWINS':'MIN','NYM':'NYM','NEW YORK METS':'NYM','NYY':'NYY','NEW YORK YANKEES':'NYY','PHI':'PHI','PHILADELPHIA PHILLIES':'PHI','PIT':'PIT','PITTSBURGH PIRATES':'PIT','SD':'SD','SDP':'SD','SAN DIEGO PADRES':'SD','SEA':'SEA','SEATTLE MARINERS':'SEA','SF':'SF','SFG':'SF','SAN FRANCISCO GIANTS':'SF','STL':'STL','ST. LOUIS CARDINALS':'STL','TB':'TB','TBR':'TB','TAMPA BAY RAYS':'TB','TEX':'TEX','TEXAS RANGERS':'TEX','TOR':'TOR','TORONTO BLUE JAYS':'TOR','WSH':'WSH','WSN':'WSH','WAS':'WSH','WASHINGTON NATIONALS':'WSH'}
    s=str(x).upper().strip(); return m.get(s,s)

def add_baseball_total(d,days=365,pseudo=30.0):
    d=d.sort_values(['game_date','away_team_code','home_team_code']).copy(); off={}; deff={}; league=[]; rows=[]
    for dt,g in d.groupby('game_date',sort=True):
        cut=pd.Timestamp(dt)-pd.Timedelta(days=days)
        for st in (off,deff):
            for t in list(st): st[t]=[(x,r) for x,r in st[t] if x>=cut]
        league[:]=[(x,r) for x,r in league if x>=cut]
        lm=np.mean([r for _,r in league]) if league else 4.5
        def rate(st,t):
            v=[r for _,r in st.get(t,[])]; return (sum(v)+pseudo*lm)/(len(v)+pseudo)
        for i,r in g.iterrows():
            a,h=r.away_team_code,r.home_team_code
            ae=.5*(rate(off,a)+rate(deff,h)); he=.5*(rate(off,h)+rate(deff,a)); rows.append((i,ae+he,ae,he))
        for _,r in g.iterrows():
            a,h=r.away_team_code,r.home_team_code; ar=float(r.away_score_num); hr=float(r.home_score_num)
            off.setdefault(a,[]).append((pd.Timestamp(dt),ar)); deff.setdefault(h,[]).append((pd.Timestamp(dt),ar)); off.setdefault(h,[]).append((pd.Timestamp(dt),hr)); deff.setdefault(a,[]).append((pd.Timestamp(dt),hr)); league.extend([(pd.Timestamp(dt),ar),(pd.Timestamp(dt),hr)])
    z=pd.DataFrame(rows,columns=['idx','baseball_expected_total','away_expected_runs','home_expected_runs']).set_index('idx')
    return d.join(z)

def design(x,knots):
    x=np.asarray(x,float); cols=[np.ones(len(x)),x,x*x,x*x*x]
    for k in knots: cols.append(np.maximum(x-k,0)**3)
    return np.column_stack(cols)

def fit(X,y,lam):
    b=np.zeros(X.shape[1]); P=np.eye(X.shape[1])*lam; P[0,0]=0
    for _ in range(100):
        p=1/(1+np.exp(-np.clip(X@b,-40,40))); w=np.clip(p*(1-p),1e-6,None); g=X.T@(y-p)-P@b; H=X.T@(w[:,None]*X)+P; nb=b+np.linalg.solve(H,g)
        if np.max(np.abs(nb-b))<1e-8: return nb
        b=nb
    return b

def pred(b,X): return 1/(1+np.exp(-np.clip(X@b,-40,40)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase1',required=True); ap.add_argument('--v04',required=True); ap.add_argument('--out',required=True); a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); phase=Path(a.phase1)
    d=pd.read_csv(next(phase.rglob('MLB_Game_Stats_Joined_2021_2025.csv.gz')),low_memory=False)
    if 'benchmark_matched' in d.columns: d=d[d.benchmark_matched==True].copy()
    d['game_date']=pd.to_datetime(d.game_date).dt.normalize(); sa,sh=score_cols(d); d['away_score_num']=pd.to_numeric(d[sa]); d['home_score_num']=pd.to_numeric(d[sh]); d['actual_under']=(pd.to_numeric(d.inning2_total_runs)==0).astype(int); d['away_team_code']=d.away_team_code.map(norm); d['home_team_code']=d.home_team_code.map(norm); d=add_baseball_total(d)
    games=pd.read_parquet(next(phase.rglob('games.parquet'))); games['game_date']=pd.to_datetime(games.game_date).dt.normalize(); games['away_team_code']=games.away_team.map(norm); games['home_team_code']=games.home_team.map(norm); games=games.sort_values(['game_date','away_team_code','home_team_code','game_id']); games['_seq']=games.groupby(['game_date','away_team_code','home_team_code']).cumcount(); d=d.sort_values(['game_date','away_team_code','home_team_code','retro_game_id']); d['_seq']=d.groupby(['game_date','away_team_code','home_team_code']).cumcount(); d=d.merge(games[['game_id','game_date','away_team_code','home_team_code','_seq']],on=['game_date','away_team_code','home_team_code','_seq'],how='left')
    v=pd.read_csv(next(Path(a.v04).rglob('v04_oos_predictions.csv'))); m=v.merge(d[d.game_id.notna()].drop_duplicates('game_id')[['game_id','baseball_expected_total','actual_under']],on='game_id',how='left',validate='one_to_one'); jr=float(m.baseball_expected_total.notna().mean()); print('join_rate',jr); m=m[m.baseball_expected_total.notna()].copy(); print('common_n',len(m),'of',len(v))
    if jr<.98: raise SystemExit('Common join rate below 98%')
    m['y']=m.actual_under.astype(float); candidates=[(n,l) for n in [0,2,3,4] for l in [.1,1,10,100]]; parts=[]; annual=[]
    for season in [2022,2023,2024,2025]:
        tr=m[m.season<season]; te=m[m.season==season]
        if tr.empty or te.empty: continue
        ps=sorted(tr.season.unique()); scores=[]
        for nk,lam in candidates:
            fs=[]
            if len(ps)>=2:
                for vs in ps[1:]:
                    tt=tr[tr.season<vs]; vv=tr[tr.season==vs]; qs=np.linspace(.2,.8,nk+2)[1:-1] if nk else []; ks=np.quantile(tt.baseball_expected_total,qs) if nk else []; bb=fit(design(tt.baseball_expected_total,ks),tt.y.to_numpy(),lam); fs.append(logloss(vv.y,pred(bb,design(vv.baseball_expected_total,ks))))
            else: fs=[999+nk*.001]
            scores.append((np.mean(fs),nk,lam))
        _,nk,lam=min(scores); qs=np.linspace(.2,.8,nk+2)[1:-1] if nk else []; ks=np.quantile(tr.baseball_expected_total,qs) if nk else []; bb=fit(design(tr.baseball_expected_total,ks),tr.y.to_numpy(),lam); p=pred(bb,design(te.baseball_expected_total,ks)); q=te.copy(); q['p_under_baseball_continuous']=p; q['selected_knots']=nk; q['selected_lambda']=lam; parts.append(q); annual.append({'season':season,'n':len(q),'knots':nk,'lambda':lam,'logloss':logloss(q.y,p),'brier':brier(q.y,p)})
    o=pd.concat(parts,ignore_index=True); comp=[]
    for split,x in [('ALL',o),('DEV',o[o.season<=2024]),('2025',o[o.season==2025])]:
        for name,col in [('BASEBALL_CONTINUOUS','p_under_baseball_continuous'),('V04_LOCAL_CV','p_under_local_cv')]: comp.append({'split':split,'model':name,'n':len(x),'logloss':logloss(x.y,x[col]),'brier':brier(x.y,x[col]),'mean_p_under':float(x[col].mean()),'actual_under':float(x.y.mean())})
    pd.DataFrame(comp).to_csv(out/'performance_comparison.csv',index=False); pd.DataFrame(annual).to_csv(out/'annual_metrics.csv',index=False); o[['game_id','season','baseball_expected_total','p_under_baseball_continuous','p_under_local_cv','actual_under','selected_knots','selected_lambda']].to_csv(out/'oos_predictions.csv',index=False)
    def mom(s):
        s=pd.Series(s); return {'mean':float(s.mean()),'sd':float(s.std()),'skew':float(s.skew()),'excess_kurtosis':float(s.kurt()),'p01':float(s.quantile(.01)),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99))}
    shape={'baseball_continuous':mom(o.p_under_baseball_continuous),'v04':mom(o.p_under_local_cv),'baseball_expected_total':mom(o.baseball_expected_total),'join_rate':jr,'common_n':len(m),'full_v04_n':len(v),'price_used':False}; (out/'shape.json').write_text(json.dumps(shape,indent=2)); print(pd.DataFrame(comp).to_string(index=False)); print(json.dumps(shape,indent=2))
if __name__=='__main__': main()

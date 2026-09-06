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

def find_score_cols(df):
    pairs=[('away_final_runs','home_final_runs'),('away_score','home_score'),('away_runs','home_runs'),('away_team_runs','home_team_runs'),('away_final_score','home_final_score'),('visitor_score','home_score')]
    for a,h in pairs:
        if a in df.columns and h in df.columns: return a,h
    raise ValueError('Could not identify away/home final score columns. Columns='+','.join(df.columns[:120]))

def norm_code(x):
    m={'ARI':'AZ','AZ':'AZ','OAK':'ATH','ATH':'ATH','CWS':'CHW','CHW':'CHW','KCR':'KC','KC':'KC','SDP':'SD','SD':'SD','SFG':'SF','SF':'SF','TBR':'TB','TB':'TB','WSN':'WSH','WAS':'WSH','WSH':'WSH'}
    s=str(x).upper().strip(); return m.get(s,s)

def build_continuous_total(df, window_days=365, shrink_games=30.0):
    d=df.sort_values(['game_date','away_team_code','home_team_code']).copy()
    hist_off={}; hist_def={}; league=[]; out=[]
    for date,grp in d.groupby('game_date', sort=True):
        cutoff=pd.Timestamp(date)-pd.Timedelta(days=window_days)
        for store in [hist_off,hist_def]:
            for t in list(store): store[t]=[(dt,r) for dt,r in store[t] if dt>=cutoff]
        league[:]=[(dt,r) for dt,r in league if dt>=cutoff]
        league_mean=np.mean([r for _,r in league]) if league else float(d.loc[d.game_date<date,['away_score_num','home_score_num']].stack().mean())
        if not np.isfinite(league_mean): league_mean=4.5
        for idx,r in grp.iterrows():
            a=norm_code(r.away_team_code); h=norm_code(r.home_team_code)
            def shrunk(store,t):
                vals=[v for _,v in store.get(t,[])]; n=len(vals); s=sum(vals)
                return (s+shrink_games*league_mean)/(n+shrink_games), n
            a_off,nao=shrunk(hist_off,a); h_off,nho=shrunk(hist_off,h); a_def,nad=shrunk(hist_def,a); h_def,nhd=shrunk(hist_def,h)
            away_exp=0.5*(a_off+h_def); home_exp=0.5*(h_off+a_def)
            out.append((idx,away_exp+home_exp,away_exp,home_exp,min(nao,nho,nad,nhd)))
        for idx,r in grp.iterrows():
            a=norm_code(r.away_team_code); h=norm_code(r.home_team_code); ar=float(r.away_score_num); hr=float(r.home_score_num)
            hist_off.setdefault(a,[]).append((pd.Timestamp(date),ar)); hist_def.setdefault(h,[]).append((pd.Timestamp(date),ar))
            hist_off.setdefault(h,[]).append((pd.Timestamp(date),hr)); hist_def.setdefault(a,[]).append((pd.Timestamp(date),hr))
            league.append((pd.Timestamp(date),ar)); league.append((pd.Timestamp(date),hr))
    f=pd.DataFrame(out,columns=['idx','baseball_expected_total','away_expected_runs','home_expected_runs','min_team_history_games']).set_index('idx')
    return d.join(f,how='left')

def spline_design(x, knots):
    x=np.asarray(x,float); cols=[np.ones(len(x)),x,x*x,x*x*x]
    for k in knots: cols.append(np.maximum(x-k,0.0)**3)
    return np.column_stack(cols)

def fit_logistic_ridge(X,y,lam=1.0,max_iter=100):
    beta=np.zeros(X.shape[1]); pen=np.eye(X.shape[1])*lam; pen[0,0]=0
    for _ in range(max_iter):
        eta=X@beta; p=1/(1+np.exp(-np.clip(eta,-40,40))); w=np.clip(p*(1-p),1e-6,None)
        grad=X.T@(y-p)-pen@beta; H=X.T@(w[:,None]*X)+pen; step=np.linalg.solve(H,grad); nb=beta+step
        if np.max(np.abs(nb-beta))<1e-8: beta=nb; break
        beta=nb
    return beta

def predict(beta,X): return 1/(1+np.exp(-np.clip(X@beta,-40,40)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase1',required=True); ap.add_argument('--v04',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    phase=Path(a.phase1)
    master=next(phase.rglob('MLB_Game_Stats_Joined_2021_2025.csv.gz')); d=pd.read_csv(master,low_memory=False)
    if 'benchmark_matched' in d.columns: d=d[d.benchmark_matched==True].copy()
    d['game_date']=pd.to_datetime(d['game_date'],errors='coerce').dt.normalize(); sca,sch=find_score_cols(d)
    d['away_score_num']=pd.to_numeric(d[sca],errors='coerce'); d['home_score_num']=pd.to_numeric(d[sch],errors='coerce'); d['i2_runs']=pd.to_numeric(d['inning2_total_runs'],errors='coerce'); d['actual_under']=(d.i2_runs==0).astype(int)
    d=d.dropna(subset=['game_date','away_score_num','home_score_num','i2_runs']).copy(); d['away_team_code']=d.away_team_code.map(norm_code); d['home_team_code']=d.home_team_code.map(norm_code); d=build_continuous_total(d)

    # Map canonical rows to StatsAPI game_id using normalized game table. This gives an exact key shared by v0.4 OOS predictions.
    gp=next(phase.rglob('games.parquet')); games=pd.read_parquet(gp)
    games['game_date']=pd.to_datetime(games['game_date'],errors='coerce').dt.normalize(); games['away_team_code']=games['away_team'].map(norm_code); games['home_team_code']=games['home_team'].map(norm_code)
    games=games.sort_values(['game_date','away_team_code','home_team_code','game_id']).copy(); games['_seq']=games.groupby(['game_date','away_team_code','home_team_code']).cumcount()
    d=d.sort_values(['game_date','away_team_code','home_team_code','retro_game_id']).copy(); d['_seq']=d.groupby(['game_date','away_team_code','home_team_code']).cumcount()
    d=d.merge(games[['game_id','game_date','away_team_code','home_team_code','_seq']],on=['game_date','away_team_code','home_team_code','_seq'],how='left')
    map_rate=float(d.game_id.notna().mean()); print('canonical_to_game_id_rate',map_rate)

    vp=next(Path(a.v04).rglob('v04_oos_predictions.csv')); v=pd.read_csv(vp)
    m=v.merge(d[['game_id','baseball_expected_total','away_expected_runs','home_expected_runs','min_team_history_games','actual_under']],on='game_id',how='left',validate='one_to_one')
    jr=float(m.baseball_expected_total.notna().mean()); print('join_rate',jr)
    if jr<0.99: raise SystemExit('Join rate below 99%')
    m=m[m.season.isin([2022,2023,2024,2025])].copy(); m['y']=m.actual_under.astype(float)

    candidates=[(n,l) for n in [0,2,3,4] for l in [0.1,1.0,10.0,100.0]]; pred_rows=[]; metric=[]
    for season in [2022,2023,2024,2025]:
        tr=m[m.season<season].copy(); te=m[m.season==season].copy()
        if tr.empty or te.empty: continue
        prior_seasons=sorted(tr.season.unique()); scores=[]
        for nknots,lam in candidates:
            fs=[]
            if len(prior_seasons)>=2:
                for vs in prior_seasons[1:]:
                    tt=tr[tr.season<vs]; vv=tr[tr.season==vs]; qs=np.linspace(.2,.8,nknots+2)[1:-1] if nknots else []; knots=np.quantile(tt.baseball_expected_total,qs) if nknots else []
                    b=fit_logistic_ridge(spline_design(tt.baseball_expected_total,knots),tt.y.to_numpy(),lam); fs.append(logloss(vv.y,predict(b,spline_design(vv.baseball_expected_total,knots))))
            else: fs=[999+nknots*1e-3]
            scores.append((np.mean(fs),nknots,lam))
        _,nk,lam=min(scores); qs=np.linspace(.2,.8,nk+2)[1:-1] if nk else []; knots=np.quantile(tr.baseball_expected_total,qs) if nk else []
        beta=fit_logistic_ridge(spline_design(tr.baseball_expected_total,knots),tr.y.to_numpy(),lam); p=predict(beta,spline_design(te.baseball_expected_total,knots))
        q=te.copy(); q['p_under_baseball_continuous']=p; q['selected_knots']=nk; q['selected_lambda']=lam; pred_rows.append(q)
        metric.append({'season':season,'n':len(q),'knots':nk,'lambda':lam,'logloss':logloss(q.y,p),'brier':brier(q.y,p),'mean_p':float(np.mean(p)),'actual_under':float(np.mean(q.y))})
    oos=pd.concat(pred_rows,ignore_index=True); comps=[]
    for split,dd in [('ALL',oos),('DEV',oos[oos.season<=2024]),('2025',oos[oos.season==2025])]:
        for name,col in [('BASEBALL_CONTINUOUS','p_under_baseball_continuous'),('V04_LOCAL_CV','p_under_local_cv')]:
            comps.append({'split':split,'model':name,'n':len(dd),'logloss':logloss(dd.y,dd[col]),'brier':brier(dd.y,dd[col]),'mean_p_under':float(dd[col].mean()),'actual_under':float(dd.y.mean())})
    pd.DataFrame(metric).to_csv(out/'annual_metrics.csv',index=False); pd.DataFrame(comps).to_csv(out/'performance_comparison.csv',index=False)
    oos[['game_id','season','baseball_expected_total','p_under_baseball_continuous','p_under_local_cv','actual_under','selected_knots','selected_lambda']].to_csv(out/'oos_predictions.csv',index=False)
    def moments(s):
        s=pd.Series(s).dropna(); return {'mean':float(s.mean()),'sd':float(s.std()),'skew':float(s.skew()),'excess_kurtosis':float(s.kurt()),'p01':float(s.quantile(.01)),'p05':float(s.quantile(.05)),'p50':float(s.quantile(.5)),'p95':float(s.quantile(.95)),'p99':float(s.quantile(.99))}
    shape={'baseball_continuous':moments(oos.p_under_baseball_continuous),'v04':moments(oos.p_under_local_cv),'baseball_expected_total':moments(oos.baseball_expected_total),'score_columns':[sca,sch],'canonical_to_game_id_rate':map_rate,'join_rate':jr,'price_used':False}
    (out/'shape.json').write_text(json.dumps(shape,indent=2)); print(pd.DataFrame(comps).to_string(index=False)); print(json.dumps(shape,indent=2))
if __name__=='__main__': main()

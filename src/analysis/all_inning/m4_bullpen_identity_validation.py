#!/usr/bin/env python3
"""Validate a pregame bullpen reliever identity mixture for the M4 bullpen branch.

Development only: 2021-2024. 2025 is never loaded. Targets are realized
half-innings whose first pitcher is not the game's starter. Candidate relievers
come only from strictly prior-date relief work for that team. Candidate scores
blend team relief-PA share with same-inning first-reliever usage. An explicit
OTHER state receives the training-fold empirical unseen-reliever mass.

No market data, smoothing, or legacy bullpen-quality shrinkage is used.
"""
from __future__ import annotations
import argparse,json,sys
from collections import defaultdict,deque
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from src.ingestion.build_historical_order_bullpen_features import prep_pa, classify_relief

WINDOWS=[30,90,365]
ALPHAS=[0.0,0.25,0.5,0.75]
TEST_YEARS=[2022,2023,2024]
EPS=1e-12

def read(p): return pd.read_parquet(p) if Path(p).suffix=='.parquet' else pd.read_csv(p)

def build_targets(pa,starters):
    x=classify_relief(pa,starters)
    st=starters[["game_id","team_id","pitcher_id"]].dropna(subset=["pitcher_id"]).rename(columns={"team_id":"pitching_team_id","pitcher_id":"starter_id2"})
    z=(x[x.inning.between(1,9)].sort_values(["game_id","inning","pitching_team_id","play_index"],kind="stable")
       .groupby(["game_id","game_date","season","inning","pitching_team_id"],as_index=False).first())
    z=z.merge(st,on=["game_id","pitching_team_id"],how="left",validate="many_to_one")
    z=z[z.pitcher_id!=z.starter_id2].copy().rename(columns={"pitcher_id":"actual_reliever_id"})
    z=z[["game_id","game_date","season","inning","pitching_team_id","actual_reliever_id"]].reset_index(drop=True)
    z['target_id']=z.index.astype(int)
    return z

def build_daily_history(pa,starters,targets):
    x=classify_relief(pa,starters); r=x[x.is_relief].copy()
    bf=(r.groupby(["pitching_team_id","pitcher_id","game_date"],as_index=False).size().rename(columns={"size":"relief_pa"}))
    first=(targets.groupby(["pitching_team_id","actual_reliever_id","game_date","inning"],as_index=False).size()
           .rename(columns={"actual_reliever_id":"pitcher_id","size":"first_count"}))
    return bf,first

def materialize(targets,bf,first):
    """Sliding-window materialization; same-day events are added only after targets."""
    meta=[]; candidates=[]
    teams=sorted(targets.pitching_team_id.dropna().unique())
    for team in teams:
        tt=targets[targets.pitching_team_id==team].sort_values(['game_date','target_id'])
        b=bf[bf.pitching_team_id==team].sort_values('game_date')
        f=first[first.pitching_team_id==team].sort_values('game_date')
        tdates={pd.Timestamp(d):g for d,g in tt.groupby('game_date',sort=True)}
        bdates={pd.Timestamp(d):g for d,g in b.groupby('game_date',sort=True)}
        fdates={pd.Timestamp(d):g for d,g in f.groupby('game_date',sort=True)}
        dates=sorted(set(tdates)|set(bdates)|set(fdates))
        states={w:{'bfq':deque(),'fq':deque(),'bf':defaultdict(float),'fi':defaultdict(float)} for w in WINDOWS}
        for date in dates:
            for w,s in states.items():
                cutoff=date-pd.Timedelta(days=w)
                while s['bfq'] and s['bfq'][0][0] < cutoff:
                    d,p,v=s['bfq'].popleft(); s['bf'][p]-=v
                    if s['bf'][p] <= 1e-12: s['bf'].pop(p,None)
                while s['fq'] and s['fq'][0][0] < cutoff:
                    d,inn,p,v=s['fq'].popleft(); key=(inn,p); s['fi'][key]-=v
                    if s['fi'][key] <= 1e-12: s['fi'].pop(key,None)
            if date in tdates:
                for _,t in tdates[date].iterrows():
                    rid=int(t.target_id); inn=int(t.inning); actual=t.actual_reliever_id
                    for w,s in states.items():
                        pids=list(s['bf'].keys()); n=len(pids); seen=actual in s['bf']
                        meta.append({'target_id':rid,'game_id':t.game_id,'game_date':date,'season':int(t.season),'inning':inn,'pitching_team_id':team,'actual_reliever_id':actual,'window':w,'candidate_n':n,'actual_seen':bool(seen)})
                        if n:
                            gb=sum(s['bf'].values()); ivals={p:s['fi'].get((inn,p),0.0) for p in pids}; ib=sum(ivals.values())
                            for p in pids:
                                gs=s['bf'][p]/gb
                                ishare=ivals[p]/ib if ib>0 else gs
                                candidates.append({'target_id':rid,'window':w,'candidate_pitcher_id':p,'global_share':float(gs),'inning_share':float(ishare)})
            # Current-date relief is intentionally unavailable to all same-date targets.
            if date in bdates:
                for _,r in bdates[date].iterrows():
                    for w,s in states.items():
                        v=float(r.relief_pa); p=r.pitcher_id; s['bfq'].append((date,p,v)); s['bf'][p]+=v
            if date in fdates:
                for _,r in fdates[date].iterrows():
                    for w,s in states.items():
                        v=float(r.first_count); p=r.pitcher_id; inn=int(r.inning); s['fq'].append((date,inn,p,v)); s['fi'][(inn,p)]+=v
    return pd.DataFrame(meta),pd.DataFrame(candidates)

def score(targets,cands):
    out=[]; groups={(int(k[0]),int(k[1])):g for k,g in cands.groupby(['target_id','window'],sort=False)} if len(cands) else {}
    for year in TEST_YEARS:
        for days in WINDOWS:
            tr=targets[(targets.season<year)&(targets.window==days)]; te=targets[(targets.season==year)&(targets.window==days)]
            for inn in range(2,10):
                a=tr[tr.inning==inn]; b=te[te.inning==inn]
                if len(a)<50 or len(b)<50: continue
                unseen=float((~a.actual_seen).mean())
                if unseen<=0: unseen=float((~tr.actual_seen).mean())
                unseen=float(np.clip(unseen,EPS,1-EPS))
                for alpha in ALPHAS:
                    loss=[]; base_loss=[]; brier=[]; base_brier=[]; top1=[]; top3=[]; rr=[]
                    for _,t in b.iterrows():
                        g=groups.get((int(t.target_id),days)); n=int(t.candidate_n)
                        if g is None or n==0:
                            p_act=p_base=unseen; sq=sq_base=unseen**2; rank=np.inf
                        else:
                            scores=(1-alpha)*g.global_share.to_numpy(float)+alpha*g.inning_share.to_numpy(float); scores=scores/float(scores.sum())
                            ids=g.candidate_pitcher_id.to_numpy(); hit=np.where(ids==t.actual_reliever_id)[0]
                            if len(hit):
                                j=int(hit[0]); p_act=(1-unseen)*float(scores[j]); p_base=(1-unseen)/n; order=np.argsort(-scores); rank=int(np.where(order==j)[0][0])+1
                            else: p_act=p_base=unseen; rank=np.inf
                            sq=unseen**2+float(np.sum(((1-unseen)*scores)**2)); sq_base=unseen**2+n*((1-unseen)/n)**2
                        loss.append(-np.log(max(p_act,EPS))); base_loss.append(-np.log(max(p_base,EPS))); brier.append(1+sq-2*p_act); base_brier.append(1+sq_base-2*p_base)
                        top1.append(float(rank==1)); top3.append(float(rank<=3)); rr.append(0.0 if not np.isfinite(rank) else 1.0/rank)
                    out.append({'test_year':year,'inning':inn,'window':days,'alpha_inning_usage':alpha,'n_test':len(b),'training_other_mass':unseen,'identity_logloss':float(np.mean(loss)),'uniform_logloss':float(np.mean(base_loss)),'logloss_improvement':float(np.mean(base_loss)-np.mean(loss)),'identity_brier':float(np.mean(brier)),'uniform_brier':float(np.mean(base_brier)),'brier_improvement':float(np.mean(base_brier)-np.mean(brier)),'top1_hit_rate':float(np.mean(top1)),'top3_hit_rate':float(np.mean(top3)),'mean_reciprocal_rank':float(np.mean(rr)),'mean_candidate_n':float(b.candidate_n.mean()),'seen_rate':float(b.actual_seen.mean())})
    return pd.DataFrame(out)

def summarize(f):
    s=(f.groupby(['inning','window','alpha_inning_usage'],as_index=False).agg(mean_logloss_improvement=('logloss_improvement','mean'),worst_year_logloss_improvement=('logloss_improvement','min'),mean_brier_improvement=('brier_improvement','mean'),worst_year_brier_improvement=('brier_improvement','min'),mean_top1=('top1_hit_rate','mean'),mean_top3=('top3_hit_rate','mean'),mean_mrr=('mean_reciprocal_rank','mean'),mean_seen_rate=('seen_rate','mean'),mean_candidate_n=('mean_candidate_n','mean')))
    s['all_years_logloss_positive']=s.worst_year_logloss_improvement>0; s['all_years_brier_positive']=s.worst_year_brier_improvement>0
    best=(s.sort_values(['inning','mean_logloss_improvement'],ascending=[True,False]).groupby('inning',as_index=False).head(1).sort_values('inning'))
    return s,best

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plate-appearances',type=Path,required=True); ap.add_argument('--starters',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa,_=prep_pa(read(a.plate_appearances)); st=read(a.starters)
    if set(pa.season.dropna().astype(int).unique())!={2021,2022,2023,2024}: raise RuntimeError('development seasons must be exactly 2021-2024')
    if (pa.season>=2025).any(): raise RuntimeError('2025 leakage')
    targets=build_targets(pa,st); bf,first=build_daily_history(pa,st,targets); tm,cm=materialize(targets,bf,first); folds=score(tm,cm); summary,best=summarize(folds)
    tm.to_parquet(a.output_dir/'m4_bullpen_identity_targets.parquet',index=False); cm.to_parquet(a.output_dir/'m4_bullpen_identity_candidates.parquet',index=False); folds.to_csv(a.output_dir/'m4_bullpen_identity_folds.csv',index=False); summary.to_csv(a.output_dir/'m4_bullpen_identity_summary.csv',index=False); best.to_csv(a.output_dir/'m4_bullpen_identity_best_by_inning.csv',index=False)
    manifest={'status':'PASS','architecture':'M4_bullpen_first_reliever_identity','development_seasons':[2021,2022,2023,2024],'test_folds':TEST_YEARS,'holdout_season':2025,'holdout_opened':False,'candidate_windows_days':WINDOWS,'candidate_inning_usage_blends':ALPHAS,'other_state':'training-fold empirical unseen reliever rate','history_cutoff':'strictly prior date; same-day excluded','legacy_quality_shrinkage_used':False,'market_data_used':False,'relief_target_rows':int(len(targets)),'best_identity_spec_by_inning':best.to_dict('records'),'automatic_production_promotion':False,'note':'Reliever identity/state validation only. Quality skill attachment occurs after identity mixture validation.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

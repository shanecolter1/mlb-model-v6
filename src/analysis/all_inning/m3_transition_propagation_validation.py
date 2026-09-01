#!/usr/bin/env python3
"""Validate pregame I1-I9 batting-order path propagation.

Development only: 2021-2024. 2025 is never loaded.

M3 already established that raw empirical inning start-slot distributions are
predictive. This audit tests the more structural pregame path model: start from
the deterministic I1 slot=1 state and recursively propagate raw empirical
transition matrices P(start_i | start_{i-1}) estimated only from prior seasons.

The propagated distribution is compared directly with the unconditional raw
prior-season inning distribution on chronological 2022/2023/2024 folds. No
smoothing is used; a 1e-12 floor exists only for finite evaluation log loss and
never changes the stored model probabilities.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

TEST_YEARS=[2022,2023,2024]
EPS=1e-12

def distribution(g):
    c=g.batting_order_slot.value_counts(); n=len(g)
    return np.array([float(c.get(s,0))/n for s in range(1,10)],float)

def transition_matrix(first,inning):
    x=first.sort_values(['game_id','team_id','inning']).copy()
    x['prev_inning']=x.groupby(['game_id','team_id']).inning.shift(1)
    x['prev_slot']=x.groupby(['game_id','team_id']).batting_order_slot.shift(1)
    x=x[(x.inning==inning)&(x.prev_inning==inning-1)].copy()
    M=np.zeros((9,9),float)
    for prev in range(1,10):
        g=x[x.prev_slot==prev]
        if len(g): M[prev-1,:]=distribution(g)
    return M

def metrics(y,p):
    ll=[]; br=[]
    for obs in y:
        q=float(p[int(obs)-1]); ll.append(-np.log(max(q,EPS)))
        one=np.zeros(9); one[int(obs)-1]=1; br.append(float(np.sum((p-one)**2)))
    return float(np.mean(ll)),float(np.mean(br))

def run(first):
    rows=[]; probs=[]
    for year in TEST_YEARS:
        tr=first[first.season<year].copy(); te=first[first.season==year].copy()
        propagated=np.zeros(9); propagated[0]=1.0
        for inning in range(1,10):
            b=te[te.inning==inning]
            if inning==1:
                prop=propagated.copy()
            else:
                M=transition_matrix(tr,inning)
                # If a prior-state row has no historical support, use destination unconditional raw distribution.
                dest=distribution(tr[tr.inning==inning])
                empty=M.sum(axis=1)==0; M[empty,:]=dest
                prop=propagated@M
                propagated=prop.copy()
            unc=distribution(tr[tr.inning==inning])
            y=b.batting_order_slot.astype(int).to_numpy()
            pll,pbr=metrics(y,prop); ull,ubr=metrics(y,unc)
            rows.append({'test_year':year,'inning':inning,'n_test':len(b),'propagated_logloss':pll,'unconditional_logloss':ull,'logloss_improvement_vs_unconditional':ull-pll,'propagated_brier':pbr,'unconditional_brier':ubr,'brier_improvement_vs_unconditional':ubr-pbr})
            for s in range(1,10): probs.append({'test_year':year,'inning':inning,'slot':s,'propagated_probability':float(prop[s-1]),'unconditional_probability':float(unc[s-1])})
    return pd.DataFrame(rows),pd.DataFrame(probs)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--realized-slots',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.realized_slots); x['season']=pd.to_numeric(x.season).astype(int); x['inning']=pd.to_numeric(x.inning).astype(int)
    if set(x.season.unique())!={2021,2022,2023,2024}: raise RuntimeError('development seasons must be exactly 2021-2024')
    if (x.season>=2025).any(): raise RuntimeError('2025 leakage')
    folds,probs=run(x)
    summary=(folds.groupby('inning',as_index=False).agg(mean_logloss_improvement=('logloss_improvement_vs_unconditional','mean'),worst_year_logloss_improvement=('logloss_improvement_vs_unconditional','min'),mean_brier_improvement=('brier_improvement_vs_unconditional','mean'),worst_year_brier_improvement=('brier_improvement_vs_unconditional','min')))
    summary['all_years_logloss_nonnegative']=summary.worst_year_logloss_improvement>=-1e-12; summary['all_years_brier_nonnegative']=summary.worst_year_brier_improvement>=-1e-12
    folds.to_csv(a.output_dir/'m3_transition_propagation_folds.csv',index=False); probs.to_csv(a.output_dir/'m3_transition_propagation_probabilities.csv',index=False); summary.to_csv(a.output_dir/'m3_transition_propagation_summary.csv',index=False)
    manifest={'status':'PASS','architecture':'M3_pregame_transition_propagation','development_seasons':[2021,2022,2023,2024],'test_folds':TEST_YEARS,'holdout_season':2025,'holdout_opened':False,'initial_state':'I1 start slot 1 deterministic','transition_source':'raw prior-season empirical P(start_i | start_i-1)','smoothing_used':False,'evaluation_floor':EPS,'market_data_used':False,'summary_by_inning':summary.to_dict('records'),'automatic_production_promotion':False,'note':'Determines whether recursive pregame path propagation should replace unconditional M3 inning distributions.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()

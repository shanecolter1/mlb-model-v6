#!/usr/bin/env python3
"""Materialize and validate the all-inning M3 batting-order path layer.

This is a development-only 2021-2024 materialization of the already validated
raw empirical batting-order method. It intentionally reuses the existing
substitution-safe PA-ordinal modulo-nine reconstruction and does NOT introduce
smoothing or new shrinkage.

Outputs provide:
- realized inning start slot for each batting team/game/inning;
- chronological league start-slot probabilities using strictly earlier seasons
  for 2022/2023/2024 forward folds;
- raw transition distributions from prior inning start slot to next inning;
- a canonical I1-I9 development distribution for later pitcher x batter matchup
  assembly.

2025 is never loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.build_historical_order_bullpen_features import (
    prep_pa,
    inning_start_observations,
    empirical_distribution,
    build_transitions,
)

EPS=1e-12


def logloss_multiclass(obs, probs):
    vals=[]
    for y,pmap in zip(obs,probs):
        vals.append(-np.log(max(float(pmap.get(int(y),0.0)),EPS)))
    return float(np.mean(vals))


def brier_multiclass(obs, probs):
    vals=[]
    for y,pmap in zip(obs,probs):
        vals.append(sum((float(pmap.get(s,0.0))-(1.0 if s==int(y) else 0.0))**2 for s in range(1,10)))
    return float(np.mean(vals))


def chronological_validate(first):
    rows=[]; pred_rows=[]
    for year in [2022,2023,2024]:
        tr=first[first.season<year].copy(); te=first[first.season==year].copy()
        for inning in range(1,10):
            a=tr[tr.inning==inning]; b=te[te.inning==inning]
            if a.empty or b.empty: continue
            counts=a.batting_order_slot.value_counts()
            probs={s:float(counts.get(s,0))/len(a) for s in range(1,10)}
            obs=b.batting_order_slot.astype(int).tolist(); pm=[probs]*len(obs)
            ell=logloss_multiclass(obs,pm); ebr=brier_multiclass(obs,pm)
            ull=float(np.log(9.0)); ubr=float(8/9)
            rows.append({"test_year":year,"inning":inning,"train_n":len(a),"test_n":len(b),
                         "empirical_logloss":ell,"uniform_logloss":ull,"logloss_improvement":ull-ell,
                         "empirical_brier":ebr,"uniform_brier":ubr,"brier_improvement":ubr-ebr})
            for s in range(1,10):
                pred_rows.append({"test_year":year,"inning":inning,"batting_order_slot":s,
                                  "train_n":len(a),"probability":probs[s]})
    return pd.DataFrame(rows),pd.DataFrame(pred_rows)


def make_state_matrix(first):
    """Attach strictly prior-season league distribution to realized targets.

    For each target season 2022-2024, probabilities are estimated from all prior
    development seasons only. 2021 is retained as foundation rows with NA prior.
    """
    x=first.copy()
    for s in range(1,10): x[f"p_start_slot_{s}"]=np.nan
    x["prior_train_n"]=0
    for year in [2022,2023,2024]:
        tr=first[first.season<year]
        mask=x.season==year
        for inn in range(1,10):
            a=tr[tr.inning==inn]
            if a.empty: continue
            counts=a.batting_order_slot.value_counts(); n=len(a)
            m=mask & (x.inning==inn)
            x.loc[m,"prior_train_n"]=n
            for s in range(1,10): x.loc[m,f"p_start_slot_{s}"]=float(counts.get(s,0))/n
    x["distribution_source"]="league_empirical_prior_seasons_raw_no_smoothing"
    return x


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plate-appearances",type=Path,required=True); ap.add_argument("--lineups",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    pa=pd.read_parquet(a.plate_appearances); lu=pd.read_parquet(a.lineups)
    pa,_=prep_pa(pa)
    seasons=set(pa.season.dropna().astype(int).unique())
    if seasons!={2021,2022,2023,2024}: raise RuntimeError(f"expected exactly 2021-2024, got {seasons}")
    if (pa.season>=2025).any(): raise RuntimeError("2025 holdout leakage")
    first=inning_start_observations(pa,lu)
    if set(first.inning.astype(int).unique())!=set(range(1,10)): raise RuntimeError("I1-I9 incomplete")

    overall=empirical_distribution(first,["inning"])
    byseason=empirical_distribution(first,["season","inning"])
    transitions=build_transitions(first)
    folds,fold_probs=chronological_validate(first)
    state=make_state_matrix(first)

    # Exact probability-mass audits.
    chk=overall.groupby("inning").probability.sum()
    if not ((chk-1).abs()<1e-12).all(): raise RuntimeError(f"overall probability mass failure {chk}")
    tchk=transitions.groupby(["inning","prev_start_slot"]).probability.sum()
    if not ((tchk-1).abs()<1e-12).all(): raise RuntimeError("transition probability mass failure")
    pcols=[f"p_start_slot_{s}" for s in range(1,10)]
    hist=state[state.prior_train_n>0]
    if not ((hist[pcols].sum(axis=1)-1).abs()<1e-12).all(): raise RuntimeError("state probability mass failure")

    summary=(folds.groupby("inning",as_index=False)
             .agg(mean_logloss_improvement=("logloss_improvement","mean"),
                  worst_year_logloss_improvement=("logloss_improvement","min"),
                  mean_brier_improvement=("brier_improvement","mean"),
                  worst_year_brier_improvement=("brier_improvement","min")))
    summary["all_years_logloss_positive"]=summary.worst_year_logloss_improvement>0
    summary["all_years_brier_positive"]=summary.worst_year_brier_improvement>0

    first.to_parquet(a.output_dir/"m3_realized_start_slots.parquet",index=False)
    state.to_parquet(a.output_dir/"m3_batter_path_state_matrix.parquet",index=False)
    overall.to_csv(a.output_dir/"m3_empirical_start_slot_distribution_2021_2024.csv",index=False)
    byseason.to_csv(a.output_dir/"m3_start_slot_distribution_by_season.csv",index=False)
    transitions.to_csv(a.output_dir/"m3_transition_distribution.csv",index=False)
    folds.to_csv(a.output_dir/"m3_walkforward_fold_results.csv",index=False)
    fold_probs.to_csv(a.output_dir/"m3_walkforward_probabilities.csv",index=False)
    summary.to_csv(a.output_dir/"m3_walkforward_summary_by_inning.csv",index=False)

    manifest={
      "status":"PASS","architecture":"M3_batter_path_identification_I1_I9",
      "development_seasons":[2021,2022,2023,2024],"test_folds":[2022,2023,2024],
      "holdout_season":2025,"holdout_opened":False,
      "innings":list(range(1,10)),"slot_source":"chronological_team_PA_ordinal_modulo_9_substitution_safe",
      "probability_method":"raw_empirical_prior_seasons_by_inning","smoothing_used":False,"shrinkage_used":False,
      "market_data_used":False,"realized_rows":int(len(first)),"state_rows":int(len(state)),
      "walkforward_summary_by_inning":summary.to_dict("records"),
      "automatic_production_promotion":False,
      "note":"Development-only materialization of the previously validated raw batting-order path method. No 2025 data loaded."
    }
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()

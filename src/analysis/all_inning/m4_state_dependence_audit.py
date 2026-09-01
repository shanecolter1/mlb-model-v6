#!/usr/bin/env python3
"""Chronological dependence audit between M2 pitcher state and M3 batter path.

Development only: 2021-2024. 2025 is never loaded.

The M4 architecture must not assume pitcher-state and batting-path independence
without evidence. This audit pairs each realized batting half-inning with the
opposing pitching team's M2 state and tests whether conditioning on the realized
starter-vs-bullpen state improves prediction of the inning start slot beyond the
training-fold inning distribution. It also runs the reverse test: whether the
realized start slot improves starter-state prediction beyond the inning baseline.

No smoothing, shrinkage, market data, or production promotion is introduced.
Sparse conditional cells fall back to the corresponding training-fold inning
baseline; this is an explicit fallback, not smoothing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TEST_YEARS=[2022,2023,2024]
EPS=1e-12


def multiclass_brier(y, probs):
    vals=[]
    for obs,p in zip(y,probs):
        vals.append(sum((float(p.get(s,0.0))-(1.0 if int(obs)==s else 0.0))**2 for s in range(1,10)))
    return float(np.mean(vals))

def multiclass_logloss(y,probs):
    return float(np.mean([-np.log(max(float(p.get(int(obs),0.0)),EPS)) for obs,p in zip(y,probs)]))

def binary_logloss(y,p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS); y=np.asarray(y,float)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())

def binary_brier(y,p):
    return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))


def pair_states(m2,m3):
    a=m2[["game_id","game_date","season","inning","pitching_team_id","starter_begins_inning","starter_pa_share"]].copy()
    b=m3[["game_id","game_date","season","team_id","inning","batting_order_slot"]].copy().rename(columns={"team_id":"batting_team_id"})
    x=a.merge(b,on=["game_id","game_date","season","inning"],how="inner")
    x=x[x.pitching_team_id!=x.batting_team_id].copy()
    # Exactly one opponent batting row should remain for each realized pitching half.
    key=["game_id","inning","pitching_team_id"]
    dup=x.duplicated(key,keep=False)
    if dup.any():
        raise RuntimeError(f"nonunique opposing half pairing: {int(dup.sum())} rows")
    return x


def slot_given_state(x):
    rows=[]
    for year in TEST_YEARS:
        tr=x[x.season<year]; te=x[x.season==year]
        for inning in range(1,10):
            a=tr[tr.inning==inning]; b=te[te.inning==inning]
            if len(a)<100 or len(b)<100: continue
            c=a.batting_order_slot.value_counts(); base={s:float(c.get(s,0))/len(a) for s in range(1,10)}
            cond={}
            for state,g in a.groupby("starter_begins_inning"):
                vc=g.batting_order_slot.value_counts(); cond[int(state)]={s:float(vc.get(s,0))/len(g) for s in range(1,10)}
            y=b.batting_order_slot.astype(int).tolist(); p0=[base]*len(b)
            p1=[cond.get(int(s),base) for s in b.starter_begins_inning]
            ll0=multiclass_logloss(y,p0); ll1=multiclass_logloss(y,p1)
            br0=multiclass_brier(y,p0); br1=multiclass_brier(y,p1)
            rows.append({"direction":"slot_given_pitcher_state","test_year":year,"inning":inning,"n_train":len(a),"n_test":len(b),
                         "baseline_logloss":ll0,"conditional_logloss":ll1,"logloss_improvement":ll0-ll1,
                         "baseline_brier":br0,"conditional_brier":br1,"brier_improvement":br0-br1,
                         "train_starter_begin_rate":float(a.starter_begins_inning.mean())})
    return pd.DataFrame(rows)


def state_given_slot(x):
    rows=[]
    for year in TEST_YEARS:
        tr=x[x.season<year]; te=x[x.season==year]
        for inning in range(2,10):
            a=tr[tr.inning==inning]; b=te[te.inning==inning]
            if len(a)<100 or len(b)<100: continue
            base=float(a.starter_begins_inning.mean()); byslot=a.groupby("batting_order_slot").starter_begins_inning.mean().to_dict()
            y=b.starter_begins_inning.to_numpy(float); p0=np.full(len(b),base); p1=np.array([float(byslot.get(int(s),base)) for s in b.batting_order_slot])
            ll0=binary_logloss(y,p0); ll1=binary_logloss(y,p1); br0=binary_brier(y,p0); br1=binary_brier(y,p1)
            rows.append({"direction":"pitcher_state_given_slot","test_year":year,"inning":inning,"n_train":len(a),"n_test":len(b),
                         "baseline_logloss":ll0,"conditional_logloss":ll1,"logloss_improvement":ll0-ll1,
                         "baseline_brier":br0,"conditional_brier":br1,"brier_improvement":br0-br1,
                         "train_starter_begin_rate":base})
    return pd.DataFrame(rows)


def summarize(f):
    s=(f.groupby(["direction","inning"],as_index=False)
       .agg(mean_logloss_improvement=("logloss_improvement","mean"),worst_year_logloss_improvement=("logloss_improvement","min"),
            mean_brier_improvement=("brier_improvement","mean"),worst_year_brier_improvement=("brier_improvement","min")))
    s["all_years_logloss_positive"]=s.worst_year_logloss_improvement>0
    s["all_years_brier_positive"]=s.worst_year_brier_improvement>0
    return s


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m2",type=Path,required=True); ap.add_argument("--m3",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    m2=pd.read_parquet(a.m2); m3=pd.read_parquet(a.m3)
    for df,name in [(m2,"m2"),(m3,"m3")]:
        df["season"]=pd.to_numeric(df.season,errors="coerce").astype(int); df["inning"]=pd.to_numeric(df.inning,errors="coerce").astype(int)
        if set(df.season.unique())!={2021,2022,2023,2024}: raise RuntimeError(f"{name} seasons not dev-only")
        if (df.season>=2025).any(): raise RuntimeError(f"{name} 2025 leakage")
    x=pair_states(m2,m3)
    if set(x.inning.unique())!=set(range(1,10)): raise RuntimeError("paired I1-I9 coverage incomplete")
    f=pd.concat([slot_given_state(x),state_given_slot(x)],ignore_index=True); s=summarize(f)
    x.to_parquet(a.output_dir/"m4_paired_realized_states.parquet",index=False)
    f.to_csv(a.output_dir/"m4_state_dependence_folds.csv",index=False); s.to_csv(a.output_dir/"m4_state_dependence_summary.csv",index=False)
    # Evidence flag is descriptive: conditioning must improve both metrics in all dev years for a direction/inning to be called stable dependence.
    stable=s[s.all_years_logloss_positive & s.all_years_brier_positive]
    manifest={"status":"PASS","architecture":"M4_pitcher_batter_state_dependence_audit","development_seasons":[2021,2022,2023,2024],"test_folds":TEST_YEARS,
              "holdout_season":2025,"holdout_opened":False,"paired_rows":int(len(x)),"market_data_used":False,"smoothing_used":False,"shrinkage_used":False,
              "sparse_cell_handling":"training-fold inning baseline fallback","stable_dependence_cells":stable.to_dict("records"),
              "automatic_production_promotion":False,"note":"Use results to decide whether M2 and M3 may be factorized in M4. 2025 remains untouched."}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()

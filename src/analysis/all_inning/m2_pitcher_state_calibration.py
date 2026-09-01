#!/usr/bin/env python3
"""Empirically calibrate M2 starter-state probabilities for I1-I9.

Development only: 2021-2024. 2025 is never loaded.

The raw M2 state-history model demonstrated useful Brier/rank information but
poor log loss because unsmoothed player histories can become extreme. This
script therefore tests transparent partial-pooling blend weights between the
training-fold inning baseline and the strictly-prior-date raw player history:

  calibrated = baseline + weight * (raw_history - baseline)

Weight=0 is the baseline-only candidate. Weight=1 is the uncalibrated raw
history. Window and weight are selected only on chronological 2022/2023/2024
folds. No production promotion occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS=["season","30d","90d","365d"]
WEIGHTS=[0.0,0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.75,1.0]
TEST_YEARS=[2022,2023,2024]
EPS=1e-12


def logloss(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())

def mse(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean((p-y)**2))

def mae(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    return float(np.mean(np.abs(p-y)))


def evaluate_binary(x):
    rows=[]
    for inning in range(2,10):
        for year in TEST_YEARS:
            tr=x[(x.season<year)&(x.inning==inning)]
            te=x[(x.season==year)&(x.inning==inning)]
            base=float(tr.starter_begins_inning.mean())
            y=te.starter_begins_inning.to_numpy(float)
            p0=np.full(len(te),base); base_ll=logloss(y,p0); base_br=mse(y,p0)
            for w in WINDOWS:
                col=f"{w}_starter_begin_rate"
                raw=pd.to_numeric(te[col],errors="coerce").to_numpy(float)
                valid=np.isfinite(raw)
                for weight in WEIGHTS:
                    p=np.full(len(te),base,float)
                    p[valid]=base+weight*(raw[valid]-base)
                    p=np.clip(p,EPS,1-EPS)
                    ll=logloss(y,p); br=mse(y,p)
                    rows.append({"target":"starter_begins_inning","window":w,"weight":weight,
                                 "inning":inning,"test_year":year,"n_test":len(te),
                                 "history_coverage":float(valid.mean()),"baseline_logloss":base_ll,
                                 "model_logloss":ll,"baseline_brier":base_br,"model_brier":br,
                                 "logloss_improvement":base_ll-ll,"brier_improvement":base_br-br})
    return pd.DataFrame(rows)


def evaluate_share(x):
    rows=[]
    for inning in range(1,10):
        for year in TEST_YEARS:
            tr=x[(x.season<year)&(x.inning==inning)]
            te=x[(x.season==year)&(x.inning==inning)]
            base=float(tr.starter_pa_share.mean())
            y=te.starter_pa_share.to_numpy(float)
            p0=np.full(len(te),base); base_mse=mse(y,p0); base_mae=mae(y,p0)
            for w in WINDOWS:
                col=f"{w}_starter_pa_share_mean"
                raw=pd.to_numeric(te[col],errors="coerce").to_numpy(float)
                valid=np.isfinite(raw)
                for weight in WEIGHTS:
                    p=np.full(len(te),base,float)
                    p[valid]=base+weight*(raw[valid]-base)
                    p=np.clip(p,0,1)
                    mm=mse(y,p); aa=mae(y,p)
                    rows.append({"target":"starter_pa_share","window":w,"weight":weight,
                                 "inning":inning,"test_year":year,"n_test":len(te),
                                 "history_coverage":float(valid.mean()),"baseline_mse":base_mse,
                                 "model_mse":mm,"baseline_mae":base_mae,"model_mae":aa,
                                 "mse_improvement":base_mse-mm,"mae_improvement":base_mae-aa})
    return pd.DataFrame(rows)


def summarize_binary(f):
    s=(f.groupby(["window","weight","inning"],as_index=False)
       .agg(mean_logloss_improvement=("logloss_improvement","mean"),
            worst_year_logloss_improvement=("logloss_improvement","min"),
            mean_brier_improvement=("brier_improvement","mean"),
            worst_year_brier_improvement=("brier_improvement","min"),
            mean_history_coverage=("history_coverage","mean")))
    s["all_years_logloss_nonnegative"]=s.worst_year_logloss_improvement>=-1e-12
    best=(s.sort_values(["inning","mean_logloss_improvement"],ascending=[True,False])
          .groupby("inning",as_index=False).head(1).sort_values("inning"))
    return s,best


def summarize_share(f):
    s=(f.groupby(["window","weight","inning"],as_index=False)
       .agg(mean_mse_improvement=("mse_improvement","mean"),
            worst_year_mse_improvement=("mse_improvement","min"),
            mean_mae_improvement=("mae_improvement","mean"),
            worst_year_mae_improvement=("mae_improvement","min"),
            mean_history_coverage=("history_coverage","mean")))
    s["all_years_mse_nonnegative"]=s.worst_year_mse_improvement>=-1e-12
    best=(s.sort_values(["inning","mean_mse_improvement"],ascending=[True,False])
          .groupby("inning",as_index=False).head(1).sort_values("inning"))
    return s,best


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--matrix",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.matrix)
    x["season"]=pd.to_numeric(x.season,errors="coerce").astype(int)
    x["inning"]=pd.to_numeric(x.inning,errors="coerce").astype(int)
    if set(x.season.unique())!={2021,2022,2023,2024}: raise RuntimeError("development seasons must be exactly 2021-2024")
    if (x.season>=2025).any(): raise RuntimeError("2025 holdout leakage")
    if set(x.inning.unique())!=set(range(1,10)): raise RuntimeError("I1-I9 incomplete")
    required=[]
    for w in WINDOWS: required += [f"{w}_starter_begin_rate",f"{w}_starter_pa_share_mean"]
    missing=[c for c in required if c not in x.columns]
    if missing: raise RuntimeError(f"missing M2 raw-rate columns: {missing}")

    bf=evaluate_binary(x); bs,bb=summarize_binary(bf)
    sf=evaluate_share(x); ss,sb=summarize_share(sf)
    bf.to_csv(a.output_dir/"m2_calibrated_begin_folds.csv",index=False)
    bs.to_csv(a.output_dir/"m2_calibrated_begin_summary.csv",index=False)
    bb.to_csv(a.output_dir/"m2_calibrated_begin_best_by_inning.csv",index=False)
    sf.to_csv(a.output_dir/"m2_calibrated_share_folds.csv",index=False)
    ss.to_csv(a.output_dir/"m2_calibrated_share_summary.csv",index=False)
    sb.to_csv(a.output_dir/"m2_calibrated_share_best_by_inning.csv",index=False)

    manifest={
      "status":"PASS","architecture":"M2_pitcher_state_empirical_partial_pooling",
      "development_seasons":[2021,2022,2023,2024],"test_folds":TEST_YEARS,
      "holdout_season":2025,"holdout_opened":False,
      "candidate_windows":WINDOWS,"candidate_history_weights":WEIGHTS,
      "baseline_candidate_included":True,"raw_history_candidate_included":True,
      "pooling_formula":"baseline + weight * (raw_history - baseline)",
      "pooling_center":"training-fold inning empirical mean",
      "history_timing":"inherited strictly-prior-date M2 state matrix",
      "market_data_used":False,"automatic_production_promotion":False,
      "best_begin_calibration_by_inning":bb.to_dict("records"),
      "best_share_calibration_by_inning":sb.to_dict("records"),
      "note":"Empirical calibration research only. Selection uses 2022-2024 chronological folds; 2025 remains untouched."
    }
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()

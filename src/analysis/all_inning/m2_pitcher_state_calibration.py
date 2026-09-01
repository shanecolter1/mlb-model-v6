#!/usr/bin/env python3
"""Empirically calibrate M2 starter-state probabilities for I1-I9.

Development only: 2021-2024. 2025 is never loaded.

The raw M2 state-history model demonstrated useful rank/Brier information but
poor log loss because unsmoothed player histories can become extreme. This
script tests transparent empirical-Bayes partial pooling strengths rather than
assuming a shrinkage rule.

For starter-begins-inning (binary):
  p = (prior_successes + strength * fold_inning_baseline) / (prior_starts + strength)

For starter PA share (continuous [0,1]):
  mu = (prior_share_sum + strength * fold_inning_mean) / (prior_starts + strength)

Window and strength are selected only on chronological 2022/2023/2024 folds.
A baseline-only candidate is included, so player history is retained only when
it beats the training-fold inning baseline. No production promotion occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = ["season", "30d", "90d", "365d"]
STRENGTHS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
TEST_YEARS = [2022, 2023, 2024]
EPS = 1e-12


def logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def mae(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(np.mean(np.abs(p - y)))


def prior_cols(window):
    prefix = "season" if window == "season" else window
    return f"{prefix}_starts_prior", f"{prefix}_begins_prior", f"{prefix}_share_prior"


def evaluate_binary(x):
    rows = []
    for inning in range(2, 10):
        for year in TEST_YEARS:
            tr = x[(x.season < year) & (x.inning == inning)].copy()
            te = x[(x.season == year) & (x.inning == inning)].copy()
            base = float(tr.starter_begins_inning.mean())
            y = te.starter_begins_inning.to_numpy(float)
            p0 = np.full(len(te), base)
            base_ll = logloss(y, p0); base_br = brier(y, p0)
            rows.append({"target":"starter_begins_inning","window":"baseline_only","strength":np.inf,
                         "inning":inning,"test_year":year,"n_test":len(te),"history_coverage":0.0,
                         "baseline_logloss":base_ll,"model_logloss":base_ll,
                         "baseline_brier":base_br,"model_brier":base_br,
                         "logloss_improvement":0.0,"brier_improvement":0.0})
            for w in WINDOWS:
                ncol, scol, _ = prior_cols(w)
                n = pd.to_numeric(te[ncol], errors="coerce").to_numpy(float)
                s = pd.to_numeric(te[scol], errors="coerce").to_numpy(float)
                valid = np.isfinite(n) & np.isfinite(s) & (n > 0)
                for strength in STRENGTHS:
                    p = np.full(len(te), base, dtype=float)
                    p[valid] = (s[valid] + strength * base) / (n[valid] + strength)
                    ll = logloss(y,p); br = brier(y,p)
                    rows.append({"target":"starter_begins_inning","window":w,"strength":strength,
                                 "inning":inning,"test_year":year,"n_test":len(te),
                                 "history_coverage":float(valid.mean()),"baseline_logloss":base_ll,
                                 "model_logloss":ll,"baseline_brier":base_br,"model_brier":br,
                                 "logloss_improvement":base_ll-ll,"brier_improvement":base_br-br})
    return pd.DataFrame(rows)


def evaluate_share(x):
    rows = []
    for inning in range(1, 10):
        for year in TEST_YEARS:
            tr = x[(x.season < year) & (x.inning == inning)].copy()
            te = x[(x.season == year) & (x.inning == inning)].copy()
            base = float(tr.starter_pa_share.mean())
            y = te.starter_pa_share.to_numpy(float)
            p0 = np.full(len(te), base)
            base_mse = brier(y,p0); base_mae = mae(y,p0)
            rows.append({"target":"starter_pa_share","window":"baseline_only","strength":np.inf,
                         "inning":inning,"test_year":year,"n_test":len(te),"history_coverage":0.0,
                         "baseline_mse":base_mse,"model_mse":base_mse,
                         "baseline_mae":base_mae,"model_mae":base_mae,
                         "mse_improvement":0.0,"mae_improvement":0.0})
            for w in WINDOWS:
                ncol, _, shcol = prior_cols(w)
                n = pd.to_numeric(te[ncol], errors="coerce").to_numpy(float)
                ss = pd.to_numeric(te[shcol], errors="coerce").to_numpy(float)
                valid = np.isfinite(n) & np.isfinite(ss) & (n > 0)
                for strength in STRENGTHS:
                    p = np.full(len(te), base, dtype=float)
                    p[valid] = (ss[valid] + strength * base) / (n[valid] + strength)
                    p = np.clip(p,0,1)
                    mse = brier(y,p); ma = mae(y,p)
                    rows.append({"target":"starter_pa_share","window":w,"strength":strength,
                                 "inning":inning,"test_year":year,"n_test":len(te),
                                 "history_coverage":float(valid.mean()),"baseline_mse":base_mse,
                                 "model_mse":mse,"baseline_mae":base_mae,"model_mae":ma,
                                 "mse_improvement":base_mse-mse,"mae_improvement":base_mae-ma})
    return pd.DataFrame(rows)


def summarize_binary(f):
    s=(f.groupby(["window","strength","inning"],dropna=False,as_index=False)
       .agg(mean_logloss_improvement=("logloss_improvement","mean"),
            worst_year_logloss_improvement=("logloss_improvement","min"),
            mean_brier_improvement=("brier_improvement","mean"),
            worst_year_brier_improvement=("brier_improvement","min"),
            mean_history_coverage=("history_coverage","mean")))
    s["all_years_logloss_nonnegative"] = s.worst_year_logloss_improvement >= -1e-12
    best=(s.sort_values(["inning","mean_logloss_improvement"],ascending=[True,False])
          .groupby("inning",as_index=False).head(1).sort_values("inning"))
    return s,best


def summarize_share(f):
    s=(f.groupby(["window","strength","inning"],dropna=False,as_index=False)
       .agg(mean_mse_improvement=("mse_improvement","mean"),
            worst_year_mse_improvement=("mse_improvement","min"),
            mean_mae_improvement=("mae_improvement","mean"),
            worst_year_mae_improvement=("mae_improvement","min"),
            mean_history_coverage=("history_coverage","mean")))
    s["all_years_mse_nonnegative"] = s.worst_year_mse_improvement >= -1e-12
    best=(s.sort_values(["inning","mean_mse_improvement"],ascending=[True,False])
          .groupby("inning",as_index=False).head(1).sort_values("inning"))
    return s,best


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--matrix",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(a.matrix)
    seasons=set(pd.to_numeric(x.season,errors="coerce").dropna().astype(int).unique())
    if seasons != {2021,2022,2023,2024}: raise RuntimeError(f"expected development seasons only, got {seasons}")
    if (pd.to_numeric(x.season,errors="coerce")>=2025).any(): raise RuntimeError("2025 holdout leakage")
    if set(pd.to_numeric(x.inning,errors="coerce").dropna().astype(int).unique()) != set(range(1,10)): raise RuntimeError("I1-I9 incomplete")

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
      "candidate_windows":WINDOWS,"candidate_prior_strengths":STRENGTHS,
      "baseline_candidate_included":True,
      "pooling_center":"training-fold inning empirical mean",
      "history_timing":"inherited strictly-prior-date M2 state matrix",
      "market_data_used":False,"automatic_production_promotion":False,
      "best_begin_calibration_by_inning":bb.replace({np.inf:"baseline_only"}).to_dict("records"),
      "best_share_calibration_by_inning":sb.replace({np.inf:"baseline_only"}).to_dict("records"),
      "note":"Empirical calibration research only. Selection uses 2022-2024 chronological folds; 2025 remains untouched."
    }
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__": main()

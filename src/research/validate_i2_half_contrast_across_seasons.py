#!/usr/bin/env python3
"""Validate a zero-sum top/bottom I2 logit contrast across seasons.

The training audit estimates separate top and bottom calibration offsets.
This script removes their common component and transfers only the half-specific
contrast to a later season:
  top    <- logit(p_top)    - h
  bottom <- logit(p_bottom) + h
where h = (bottom_offset - top_offset) / 2.

The common calibration level is intentionally left untouched for the final
full-I2 calibration layer.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

MATCHED="RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT"


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--train-audit",type=Path,required=True)
    p.add_argument("--test-replay",type=Path,required=True)
    p.add_argument("--test-inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=10000)
    return p.parse_args()


def loss(y,p):
    p=min(1-1e-12,max(1e-12,float(p)))
    return (p-y)**2, -(y*math.log(p)+(1-y)*math.log1p(-p))


def metric(rows,pfield,yfield):
    p=np.array([r[pfield] for r in rows],float)
    y=np.array([r[yfield] for r in rows],float)
    return {
        "n":len(rows),
        "mean_prediction":float(p.mean()),
        "observed":float(y.mean()),
        "actual_minus_predicted":float((y-p).mean()),
        "brier":float(np.mean((p-y)**2)),
        "logloss":float(np.mean(-y*np.log(np.clip(p,1e-12,1-1e-12))-(1-y)*np.log1p(-np.clip(p,1e-12,1-1e-12)))),
    }


def clustered_deltas(rows,draws):
    by=defaultdict(lambda:np.zeros(5))
    for r in rows:
        bt0,lt0=loss(r["y_top"],r["p_top"])
        bt1,lt1=loss(r["y_top"],r["p_top_adj"])
        bb0,lb0=loss(r["y_bottom"],r["p_bottom"])
        bb1,lb1=loss(r["y_bottom"],r["p_bottom_adj"])
        full_adj=(1-r["p_top_adj"])*(1-r["p_bottom_adj"])
        f0,fl0=loss(r["y_under"],r["p_under"])
        f1,fl1=loss(r["y_under"],full_adj)
        x=by[r["date"]]
        x += [
            (bt1-bt0)+(bb1-bb0),
            (lt1-lt0)+(lb1-lb0),
            f1-f0,
            fl1-fl0,
            1,
        ]
    arr=np.array(list(by.values()),float)
    rng=np.random.default_rng(20260928)
    boot=np.empty((draws,4))
    for i in range(draws):
        s=arr[rng.integers(0,len(arr),size=len(arr))].sum(axis=0)
        boot[i]=[s[0]/(2*s[4]),s[1]/(2*s[4]),s[2]/s[4],s[3]/s[4]]
    point=[
        arr[:,0].sum()/(2*arr[:,4].sum()),
        arr[:,1].sum()/(2*arr[:,4].sum()),
        arr[:,2].sum()/arr[:,4].sum(),
        arr[:,3].sum()/arr[:,4].sum(),
    ]
    names=["half_brier","half_logloss","full_i2_brier","full_i2_logloss"]
    return {
        name:{
            "delta_adjusted_minus_raw":float(point[i]),
            "ci95_date_cluster":[float(x) for x in np.quantile(boot[:,i],[.025,.975])],
        }
        for i,name in enumerate(names)
    }


def main():
    a=args()
    train=json.loads(a.train_audit.read_text())
    replay=json.loads(a.test_replay.read_text())
    inputs=json.loads(a.test_inputs.read_text())
    if any(x.get("market_inputs_used") is not False for x in (train,replay,inputs)):
        raise ValueError("Market contamination")
    train_season=int(train["season"]); test_season=int(replay["season"])
    if test_season <= train_season:
        raise ValueError("Test season must follow train season")

    fit=train["matched_home_venue"]["fits"]["half_offset"]
    top_offset=float(fit["derived_offsets"]["top"])
    bottom_offset=float(fit["derived_offsets"]["bottom"])
    common=(top_offset+bottom_offset)/2
    h=(bottom_offset-top_offset)/2

    games={g["gid"]:g for g in inputs["games"]}
    rows=[]
    for p in replay["predictions"]:
        if p["park_status"] != MATCHED:
            continue
        g=games[p["gid"]]
        pt=float(p["top2_score_probability"])
        pb=float(p["bottom2_score_probability"])
        rows.append({
            "gid":p["gid"],"date":p["date"],
            "p_top":pt,
            "p_bottom":pb,
            "p_top_adj":float(expit(logit(np.clip(pt,1e-9,1-1e-9))-h)),
            "p_bottom_adj":float(expit(logit(np.clip(pb,1e-9,1-1e-9))+h)),
            "y_top":int(g["observed"]["top2_runs"]>0),
            "y_bottom":int(g["observed"]["bottom2_runs"]>0),
            "p_under":float(p["raw_under05"]),
            "y_under":int(p["observed_under05"]),
        })
    if not rows:
        raise ValueError("No matched test games")

    raw_top=metric(rows,"p_top","y_top")
    adj_top=metric(rows,"p_top_adj","y_top")
    raw_bottom=metric(rows,"p_bottom","y_bottom")
    adj_bottom=metric(rows,"p_bottom_adj","y_bottom")
    full_adj=np.array([(1-r["p_top_adj"])*(1-r["p_bottom_adj"]) for r in rows])
    full_raw=np.array([r["p_under"] for r in rows])
    full_y=np.array([r["y_under"] for r in rows])
    result={
        "version":"i2-half-zero-sum-contrast-cross-season-v1",
        "market_inputs_used":False,
        "train_season":train_season,
        "test_season":test_season,
        "train_matched_games":train["matched_home_venue"]["n_games"],
        "test_matched_games":len(rows),
        "trained_offsets":{
            "top_offset":top_offset,
            "bottom_offset":bottom_offset,
            "common_component_removed":common,
            "zero_sum_half_contrast_h":h,
            "applied_top_logit_delta":-h,
            "applied_bottom_logit_delta":h,
        },
        "top":{"raw":raw_top,"adjusted":adj_top},
        "bottom":{"raw":raw_bottom,"adjusted":adj_bottom},
        "combined":{
            "deltas":clustered_deltas(rows,a.bootstrap),
            "full_i2_under":{
                "raw_mean_prediction":float(full_raw.mean()),
                "adjusted_mean_prediction":float(full_adj.mean()),
                "observed":float(full_y.mean()),
            },
        },
        "governance":{
            "calendar_month_used":False,
            "test_season_used_to_fit_contrast":False,
            "common_calibration_component_applied":False,
            "final_full_i2_calibration_left_separate":True,
            "production_changed":False,
            "historical_hyperparameter_overlap_warning":train_season==2024,
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

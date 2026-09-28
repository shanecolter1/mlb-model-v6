#!/usr/bin/env python3
"""Apply a historical half-inning calibration adjustment to a later season."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

MATCHED="RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT"


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--train-audit",type=Path,required=True)
    p.add_argument("--test-replay",type=Path,required=True)
    p.add_argument("--test-inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    return p.parse_args()


def loss(y,p):
    p=min(1-1e-12,max(1e-12,float(p)))
    return (p-y)**2, -(y*math.log(p)+(1-y)*math.log1p(-p))


def metrics(rows,pfield,yfield):
    n=len(rows)
    ps=np.array([r[pfield] for r in rows],float)
    ys=np.array([r[yfield] for r in rows],float)
    return {
        "n":n,
        "mean_prediction":float(ps.mean()),
        "observed":float(ys.mean()),
        "actual_minus_predicted":float((ys-ps).mean()),
        "brier":float(np.mean((ps-ys)**2)),
        "logloss":float(np.mean(-ys*np.log(np.clip(ps,1e-12,1-1e-12))-(1-ys)*np.log1p(-np.clip(ps,1e-12,1-1e-12)))),
    }


def bootstrap(rows,draws,seed):
    by_date=defaultdict(lambda:[0.0,0.0,0])
    for r in rows:
        b0,l0=loss(r["y_bottom"],r["p_bottom"])
        b1,l1=loss(r["y_bottom"],r["p_bottom_adj"])
        x=by_date[r["date"]]
        x[0]+=b1-b0; x[1]+=l1-l0; x[2]+=1
    clusters=np.array(list(by_date.values()),float)
    rng=np.random.default_rng(seed)
    out=np.empty((draws,2))
    for i in range(draws):
        s=clusters[rng.integers(0,len(clusters),size=len(clusters))].sum(axis=0)
        out[i]=[s[0]/s[2],s[1]/s[2]]
    return {
        "brier_delta":[float(x) for x in np.quantile(out[:,0],[.025,.975])],
        "logloss_delta":[float(x) for x in np.quantile(out[:,1],[.025,.975])],
    }


def full_metrics(rows,draws):
    raw=[]; adj=[]; ys=[]; dates=[]
    for r in rows:
        raw.append(r["p_under"])
        adj.append((1-r["p_top"])*(1-r["p_bottom_adj"]))
        ys.append(r["y_under"])
        dates.append(r["date"])
    raw=np.asarray(raw); adj=np.asarray(adj); ys=np.asarray(ys,float)
    b0=(raw-ys)**2; b1=(adj-ys)**2
    l0=-ys*np.log(np.clip(raw,1e-12,1-1e-12))-(1-ys)*np.log1p(-np.clip(raw,1e-12,1-1e-12))
    l1=-ys*np.log(np.clip(adj,1e-12,1-1e-12))-(1-ys)*np.log1p(-np.clip(adj,1e-12,1-1e-12))
    grouped=defaultdict(lambda:[0.0,0.0,0])
    for d,db,dl in zip(dates,b1-b0,l1-l0):
        x=grouped[d]; x[0]+=db; x[1]+=dl; x[2]+=1
    clusters=np.array(list(grouped.values()),float)
    rng=np.random.default_rng(1818)
    boot=np.empty((draws,2))
    for i in range(draws):
        s=clusters[rng.integers(0,len(clusters),size=len(clusters))].sum(axis=0)
        boot[i]=[s[0]/s[2],s[1]/s[2]]
    return {
        "raw_brier":float(b0.mean()),
        "adjusted_brier":float(b1.mean()),
        "brier_delta":float((b1-b0).mean()),
        "raw_logloss":float(l0.mean()),
        "adjusted_logloss":float(l1.mean()),
        "logloss_delta":float((l1-l0).mean()),
        "delta_ci95_date_cluster":{
            "brier":[float(x) for x in np.quantile(boot[:,0],[.025,.975])],
            "logloss":[float(x) for x in np.quantile(boot[:,1],[.025,.975])],
        },
    }


def main():
    a=parse_args()
    train=json.loads(a.train_audit.read_text())
    replay=json.loads(a.test_replay.read_text())
    inputs=json.loads(a.test_inputs.read_text())
    if any(x.get("market_inputs_used") is not False for x in (train,replay,inputs)):
        raise ValueError("Market contamination")
    train_season=int(train["season"]); test_season=int(replay["season"])
    if test_season <= train_season:
        raise ValueError("Test season must follow train season")
    offset=float(train["matched_home_venue"]["fits"]["bottom_only"]["coefficients"]["bottom_offset"]["coef"])
    games={g["gid"]:g for g in inputs["games"]}
    rows=[]
    for p in replay["predictions"]:
        if p["park_status"] != MATCHED:
            continue
        g=games[p["gid"]]
        pt=float(p["top2_score_probability"])
        pb=float(p["bottom2_score_probability"])
        pba=float(expit(logit(np.clip(pb,1e-9,1-1e-9))+offset))
        rows.append({
            "gid":p["gid"],"date":p["date"],
            "p_top":pt,"p_bottom":pb,"p_bottom_adj":pba,
            "y_top":int(g["observed"]["top2_runs"]>0),
            "y_bottom":int(g["observed"]["bottom2_runs"]>0),
            "p_under":float(p["raw_under05"]),
            "y_under":int(p["observed_under05"]),
        })
    if not rows:
        raise ValueError("No matched test games")
    raw=metrics(rows,"p_bottom","y_bottom")
    adj=metrics(rows,"p_bottom_adj","y_bottom")
    b0=np.mean([(r["p_bottom"]-r["y_bottom"])**2 for r in rows])
    b1=np.mean([(r["p_bottom_adj"]-r["y_bottom"])**2 for r in rows])
    ll0=np.mean([loss(r["y_bottom"],r["p_bottom"])[1] for r in rows])
    ll1=np.mean([loss(r["y_bottom"],r["p_bottom_adj"])[1] for r in rows])
    result={
        "version":"i2-half-adjustment-cross-season-validation-v1",
        "market_inputs_used":False,
        "train_season":train_season,
        "test_season":test_season,
        "training_source":"matched-home-venue bottom-only logit offset",
        "trained_bottom_logit_offset":offset,
        "test_matched_games":len(rows),
        "bottom":{
            "raw":raw,
            "adjusted":adj,
            "adjusted_minus_raw":{
                "brier":float(b1-b0),
                "logloss":float(ll1-ll0),
                "ci95_date_cluster":bootstrap(rows,a.bootstrap,1817),
            },
            "absolute_bias_reduction":float(abs(raw["actual_minus_predicted"])-abs(adj["actual_minus_predicted"])),
        },
        "full_i2_under_from_half_adjustment":full_metrics(rows,a.bootstrap),
        "governance":{
            "test_season_used_to_fit_offset":False,
            "calendar_month_used":False,
            "top_half_adjusted":False,
            "production_changed":False,
            "historical_hyperparameter_overlap_warning":train_season==2024,
            "eligible_for_direct_promotion":False,
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()

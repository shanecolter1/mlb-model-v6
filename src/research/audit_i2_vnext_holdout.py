#!/usr/bin/env python3
"""Final leakage-safe holdout audit for I2 vNext.

Compares the untouched 2025 point-in-time replay against a genuinely available
pregame baseline: the prior-season (2024) regular-season full-I2 Under 0.5 rate.

The script also reports discrimination and the later-2025 calibration-selection
comparison. It does not use markets and does not tune any model parameter.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--prior-retrosheet-zip",type=Path,required=True)
    p.add_argument("--prior-season",type=int,default=2024)
    p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def as_int(v):
    try:
        return int(v or 0)
    except (TypeError,ValueError):
        return 0


def prior_full_i2_under_rate(zip_path:Path, season:int):
    member=f"{season}plays.csv"
    games=defaultdict(lambda:[0,0])
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise RuntimeError(f"{zip_path} missing {member}")
        with zf.open(member) as raw:
            rows=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",newline=""))
            for row in rows:
                if row.get("gametype")!="regular":
                    continue
                if as_int(row.get("inning"))!=2:
                    continue
                gid=str(row.get("gid") or "").strip()
                if not gid:
                    continue
                half=as_int(row.get("top_bot"))
                if half not in (0,1):
                    continue
                games[gid][half]+=as_int(row.get("runs"))
    if not games:
        raise RuntimeError("No prior-season I2 games found")
    eligible=[v for v in games.values()]
    under=sum(1 for top,bot in eligible if top+bot==0)
    return {
        "season":season,
        "games":len(eligible),
        "under_games":under,
        "under_rate":under/len(eligible),
    }


def clip(p):
    return min(1-1e-12,max(1e-12,float(p)))


def metrics(y,p):
    y=np.asarray(y,dtype=float)
    p=np.asarray([clip(v) for v in p],dtype=float)
    return {
        "n":int(len(y)),
        "realized_under_rate":float(y.mean()),
        "predicted_under_mean":float(p.mean()),
        "brier":float(np.mean((p-y)**2)),
        "logloss":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
    }


def auc_rank(y,p):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    n1=int(y.sum()); n0=len(y)-n1
    if n1==0 or n0==0:
        return None
    order=np.argsort(p,kind="mergesort")
    ranks=np.empty(len(p),dtype=float)
    i=0
    while i<len(p):
        j=i+1
        while j<len(p) and p[order[j]]==p[order[i]]:
            j+=1
        avg=(i+1+j)/2
        ranks[order[i:j]]=avg
        i=j
    sum_pos=float(ranks[y==1].sum())
    return (sum_pos-n1*(n1+1)/2)/(n1*n0)


def deciles(rows):
    rows=sorted(rows,key=lambda r:float(r["raw_under05"]))
    chunks=np.array_split(np.arange(len(rows)),10)
    out=[]
    for k,idx in enumerate(chunks,1):
        vals=[rows[int(i)] for i in idx]
        out.append({
            "decile":k,
            "n":len(vals),
            "predicted_mean":float(np.mean([float(r["raw_under05"]) for r in vals])),
            "realized_under_rate":float(np.mean([int(r["observed_under05"]) for r in vals])),
        })
    return out


def main():
    args=parse_args()
    replay=json.loads(args.replay.read_text())
    cal=json.loads(args.calibration.read_text())
    if replay.get("market_inputs_used") is not False or cal.get("market_inputs_used") is not False:
        raise RuntimeError("Audit inputs must be market-isolated")
    if replay.get("observed_i2_state_used_as_predictor") is not False:
        raise RuntimeError("Observed I2 state leakage detected")
    if replay.get("point_in_time_player_refits") is not True:
        raise RuntimeError("Replay is not point-in-time")

    prior=prior_full_i2_under_rate(args.prior_retrosheet_zip,args.prior_season)
    rows=sorted(replay["predictions"],key=lambda r:(str(r["date"]),str(r["gid"])))
    y=np.asarray([int(r["observed_under05"]) for r in rows],dtype=int)
    p=np.asarray([float(r["raw_under05"]) for r in rows],dtype=float)
    baseline=np.full(len(rows),prior["under_rate"],dtype=float)

    split=str(cal["calibration_selection"]["split_date"])
    val=[r for r in rows if str(r["date"])>=split]
    yv=np.asarray([int(r["observed_under05"]) for r in val],dtype=int)
    pv=np.asarray([float(r["raw_under05"]) for r in val],dtype=float)
    baseline_v=np.full(len(val),prior["under_rate"],dtype=float)

    raw_all=metrics(y,p)
    prior_all=metrics(y,baseline)
    raw_val=metrics(yv,pv)
    prior_val=metrics(yv,baseline_v)
    sel=cal["calibration_selection"]

    corr=float(np.corrcoef(p,y)[0,1]) if np.std(p)>0 and np.std(y)>0 else None
    d=deciles(rows)
    decile_real=np.asarray([x["realized_under_rate"] for x in d],dtype=float)
    decile_index=np.arange(1,11,dtype=float)
    decile_corr=float(np.corrcoef(decile_index,decile_real)[0,1]) if np.std(decile_real)>0 else None

    payload={
        "version":"i2-vnext-final-holdout-audit-v1",
        "market_inputs_used":False,
        "replay_version":replay.get("version"),
        "replay_games":len(rows),
        "trials_per_game":replay.get("trials_per_game"),
        "point_in_time_player_refits":replay.get("point_in_time_player_refits"),
        "i1_state_mode":replay.get("i1_state_mode"),
        "park_match_rate":replay.get("park_match_rate"),
        "prior_season_baseline":prior,
        "all_2025":{
            "vnext_raw":raw_all,
            "prior_season_constant":prior_all,
            "delta_vnext_minus_prior":{
                "brier":raw_all["brier"]-prior_all["brier"],
                "logloss":raw_all["logloss"]-prior_all["logloss"],
            },
        },
        "later_2025_validation":{
            "split_date":split,
            "vnext_identity":raw_val,
            "prior_season_constant":prior_val,
            "selected_calibration_validation":{
                "chosen":sel["chosen"],
                "identity":sel["identity_validation"],
                "sigmoid":sel["sigmoid_validation"],
            },
        },
        "discrimination":{
            "auc":auc_rank(y,p),
            "raw_probability_outcome_correlation":corr,
            "raw_probability_std":float(np.std(p)),
            "raw_probability_min":float(np.min(p)),
            "raw_probability_max":float(np.max(p)),
            "decile_index_vs_realized_rate_correlation":decile_corr,
            "deciles":d,
        },
        "governance":{
            "baseline_available_before_2025":True,
            "2025_realized_prevalence_not_used_as_baseline_probability":True,
            "markets_used":False,
            "model_parameters_changed":False,
        },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()

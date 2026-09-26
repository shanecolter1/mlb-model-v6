#!/usr/bin/env python3
"""Compare league-average vs point-in-time player-specific I1 state generators.

The downstream I2 model is held fixed. Reports paired Brier/log-loss differences,
calibration, and a deterministic paired bootstrap confidence interval.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def clip(p: float) -> float:
    return min(1 - 1e-9, max(1e-9, float(p)))


def loss_rows(rows):
    out = {}
    for r in rows:
        gid = str(r["gid"])
        y = int(r["observed_under05"])
        p = clip(r["raw_under05"])
        out[gid] = {
            "y": y,
            "p": p,
            "brier": (p - y) ** 2,
            "logloss": -(y * math.log(p) + (1-y) * math.log(1-p)),
        }
    return out


def arm_metrics(vals):
    ys = np.asarray([v["y"] for v in vals], dtype=float)
    ps = np.asarray([v["p"] for v in vals], dtype=float)
    return {
        "n": int(len(vals)),
        "realized_under_rate": float(np.mean(ys)),
        "predicted_under_mean": float(np.mean(ps)),
        "calibration_bias_pred_minus_actual": float(np.mean(ps) - np.mean(ys)),
        "brier": float(np.mean([(v["p"]-v["y"])**2 for v in vals])),
        "logloss": float(np.mean([v["logloss"] for v in vals])),
    }


def ece(vals, bins=10):
    ps=np.asarray([v["p"] for v in vals],dtype=float)
    ys=np.asarray([v["y"] for v in vals],dtype=float)
    order=np.argsort(ps)
    splits=np.array_split(order,bins)
    total=len(vals)
    e=0.0
    table=[]
    for idx in splits:
        if len(idx)==0:
            continue
        mp=float(np.mean(ps[idx])); my=float(np.mean(ys[idx]))
        e += len(idx)/total * abs(mp-my)
        table.append({"n":int(len(idx)),"pred":mp,"actual":my,"gap":mp-my})
    return float(e), table


def bootstrap(d_brier, d_logloss, reps=5000, seed=730):
    rng=np.random.default_rng(seed)
    n=len(d_brier)
    mb=[]; ml=[]
    for _ in range(reps):
        idx=rng.integers(0,n,n)
        mb.append(float(np.mean(d_brier[idx])))
        ml.append(float(np.mean(d_logloss[idx])))
    def ci(x):
        return [float(np.quantile(x,0.025)),float(np.quantile(x,0.975))]
    return {
        "reps": reps,
        "seed": seed,
        "delta_brier_ci95": ci(np.asarray(mb)),
        "delta_logloss_ci95": ci(np.asarray(ml)),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--league",type=Path,required=True)
    p.add_argument("--player",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()

    league=json.loads(a.league.read_text())
    player=json.loads(a.player.read_text())
    if league.get("i1_state_mode")!="league":
        raise RuntimeError("League arm mislabeled")
    if player.get("i1_state_mode")!="player_asof":
        raise RuntimeError("Player arm mislabeled")

    L=loss_rows(league["predictions"])
    P=loss_rows(player["predictions"])
    ids=sorted(set(L)&set(P))
    if len(ids)!=len(L) or len(ids)!=len(P):
        raise RuntimeError("Arms do not contain identical game sets")
    for gid in ids:
        if L[gid]["y"]!=P[gid]["y"]:
            raise RuntimeError(f"Outcome mismatch for {gid}")

    lv=[L[g] for g in ids]; pv=[P[g] for g in ids]
    lm=arm_metrics(lv); pm=arm_metrics(pv)
    lece,lrel=ece(lv); pece,prel=ece(pv)
    lm["ece_decile"]=lece; pm["ece_decile"]=pece

    d_b=np.asarray([P[g]["brier"]-L[g]["brier"] for g in ids],dtype=float)
    d_l=np.asarray([P[g]["logloss"]-L[g]["logloss"] for g in ids],dtype=float)
    boot=bootstrap(d_b,d_l)

    mean_db=float(np.mean(d_b)); mean_dl=float(np.mean(d_l))
    player_better = mean_db < 0 and mean_dl < 0
    statistically_clear = (
        boot["delta_brier_ci95"][1] < 0 and boot["delta_logloss_ci95"][1] < 0
    )
    league_clear = (
        boot["delta_brier_ci95"][0] > 0 and boot["delta_logloss_ci95"][0] > 0
    )
    if statistically_clear:
        verdict="PLAYER_ASOF_PASS"
    elif league_clear:
        verdict="LEAGUE_AVERAGE_PASS"
    elif player_better:
        verdict="PLAYER_ASOF_DIRECTIONAL_ONLY"
    else:
        verdict="LEAGUE_AVERAGE_OR_NO_CLEAR_GAIN"

    payload={
        "version":"i2-i1-state-ab-v1",
        "games":len(ids),
        "comparison":"player_asof minus league; negative delta favors player_asof",
        "league":lm,
        "player_asof":pm,
        "delta_player_minus_league":{
            "brier":mean_db,
            "logloss":mean_dl,
            "predicted_under_mean":pm["predicted_under_mean"]-lm["predicted_under_mean"],
            "ece_decile":pm["ece_decile"]-lm["ece_decile"],
        },
        "paired_bootstrap":boot,
        "verdict":verdict,
        "promotion_rule":"Promote player-specific I1 only if it improves both Brier and log loss; require both paired 95% CIs below zero for a statistically clear pass.",
        "reliability":{"league":lrel,"player_asof":prel},
        "market_inputs_used":False,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))


if __name__=="__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

def clip(p): return min(1-1e-12,max(1e-12,float(p)))
def metrics(rows,key):
    y=np.asarray([int(r["observed_under05"]) for r in rows],dtype=float)
    p=np.asarray([clip(r[key]) for r in rows],dtype=float)
    return {
      "n":int(len(rows)),
      "realized_under_rate":float(y.mean()),
      "predicted_under_mean":float(p.mean()),
      "calibration_bias_pred_minus_actual":float(p.mean()-y.mean()),
      "brier":float(np.mean((p-y)**2)),
      "logloss":float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p)))),
    }
def ece(rows,key,bins=10):
    y=np.asarray([int(r["observed_under05"]) for r in rows],dtype=float)
    p=np.asarray([clip(r[key]) for r in rows],dtype=float)
    order=np.argsort(p); splits=np.array_split(order,bins); total=len(rows); acc=0.0
    for idx in splits:
      if len(idx): acc += len(idx)/total*abs(float(p[idx].mean()-y[idx].mean()))
    return float(acc)
def bootstrap(db,dl,reps=10000,seed=730):
    rng=np.random.default_rng(seed); n=len(db); mb=np.empty(reps); ml=np.empty(reps)
    for i in range(reps):
      idx=rng.integers(0,n,n); mb[i]=db[idx].mean(); ml[i]=dl[idx].mean()
    return {
      "reps":reps,"seed":seed,
      "delta_brier_ci95":[float(np.quantile(mb,.025)),float(np.quantile(mb,.975))],
      "delta_logloss_ci95":[float(np.quantile(ml,.025)),float(np.quantile(ml,.975))],
      "p_delta_brier_lt_0":float(np.mean(mb<0)),
      "p_delta_logloss_lt_0":float(np.mean(ml<0)),
    }
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    src=json.loads(a.input.read_text()); rows=src["predictions"]
    lm=metrics(rows,"league_under05"); pm=metrics(rows,"player_asof_under05")
    lm["ece_decile"]=ece(rows,"league_under05"); pm["ece_decile"]=ece(rows,"player_asof_under05")
    db=[]; dl=[]
    for r in rows:
      y=int(r["observed_under05"]); lp=clip(r["league_under05"]); pp=clip(r["player_asof_under05"])
      db.append((pp-y)**2-(lp-y)**2)
      dl.append(-(y*math.log(pp)+(1-y)*math.log(1-pp)) + (y*math.log(lp)+(1-y)*math.log(1-lp)))
    db=np.asarray(db); dl=np.asarray(dl); boot=bootstrap(db,dl)
    dB=float(db.mean()); dL=float(dl.mean())
    if boot["delta_brier_ci95"][1] < 0 and boot["delta_logloss_ci95"][1] < 0: verdict="PLAYER_ASOF_PASS"
    elif boot["delta_brier_ci95"][0] > 0 and boot["delta_logloss_ci95"][0] > 0: verdict="LEAGUE_AVERAGE_PASS"
    elif dB < 0 and dL < 0: verdict="PLAYER_ASOF_DIRECTIONAL_ONLY"
    else: verdict="LEAGUE_AVERAGE_OR_NO_CLEAR_GAIN"
    out={
      "version":"i2-i1-state-ab-exact-result-v1","games":len(rows),"market_inputs_used":False,
      "method":src["method"],"downstream_i2_model":src.get("downstream_i2_model"),
      "league":lm,"player_asof":pm,
      "delta_player_minus_league":{
        "brier":dB,"logloss":dL,
        "predicted_under_mean":pm["predicted_under_mean"]-lm["predicted_under_mean"],
        "ece_decile":pm["ece_decile"]-lm["ece_decile"],
      },
      "paired_bootstrap":boot,
      "mean_start_slot_total_variation":float(np.mean([(r["away_start_slot_tv"]+r["home_start_slot_tv"])/2 for r in rows])),
      "verdict":verdict,
      "promotion_rule":"Promote player-specific I1 only if both Brier and log loss improve; statistically clear PASS requires both paired 95% CIs below zero."
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=="__main__": main()

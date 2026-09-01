#!/usr/bin/env python3
"""Validate and materialize joint starter-state x batting-start-slot probabilities.

Development only: 2021-2024. 2025 is never loaded.

The preceding dependence audit rejected simple factorization of M2 pitcher state
and M3 batter start slot. This script therefore tests an empirically weighted
slot-state residual on top of the calibrated player-specific M2 starter-state
probability.

For an inning i and possible start slot s:
  p_player = calibrated M2 probability for the starter beginning inning i
  slot_rate = training-fold empirical P(starter begins | inning i, slot s)
  base = training-fold empirical P(starter begins | inning i)
  p(state=starter | s) = p_player + gamma * (slot_rate - base)

Gamma is selected from a development grid. Gamma=0 is the factorized model.
The M3 start-slot distribution is then multiplied by the conditional M2 state
probability to form the 18-cell joint state distribution for each half-inning.

This is a state-identification layer only; it does not predict runs or promote a
production model. Realized starter and lineup identities remain research/Tier-B.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

TEST_YEARS=[2022,2023,2024]
GAMMAS=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.75,1.0]
EPS=1e-8


def ll(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
def br(y,p): return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))


def pair(m2,m3):
    a=m2.copy(); b=m3[["game_id","game_date","season","team_id","inning","batting_order_slot"]].rename(columns={"team_id":"batting_team_id"})
    x=a.merge(b,on=["game_id","game_date","season","inning"],how="inner")
    x=x[x.pitching_team_id!=x.batting_team_id].copy()
    key=["game_id","inning","pitching_team_id"]
    if x.duplicated(key).any(): raise RuntimeError("nonunique paired half-inning")
    return x


def selected_specs(cal):
    out={}
    for _,r in cal.iterrows(): out[int(r.inning)]={"window":str(r.window),"weight":float(r.weight)}
    return out


def fold_eval(x,specs):
    rows=[]
    for year in TEST_YEARS:
        tr=x[x.season<year]; te=x[x.season==year]
        for inning in range(2,10):
            a=tr[tr.inning==inning]; b=te[te.inning==inning].copy()
            if len(a)<100 or len(b)<100: continue
            base=float(a.starter_begins_inning.mean())
            slot_rate=a.groupby("batting_order_slot").starter_begins_inning.mean().to_dict()
            sp=specs[inning]; col=f"{sp['window']}_starter_begin_rate"
            raw=pd.to_numeric(b[col],errors="coerce").to_numpy(float); valid=np.isfinite(raw)
            p_player=np.full(len(b),base,float); p_player[valid]=base+sp['weight']*(raw[valid]-base); p_player=np.clip(p_player,EPS,1-EPS)
            y=b.starter_begins_inning.to_numpy(float); slots=b.batting_order_slot.astype(int).to_numpy()
            for gamma in GAMMAS:
                delta=np.array([float(slot_rate.get(int(s),base))-base for s in slots])
                p=np.clip(p_player+gamma*delta,EPS,1-EPS)
                rows.append({"test_year":year,"inning":inning,"gamma":gamma,"n_test":len(b),"m2_window":sp['window'],"m2_weight":sp['weight'],
                             "factorized_logloss":ll(y,p_player),"conditional_logloss":ll(y,p),"logloss_improvement":ll(y,p_player)-ll(y,p),
                             "factorized_brier":br(y,p_player),"conditional_brier":br(y,p),"brier_improvement":br(y,p_player)-br(y,p)})
    return pd.DataFrame(rows)


def choose(f):
    s=(f.groupby(["inning","gamma"],as_index=False).agg(mean_logloss_improvement=("logloss_improvement","mean"),worst_year_logloss_improvement=("logloss_improvement","min"),mean_brier_improvement=("brier_improvement","mean"),worst_year_brier_improvement=("brier_improvement","min")))
    s["all_years_logloss_nonnegative"]=s.worst_year_logloss_improvement>=-1e-12
    best=(s.sort_values(["inning","mean_logloss_improvement"],ascending=[True,False]).groupby("inning",as_index=False).head(1).sort_values("inning"))
    return s,best


def materialize(x,m3,specs,best):
    """Create joint 18-cell probabilities for 2022-2024 using prior seasons.

    M3 start-slot probabilities use prior development seasons, matching the M3
    walk-forward state definition. Conditional slot-state rates also use prior
    seasons. M2 player history is already strictly prior-date.
    """
    gammas={int(r.inning):float(r.gamma) for _,r in best.iterrows()}
    pslot_cols=[f"p_start_slot_{s}" for s in range(1,10)]
    m3p=m3[["game_id","inning","team_id","prior_train_n"]+pslot_cols].rename(columns={"team_id":"batting_team_id"})
    z=x.merge(m3p,on=["game_id","inning","batting_team_id"],how="left",validate="one_to_one")
    z["p_starter_unconditional"]=np.nan
    for s in range(1,10):
        z[f"p_slot{s}_starter"]=np.nan; z[f"p_slot{s}_bullpen"]=np.nan
    for year in TEST_YEARS:
        tr=x[x.season<year]
        mask=z.season==year
        for inning in range(1,10):
            idx=z.index[mask&(z.inning==inning)]
            if len(idx)==0: continue
            a=tr[tr.inning==inning]; base=float(a.starter_begins_inning.mean())
            if inning==1:
                p_player=np.full(len(idx),base); gamma=0.0
            else:
                sp=specs[inning]; col=f"{sp['window']}_starter_begin_rate"; raw=pd.to_numeric(z.loc[idx,col],errors="coerce").to_numpy(float)
                p_player=np.full(len(idx),base); valid=np.isfinite(raw); p_player[valid]=base+sp['weight']*(raw[valid]-base); gamma=gammas[inning]
            p_player=np.clip(p_player,EPS,1-EPS); z.loc[idx,"p_starter_unconditional"]=p_player
            rates=a.groupby("batting_order_slot").starter_begins_inning.mean().to_dict()
            for s in range(1,10):
                ps=pd.to_numeric(z.loc[idx,f"p_start_slot_{s}"],errors="coerce").to_numpy(float)
                cond=np.clip(p_player+gamma*(float(rates.get(s,base))-base),EPS,1-EPS)
                z.loc[idx,f"p_slot{s}_starter"]=ps*cond; z.loc[idx,f"p_slot{s}_bullpen"]=ps*(1-cond)
    jcols=[c for c in z.columns if c.startswith("p_slot") and (c.endswith("_starter") or c.endswith("_bullpen"))]
    valid=z.prior_train_n.fillna(0)>0
    mass=z.loc[valid,jcols].sum(axis=1)
    if not ((mass-1).abs()<1e-8).all(): raise RuntimeError(f"joint probability mass failure max={float((mass-1).abs().max())}")
    return z


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--m2",type=Path,required=True); ap.add_argument("--m3",type=Path,required=True); ap.add_argument("--m2-calibration",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    m2=pd.read_parquet(a.m2); m3=pd.read_parquet(a.m3); cal=pd.read_csv(a.m2_calibration)
    for d in (m2,m3): d["season"]=pd.to_numeric(d.season).astype(int); d["inning"]=pd.to_numeric(d.inning).astype(int)
    if set(m2.season.unique())!={2021,2022,2023,2024} or set(m3.season.unique())!={2021,2022,2023,2024}: raise RuntimeError("development seasons incomplete")
    if (m2.season>=2025).any() or (m3.season>=2025).any(): raise RuntimeError("2025 leakage")
    specs=selected_specs(cal); x=pair(m2,m3); folds=fold_eval(x,specs); summary,best=choose(folds); joint=materialize(x,m3,specs,best)
    folds.to_csv(a.output_dir/"m4_joint_state_gamma_folds.csv",index=False); summary.to_csv(a.output_dir/"m4_joint_state_gamma_summary.csv",index=False); best.to_csv(a.output_dir/"m4_joint_state_best_gamma_by_inning.csv",index=False); joint.to_parquet(a.output_dir/"m4_joint_starter_slot_state_matrix.parquet",index=False)
    manifest={"status":"PASS","architecture":"M4_joint_starter_slot_state","development_seasons":[2021,2022,2023,2024],"test_folds":TEST_YEARS,"holdout_season":2025,"holdout_opened":False,
              "m2_calibration_source":"all-inning-m2-pitcher-state-calibration-2021-2024","candidate_dependence_weights":GAMMAS,"factorized_candidate_included":True,
              "market_data_used":False,"automatic_production_promotion":False,"best_dependence_weight_by_inning":best.to_dict("records"),"joint_state_rows":int(len(joint)),
              "note":"Joint state distribution only. Bullpen branch still requires reliever identity/class mixture before matchup assembly."}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()

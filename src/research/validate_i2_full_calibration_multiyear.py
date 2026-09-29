#!/usr/bin/env python3
"""Multiyear final full-I2 calibration for the vNext betting probability.

One and only one final calibration layer is considered after all component
modeling. No half-inning calibration is applied.

Development:
  2022-2025 season-specific PRIMARY_HOME_VENUE games.
  Leave-one-season-out (LOSO) selection by full-I2 log loss; Brier is reported.

Candidates:
  identity
  offset: q = logistic(logit(p) + lambda*c)
  affine: q = logistic(lambda*a + [1+lambda*(b-1)]*logit(p))

For offset and affine, lambda in [0,1] is selected only from 2022-2025 LOSO
performance. Lambda=0 is the identity curve, so shrinkage is estimated rather
than manually chosen.

Replication:
  2026 is never used to fit coefficients or shrinkage. It was already inspected
  in the preceding half-calibration research, so it is explicitly not claimed
  as a pristine holdout. A non-identity curve is promotable only if it improves
  BOTH log loss and Brier on 2026 replication; otherwise final curve = identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

DEV_YEARS=(2022,2023,2024,2025)
REPLICATION_YEAR=2026
ALL_YEARS=DEV_YEARS+(REPLICATION_YEAR,)
EPS=1e-9


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=10000)
    return p.parse_args()


def clip(p):
    return np.clip(np.asarray(p,float),EPS,1-EPS)


def metric(y,p):
    y=np.asarray(y,float); p=clip(p)
    return {
        "n":int(len(y)),
        "observed":float(y.mean()),
        "mean_prediction":float(p.mean()),
        "actual_minus_predicted":float((y-p).mean()),
        "brier":float(np.mean((p-y)**2)),
        "logloss":float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
    }


def losses(y,p):
    y=np.asarray(y,float); p=clip(p)
    return (p-y)**2,-y*np.log(p)-(1-y)*np.log1p(-p)


def read_year(root,year):
    replay=json.loads((root/f"replay_{year}_precision_10000.json").read_text())
    inputs=json.loads((root/f"replay_{year}_inputs.json").read_text())
    venue=json.loads((root/f"home_venue_{year}.json").read_text())
    for name,payload in (("replay",replay),("inputs",inputs),("venue",venue)):
        if payload.get("market_inputs_used") is not False:
            raise ValueError(f"{year} {name} not market-isolated")
    if int(replay.get("season") or 0)!=year:
        raise ValueError(f"{year} replay season mismatch")
    if replay.get("trials_per_game")!=10000:
        raise ValueError(f"{year} replay is not 10000-trial precision")
    venue_games=venue.get("games") or {}
    rows=[]
    for pred in replay.get("predictions") or []:
        gid=str(pred["gid"])
        if venue_games.get(gid,{}).get("status")!="PRIMARY_HOME_VENUE":
            continue
        rows.append({
            "season":year,
            "gid":gid,
            "date":str(pred["date"]),
            "p":float(pred["raw_under05"]),
            "y":int(pred["observed_under05"]),
        })
    if not rows:
        raise ValueError(f"{year} has no primary-home rows")
    frame=pd.DataFrame(rows)
    if frame["gid"].duplicated().any():
        raise ValueError(f"{year} duplicate game IDs")
    return frame


def fit_offset(frame):
    p=clip(frame["p"].to_numpy())
    x=logit(p)
    X=np.ones((len(frame),1))
    res=sm.GLM(
        frame["y"].to_numpy(),
        X,
        family=sm.families.Binomial(),
        offset=x,
    ).fit()
    return float(res.params[0])


def fit_affine(frame):
    p=clip(frame["p"].to_numpy())
    x=logit(p)
    X=np.column_stack((np.ones(len(frame)),x))
    res=sm.GLM(frame["y"].to_numpy(),X,family=sm.families.Binomial()).fit()
    return float(res.params[0]),float(res.params[1])


def apply_offset(p,c,lam):
    return expit(logit(clip(p))+float(lam)*float(c))


def apply_affine(p,a,b,lam):
    z=logit(clip(p))
    shrink_a=float(lam)*float(a)
    shrink_b=1.0+float(lam)*(float(b)-1.0)
    return expit(shrink_a+shrink_b*z)


def predict(frame,kind,coef,lam):
    if kind=="identity":
        return frame["p"].to_numpy(float)
    if kind=="offset":
        return apply_offset(frame["p"],coef,float(lam))
    if kind=="affine":
        return apply_affine(frame["p"],coef[0],coef[1],float(lam))
    raise ValueError(kind)


def fit_kind(frame,kind):
    if kind=="identity": return None
    if kind=="offset": return fit_offset(frame)
    if kind=="affine": return fit_affine(frame)
    raise ValueError(kind)


def loso_predictions(dev,kind,lam):
    parts=[]
    folds={}
    for year in DEV_YEARS:
        train=dev[dev["season"]!=year]
        test=dev[dev["season"]==year].copy()
        coef=fit_kind(train,kind)
        test["q"]=predict(test,kind,coef,lam)
        parts.append(test)
        folds[str(year)]={
            "train_seasons":[y for y in DEV_YEARS if y!=year],
            "coefficient":coef,
        }
    return pd.concat(parts,ignore_index=True),folds


def loso_objective(dev,kind,lam):
    pred,_=loso_predictions(dev,kind,float(lam))
    return metric(pred["y"],pred["q"])["logloss"]


def select_lambda(dev,kind):
    if kind=="identity":
        return {
            "multiplier":0.0,
            "loso_logloss":loso_objective(dev,"identity",0.0),
            "no_adjustment_logloss":loso_objective(dev,"identity",0.0),
            "unshrunk_logloss":loso_objective(dev,"identity",0.0),
        }
    opt=minimize_scalar(
        lambda x:loso_objective(dev,kind,float(x)),
        bounds=(0.0,1.0),
        method="bounded",
        options={"xatol":1e-7},
    )
    return {
        "multiplier":float(opt.x),
        "loso_logloss":float(opt.fun),
        "no_adjustment_logloss":float(loso_objective(dev,kind,0.0)),
        "unshrunk_logloss":float(loso_objective(dev,kind,1.0)),
    }


def compare_raw_adjusted(frame,q):
    b0,l0=losses(frame["y"],frame["p"])
    b1,l1=losses(frame["y"],q)
    return {
        "raw":metric(frame["y"],frame["p"]),
        "adjusted":metric(frame["y"],q),
        "adjusted_minus_raw":{
            "brier":float(np.mean(b1-b0)),
            "logloss":float(np.mean(l1-l0)),
        },
    }


def date_cluster_ci(frame,q,draws,seed):
    b0,l0=losses(frame["y"],frame["p"])
    b1,l1=losses(frame["y"],q)
    x=pd.DataFrame({
        "season":frame["season"].to_numpy(),
        "date":frame["date"].to_numpy(),
        "db":b1-b0,
        "dl":l1-l0,
    })
    g=x.groupby(["season","date"]).agg({"db":["sum","count"],"dl":"sum"}).to_numpy(float)
    rng=np.random.default_rng(seed)
    vals=np.empty((draws,2))
    for i in range(draws):
        s=g[rng.integers(0,len(g),size=len(g))].sum(axis=0)
        vals[i]=[s[0]/s[1],s[2]/s[1]]
    return {
        "brier":[float(v) for v in np.quantile(vals[:,0],[.025,.975])],
        "logloss":[float(v) for v in np.quantile(vals[:,1],[.025,.975])],
    }


def probability_map(kind,coef,lam):
    raw=np.asarray([0.45,0.50,0.55,0.60,0.65,0.70],float)
    if kind=="identity": q=raw
    elif kind=="offset": q=apply_offset(raw,coef,lam)
    else: q=apply_affine(raw,coef[0],coef[1],lam)
    return [{"raw":float(p),"adjusted":float(a)} for p,a in zip(raw,q)]


def main():
    a=parse_args()
    frames=[read_year(a.bundle_dir,y) for y in ALL_YEARS]
    all_frame=pd.concat(frames,ignore_index=True)
    dev=all_frame[all_frame["season"].isin(DEV_YEARS)].reset_index(drop=True)
    rep=all_frame[all_frame["season"]==REPLICATION_YEAR].reset_index(drop=True)

    seasonal={
        str(y):metric(
            all_frame.loc[all_frame["season"]==y,"y"],
            all_frame.loc[all_frame["season"]==y,"p"],
        )
        for y in ALL_YEARS
    }

    candidate={}
    for kind in ("identity","offset","affine"):
        shrink=select_lambda(dev,kind)
        q,folds=loso_predictions(dev,kind,shrink["multiplier"])
        q=q.sort_values(["season","gid"]).reset_index(drop=True)
        d=dev.sort_values(["season","gid"]).reset_index(drop=True)
        pooled=compare_raw_adjusted(d,q["q"].to_numpy())
        by_year={}
        for y in DEV_YEARS:
            yy=q[q["season"]==y].sort_values("gid").reset_index(drop=True)
            rr=d[d["season"]==y].sort_values("gid").reset_index(drop=True)
            by_year[str(y)]=compare_raw_adjusted(rr,yy["q"].to_numpy())
        candidate[kind]={
            "shrinkage":shrink,
            "development_loso":pooled,
            "development_by_year":by_year,
            "fold_coefficients":folds,
            "development_ci95_season_date_cluster":date_cluster_ci(
                d,q["q"].to_numpy(),a.bootstrap,2600+len(kind)
            ),
        }

    selected_dev=min(
        ("identity","offset","affine"),
        key=lambda k:(
            candidate[k]["development_loso"]["adjusted"]["logloss"],
            candidate[k]["development_loso"]["adjusted"]["brier"],
            {"identity":0,"offset":1,"affine":2}[k],
        ),
    )

    replication={}
    fitted={}
    for kind in ("identity","offset","affine"):
        coef=fit_kind(dev,kind)
        lam=candidate[kind]["shrinkage"]["multiplier"]
        q=predict(rep,kind,coef,lam)
        replication[kind]=compare_raw_adjusted(rep,q)
        replication[kind]["ci95_date_cluster"]=date_cluster_ci(
            rep,q,a.bootstrap,3600+len(kind)
        )
        fitted[kind]={
            "coefficient":coef,
            "shrinkage_multiplier":lam,
            "probability_map":probability_map(kind,coef,lam),
        }

    chosen_rep=replication[selected_dev]
    replication_pass=(
        selected_dev=="identity"
        or (
            chosen_rep["adjusted_minus_raw"]["logloss"] < 0
            and chosen_rep["adjusted_minus_raw"]["brier"] < 0
        )
    )
    final_kind=selected_dev if replication_pass else "identity"

    if final_kind=="identity":
        final_curve={"type":"none","intercept":0.0,"slope":1.0}
    elif final_kind=="offset":
        # Refit coefficient on all seasons only after candidate form/shrinkage is frozen.
        c=fit_offset(all_frame)
        lam=candidate["offset"]["shrinkage"]["multiplier"]
        final_curve={
            "type":"sigmoid_logit",
            "intercept":float(lam*c),
            "slope":1.0,
        }
    else:
        aa,bb=fit_affine(all_frame)
        lam=candidate["affine"]["shrinkage"]["multiplier"]
        final_curve={
            "type":"sigmoid_logit",
            "intercept":float(lam*aa),
            "slope":float(1.0+lam*(bb-1.0)),
        }

    result={
        "version":"i2-vnext-final-full-calibration-multiyear-v2",
        "market_inputs_used":False,
        "calendar_month_used":False,
        "target":"full I2 Under 0.5 probability",
        "sample":"season-specific PRIMARY_HOME_VENUE games",
        "component_probability_calibration":False,
        "half_adjustment":"NONE per PHASE19D_CHECKPOINT.json",
        "development_years":list(DEV_YEARS),
        "replication_year":REPLICATION_YEAR,
        "replication_pristine_holdout":False,
        "replication_note":(
            "2026 outcomes were previously inspected during half-calibration research; "
            "2026 is not used to fit coefficients or shrinkage and is used here only as "
            "an external no-promotion stress test."
        ),
        "seasonal_raw":seasonal,
        "candidates":candidate,
        "development_selected_candidate":selected_dev,
        "development_fit_for_replication":fitted,
        "replication_2026":replication,
        "promotion_rule":(
            "Select candidate on 2022-2025 LOSO by minimum full-I2 log loss, Brier tie-break. "
            "A non-identity selected candidate must improve both log loss and Brier on 2026 "
            "replication to be promoted; otherwise identity."
        ),
        "replication_pass":bool(replication_pass),
        "final_selected_candidate":final_kind,
        "final_curve":final_curve,
        "governance":{
            "probability_calibration_layers":0 if final_kind=="identity" else 1,
            "component_level_probability_calibration":False,
            "half_inning_calibration_enabled":False,
            "market_conditioning":False,
            "production_changed":False,
            "calendar_month_feature_or_control":False,
            "shrinkage_estimated_by_development_loso":True,
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "seasonal_raw":seasonal,
        "development_selected_candidate":selected_dev,
        "candidate_development":{
            k:{
                "lambda":candidate[k]["shrinkage"]["multiplier"],
                "loso_delta":candidate[k]["development_loso"]["adjusted_minus_raw"],
            } for k in candidate
        },
        "replication_2026":{
            k:replication[k]["adjusted_minus_raw"] for k in replication
        },
        "replication_pass":replication_pass,
        "final_selected_candidate":final_kind,
        "final_curve":final_curve,
    },indent=2))


if __name__=="__main__":
    main()

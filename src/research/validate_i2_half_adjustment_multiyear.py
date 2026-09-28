#!/usr/bin/env python3
"""Cross-season validation of top/bottom I2 calibration adjustments.

Primary sample: each club playing at its season-specific primary home venue.
No calendar-month controls or market inputs.

Candidate shapes:
  offset: logit(q) = logit(p) + c_side
  affine: logit(q) = a_side + b_side*logit(p)

The offset shrinkage multiplier is selected by leave-one-season-out log loss,
separately for top and bottom halves. Full-I2 diagnostics preserve each game's
raw model joint-zero dependence ratio rather than assuming independence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit


YEARS=(2022,2023,2024,2025,2026)
DEVELOPMENT_YEARS=(2022,2023,2024,2025)
HOLDOUT_YEAR=2026
EPS=1e-9


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=5000)
    return p.parse_args()


def metric(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return {
        "n":int(len(y)),
        "mean_prediction":float(p.mean()),
        "observed":float(y.mean()),
        "actual_minus_predicted":float((y-p).mean()),
        "brier":float(np.mean((p-y)**2)),
        "logloss":float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
    }


def losses(y,p):
    y=np.asarray(y,float); p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return (p-y)**2, -y*np.log(p)-(1-y)*np.log1p(-p)


def read_year(root,year):
    replay=json.loads((root/f"replay_{year}_precision_10000.json").read_text())
    inputs=json.loads((root/f"replay_{year}_inputs.json").read_text())
    venue=json.loads((root/f"home_venue_{year}.json").read_text())
    for name,payload in (("replay",replay),("inputs",inputs),("venue",venue)):
        if payload.get("market_inputs_used") is not False:
            raise ValueError(f"{year} {name} not market-isolated")
    if int(replay["season"])!=year or int(inputs["season"])!=year or int(venue["season"])!=year:
        raise ValueError(f"{year} season mismatch")
    if replay.get("trials_per_game")!=10000:
        raise ValueError(f"{year} replay is not 10k-trial canonical")
    games={str(g["gid"]):g for g in inputs["games"]}
    venue_games=venue["games"]
    rows=[]
    game_rows=[]
    for pred in replay["predictions"]:
        gid=str(pred["gid"])
        if venue_games.get(gid,{}).get("status")!="PRIMARY_HOME_VENUE":
            continue
        g=games[gid]
        top_p=float(pred["top2_score_probability"])
        bot_p=float(pred["bottom2_score_probability"])
        raw_under=float(pred["raw_under05"])
        top_y=int(g["observed"]["top2_runs"]>0)
        bot_y=int(g["observed"]["bottom2_runs"]>0)
        denom=max(EPS,(1-top_p)*(1-bot_p))
        dependence=raw_under/denom
        date=str(pred["date"])
        rows.extend([
            {"season":year,"gid":gid,"date":date,"side":"top","p":top_p,"y":top_y},
            {"season":year,"gid":gid,"date":date,"side":"bottom","p":bot_p,"y":bot_y},
        ])
        game_rows.append({
            "season":year,"gid":gid,"date":date,
            "top_p":top_p,"bottom_p":bot_p,
            "y_under":int(pred["observed_under05"]),
            "raw_under":raw_under,"dependence_factor":dependence,
        })
    if not rows:
        raise ValueError(f"{year} has no primary-home-venue rows")
    return pd.DataFrame(rows),pd.DataFrame(game_rows)


def fit_offset(frame):
    x=logit(frame["p"].clip(1e-6,1-1e-6).to_numpy())
    X=np.ones((len(frame),1))
    res=sm.GLM(frame["y"].to_numpy(),X,family=sm.families.Binomial(),offset=x).fit()
    return float(res.params[0])


def fit_affine(frame):
    x=logit(frame["p"].clip(1e-6,1-1e-6).to_numpy())
    X=np.column_stack((np.ones(len(frame)),x))
    res=sm.GLM(frame["y"].to_numpy(),X,family=sm.families.Binomial()).fit()
    return float(res.params[0]),float(res.params[1])


def fit_zero_sum_h(frame):
    """Fit relative bottom-vs-top contrast, then remove the common component."""
    x=logit(frame["p"].clip(1e-6,1-1e-6).to_numpy())
    bottom=frame["side"].eq("bottom").astype(int).to_numpy()
    X=np.column_stack((np.ones(len(frame)),bottom))
    res=sm.GLM(frame["y"].to_numpy(),X,family=sm.families.Binomial(),offset=x).fit()
    top=float(res.params[0])
    bottom_offset=float(res.params[0]+res.params[1])
    common=(top+bottom_offset)/2.0
    h=(bottom_offset-top)/2.0
    return {
        "top_offset":top,
        "bottom_offset":bottom_offset,
        "common_component":common,
        "h":h,
    }


def apply_offset(p,c):
    p=np.asarray(p,float)
    return expit(logit(np.clip(p,1e-6,1-1e-6))+c)


def apply_affine(p,a,b):
    p=np.asarray(p,float)
    return expit(a+b*logit(np.clip(p,1e-6,1-1e-6)))


def apply_zero_sum(p,side,h,lam=1.0):
    delta=(-1.0 if side=="top" else 1.0)*float(h)*float(lam)
    return apply_offset(p,delta)


def adjusted_game_under(games,half_pred):
    lookup={(str(r.gid),str(r.side)):float(r.q) for r in half_pred.itertuples()}
    out=[]
    for r in games.itertuples():
        qt=lookup[(str(r.gid),"top")]
        qb=lookup[(str(r.gid),"bottom")]
        marginal_zero=(1-qt)*(1-qb)
        q=float(np.clip(r.dependence_factor*marginal_zero,EPS,1-EPS))
        out.append(q)
    return np.asarray(out)


def fit_train_coefficients(train,shape):
    out={}
    for side in ("top","bottom"):
        s=train[train["side"]==side]
        out[side]=fit_offset(s) if shape=="offset" else fit_affine(s)
    return out


def predict_halves(test,coefs,shape,shrink=None):
    out=test.copy()
    q=np.zeros(len(out))
    for side in ("top","bottom"):
        m=out["side"].eq(side).to_numpy()
        if shape=="offset":
            c=float(coefs[side])
            if shrink is not None: c*=float(shrink[side])
            q[m]=apply_offset(out.loc[m,"p"],c)
        else:
            a,b=coefs[side]
            q[m]=apply_affine(out.loc[m,"p"],a,b)
    out["q"]=q
    return out


def loso_predictions(halves,games,shape,shrink=None):
    hp=[]; gp=[]
    fold_meta={}
    for year in YEARS:
        train=halves[halves["season"]!=year]
        test=halves[halves["season"]==year]
        gtest=games[games["season"]==year]
        if shape=="zero_sum":
            z=fit_zero_sum_h(train)
            lam=1.0 if shrink is None else float(shrink)
            pred=test.copy()
            pred["q"]=np.where(
                pred["side"].eq("top"),
                apply_zero_sum(pred["p"],"top",z["h"],lam),
                apply_zero_sum(pred["p"],"bottom",z["h"],lam),
            )
            coefs={**z,"shrinkage_multiplier":lam}
        else:
            coefs=fit_train_coefficients(train,shape)
            pred=predict_halves(test,coefs,shape,shrink)
        hp.append(pred)
        adj_under=adjusted_game_under(gtest,pred)
        temp=gtest.copy(); temp["q_under"]=adj_under
        gp.append(temp)
        fold_meta[str(year)]={"train_seasons":[y for y in YEARS if y!=year],"coefficients":coefs}
    return pd.concat(hp,ignore_index=True),pd.concat(gp,ignore_index=True),fold_meta


def cv_shrink_objective(halves,side,lam):
    total=0.0; n=0
    for year in YEARS:
        train=halves[(halves["season"]!=year)&(halves["side"]==side)]
        test=halves[(halves["season"]==year)&(halves["side"]==side)]
        c=fit_offset(train)*lam
        q=apply_offset(test["p"],c)
        _,ll=losses(test["y"],q)
        total+=float(ll.sum()); n+=len(test)
    return total/n


def zero_sum_cv_objective(halves,lam,years=YEARS):
    years=tuple(years)
    scope=halves[halves["season"].isin(years)]
    total=0.0; n=0
    for year in years:
        train=scope[scope["season"]!=year]
        test=scope[scope["season"]==year]
        h=fit_zero_sum_h(train)["h"]
        q=np.where(
            test["side"].eq("top"),
            apply_zero_sum(test["p"],"top",h,lam),
            apply_zero_sum(test["p"],"bottom",h,lam),
        )
        _,ll=losses(test["y"],q)
        total+=float(ll.sum()); n+=len(test)
    return total/n


def select_zero_sum_shrinkage(halves,years=YEARS):
    years=tuple(years)
    opt=minimize_scalar(
        lambda lam:zero_sum_cv_objective(halves,float(lam),years),
        bounds=(0.0,1.0),method="bounded",options={"xatol":1e-5},
    )
    return {
        "multiplier":float(opt.x),
        "loso_logloss":float(opt.fun),
        "years":list(years),
        "unshrunk_loso_logloss":float(zero_sum_cv_objective(halves,1.0,years)),
        "no_adjustment_loso_logloss":float(zero_sum_cv_objective(halves,0.0,years)),
    }


def select_shrinkage(halves):
    out={}
    for side in ("top","bottom"):
        opt=minimize_scalar(
            lambda lam:cv_shrink_objective(halves,side,float(lam)),
            bounds=(0.0,1.0),method="bounded",
            options={"xatol":1e-5},
        )
        out[side]={
            "multiplier":float(opt.x),
            "loso_logloss":float(opt.fun),
            "unshrunk_loso_logloss":float(cv_shrink_objective(halves,side,1.0)),
            "no_adjustment_loso_logloss":float(cv_shrink_objective(halves,side,0.0)),
        }
    return out


def season_summary(halves):
    out={}
    for year in YEARS:
        y=halves[halves["season"]==year]
        side={}
        for s in ("top","bottom"):
            g=y[y["side"]==s]
            c=fit_offset(g)
            side[s]={
                "raw":metric(g["y"],g["p"]),
                "fitted_logit_offset":c,
                "probability_map":{
                    str(p):float(apply_offset([p],c)[0])
                    for p in (0.15,0.20,0.25,0.30,0.35)
                },
            }
        z=fit_zero_sum_h(y)
        out[str(year)]={
            "n_games":int(y["gid"].nunique()),
            "sides":side,
            "half_decomposition":z,
        }
    return out


def compare_predictions(raw_halves,raw_games,hpred,gpred):
    half={}
    for side in ("top","bottom"):
        r=raw_halves[raw_halves["side"]==side]
        p=hpred[hpred["side"]==side]
        b0,l0=losses(r["y"],r["p"]); b1,l1=losses(p["y"],p["q"])
        half[side]={
            "raw":metric(r["y"],r["p"]),
            "adjusted":metric(p["y"],p["q"]),
            "adjusted_minus_raw":{"brier":float(np.mean(b1-b0)),"logloss":float(np.mean(l1-l0))},
        }
    b0,l0=losses(raw_games["y_under"],raw_games["raw_under"])
    b1,l1=losses(gpred["y_under"],gpred["q_under"])
    full={
        "raw":metric(raw_games["y_under"],raw_games["raw_under"]),
        "adjusted":metric(gpred["y_under"],gpred["q_under"]),
        "adjusted_minus_raw":{"brier":float(np.mean(b1-b0)),"logloss":float(np.mean(l1-l0))},
    }
    return {"halves":half,"full_i2_dependence_preserved":full}


def cluster_bootstrap_delta(raw_halves,hpred,draws):
    tmp=raw_halves[["season","date","side","y","p"]].copy()
    tmp["q"]=hpred["q"].to_numpy()
    _,l0=losses(tmp["y"],tmp["p"]); _,l1=losses(tmp["y"],tmp["q"])
    tmp["delta"]=l1-l0
    # cluster by unique season-date so identical calendar dates across seasons cannot collide
    grouped=tmp.groupby(["season","date"])["delta"].agg(["sum","count"]).to_numpy(float)
    rng=np.random.default_rng(1919)
    vals=np.empty(draws)
    for i in range(draws):
        pick=grouped[rng.integers(0,len(grouped),size=len(grouped))].sum(axis=0)
        vals[i]=pick[0]/pick[1]
    return [float(x) for x in np.quantile(vals,[0.025,0.975])]


def forward_2026_holdout(halves,games,zero_sum_shrink):
    dev=halves[halves["season"].isin(DEVELOPMENT_YEARS)].copy()
    test=halves[halves["season"]==HOLDOUT_YEAR].copy()
    gtest=games[games["season"]==HOLDOUT_YEAR].copy()
    z=fit_zero_sum_h(dev)
    lam=float(zero_sum_shrink["multiplier"])
    pred=test.copy()
    pred["q"]=np.where(
        pred["side"].eq("top"),
        apply_zero_sum(pred["p"],"top",z["h"],lam),
        apply_zero_sum(pred["p"],"bottom",z["h"],lam),
    )
    gpred=gtest.copy()
    gpred["q_under"]=adjusted_game_under(gtest,pred)

    test=test.sort_values(["season","gid","side"]).reset_index(drop=True)
    pred=pred.sort_values(["season","gid","side"]).reset_index(drop=True)
    gtest=gtest.sort_values(["season","gid"]).reset_index(drop=True)
    gpred=gpred.sort_values(["season","gid"]).reset_index(drop=True)

    result=compare_predictions(test,gtest,pred,gpred)
    result["training"]={
        "seasons":list(DEVELOPMENT_YEARS),
        "fitted":z,
        "shrinkage_multiplier":lam,
        "applied_h":z["h"]*lam,
    }
    result["holdout_year"]=HOLDOUT_YEAR
    return result


def main():
    a=parse_args()
    half_parts=[]; game_parts=[]
    for year in YEARS:
        h,g=read_year(a.bundle_dir,year)
        half_parts.append(h); game_parts.append(g)
    halves=pd.concat(half_parts,ignore_index=True)
    games=pd.concat(game_parts,ignore_index=True)

    seasonal=season_summary(halves)
    shrink=select_shrinkage(halves)
    shrink_map={s:shrink[s]["multiplier"] for s in ("top","bottom")}
    zero_sum_shrink=select_zero_sum_shrinkage(halves,DEVELOPMENT_YEARS)
    forward_2026=forward_2026_holdout(halves,games,zero_sum_shrink)

    zero_h,zero_g,zero_folds=loso_predictions(halves,games,"zero_sum",zero_sum_shrink["multiplier"])
    zero_raw_h,zero_raw_g,zero_raw_folds=loso_predictions(halves,games,"zero_sum",None)
    offset_h,offset_g,offset_folds=loso_predictions(halves,games,"offset",None)
    shrunk_h,shrunk_g,shrunk_folds=loso_predictions(halves,games,"offset",shrink_map)
    affine_h,affine_g,affine_folds=loso_predictions(halves,games,"affine",None)

    pooled={}
    final_map={}
    development_zero=fit_zero_sum_h(halves[halves["season"].isin(DEVELOPMENT_YEARS)])
    development_zero["shrinkage_multiplier"]=zero_sum_shrink["multiplier"]
    development_zero["applied_h"]=development_zero["h"]*zero_sum_shrink["multiplier"]
    pooled_zero=fit_zero_sum_h(halves)
    pooled_zero["shrinkage_multiplier_fixed_from_2022_2025"]=zero_sum_shrink["multiplier"]
    pooled_zero["future_refit_h"]=pooled_zero["h"]*zero_sum_shrink["multiplier"]
    zero_map={
        side:[
            {"raw":p,"adjusted":float(apply_zero_sum([p],side,pooled_zero["h"],zero_sum_shrink["multiplier"])[0])}
            for p in (0.15,0.20,0.25,0.30,0.35)
        ]
        for side in ("top","bottom")
    }
    for side in ("top","bottom"):
        s=halves[halves["side"]==side]
        c=fit_offset(s)
        sc=c*shrink_map[side]
        pooled[side]={"unshrunk_logit_offset":c,"shrinkage_multiplier":shrink_map[side],"final_logit_offset":sc}
        final_map[side]=[
            {"raw":p,"adjusted":float(apply_offset([p],sc)[0])}
            for p in (0.15,0.20,0.25,0.30,0.35)
        ]

    raw_order=halves.reset_index(drop=True)
    # loso_predictions preserves season then original row order; construct same ordering.
    sort_cols=["season","gid","side"]
    raw_sorted=halves.sort_values(sort_cols).reset_index(drop=True)
    games_sorted=games.sort_values(["season","gid"]).reset_index(drop=True)
    def sorted_pred(h,g):
        return h.sort_values(sort_cols).reset_index(drop=True),g.sort_values(["season","gid"]).reset_index(drop=True)
    zero_h,zero_g=sorted_pred(zero_h,zero_g)
    zero_raw_h,zero_raw_g=sorted_pred(zero_raw_h,zero_raw_g)
    offset_h,offset_g=sorted_pred(offset_h,offset_g)
    shrunk_h,shrunk_g=sorted_pred(shrunk_h,shrunk_g)
    affine_h,affine_g=sorted_pred(affine_h,affine_g)

    result={
        "version":"i2-vnext-phase19-multiyear-half-adjustment-v1",
        "market_inputs_used":False,
        "calendar_month_used":False,
        "seasons":[2022,2023,2024,2025,2026],
        "sample":"season-specific PRIMARY_HOME_VENUE games only",
        "seasonal":seasonal,
        "candidate_shape_comparison":{
            "zero_sum_offset_cv_shrunk":compare_predictions(raw_sorted,games_sorted,zero_h,zero_g),
            "zero_sum_offset_unshrunk":compare_predictions(raw_sorted,games_sorted,zero_raw_h,zero_raw_g),
            "offset_unshrunk":compare_predictions(raw_sorted,games_sorted,offset_h,offset_g),
            "offset_cv_shrunk":compare_predictions(raw_sorted,games_sorted,shrunk_h,shrunk_g),
            "affine_logit":compare_predictions(raw_sorted,games_sorted,affine_h,affine_g),
        },
        "shrinkage_selection":{
            "zero_sum_half_contrast":{
                "method":"2022-2025 leave-one-season-out half-level log-loss minimization for one common multiplier on h; 2026 excluded; multiplier constrained to [0,1]",
                **zero_sum_shrink,
            },
            "independent_offsets_diagnostic":{
                "method":"leave-one-season-out log-loss minimization, independently by half; multiplier constrained to [0,1]",
                "top":shrink["top"],
                "bottom":shrink["bottom"],
            },
        },
        "development_zero_sum_candidate_2022_2025":development_zero,
        "forward_2026_holdout":forward_2026,
        "pooled_zero_sum_refit_2022_2026_for_future":pooled_zero,
        "pooled_zero_sum_probability_map":zero_map,
        "pooled_final_offset_candidate":pooled,
        "pooled_final_probability_map":final_map,
        "loso_fold_coefficients":{
            "zero_sum_offset_cv_shrunk":zero_folds,
            "zero_sum_offset_unshrunk":zero_raw_folds,
            "offset_unshrunk":offset_folds,
            "offset_cv_shrunk":shrunk_folds,
            "affine_logit":affine_folds,
        },
        "uncertainty":{
            "zero_sum_shrunk_half_logloss_delta_ci95_season_date_cluster":
                cluster_bootstrap_delta(raw_sorted,zero_h,a.bootstrap),
            "independent_shrunk_offset_half_logloss_delta_ci95_season_date_cluster":
                cluster_bootstrap_delta(raw_sorted,shrunk_h,a.bootstrap)
        },
        "full_i2_diagnostic_rule":(
            "Adjusted joint Under preserves each game's raw joint-zero dependence factor: "
            "raw_under/[(1-p_top)(1-p_bottom)] times adjusted marginal zero probabilities."
        ),
        "governance":{
            "production_changed":False,
            "holdout_2026_used_for_shrinkage_selection":False,
            "holdout_2026_used_for_development_h_fit":False,
            "market_inputs_used":False,
            "calendar_month_feature_or_control":False,
            "historical_fixed_specification_caveat":(
                "Current C/half-life and transition shrinkage constants were selected later "
                "than some retrospective seasons; early years test robustness of the current "
                "specification, not an untouched historical model-development process."
            ),
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({
        "seasonal":seasonal,
        "shape_comparison":result["candidate_shape_comparison"],
        "shrinkage":result["shrinkage_selection"],
        "development_zero_sum":development_zero,
        "forward_2026_holdout":forward_2026,
        "future_refit_after_holdout":pooled_zero,
        "independent_offset_diagnostic":pooled,
        "ci":result["uncertainty"],
    },indent=2))


if __name__=="__main__":
    main()

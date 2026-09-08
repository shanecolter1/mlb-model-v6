#!/usr/bin/env python3
"""All-inning (I1-I9) empirical-total anchored matchup-variable research.

Research-only. Sportsbook inning prices are not inputs.
The opening full-game total selects the empirical inning baseline; matchup features
may only explain residual variation around that baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-9


def clamp(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p):
    p = clamp(p)
    return np.log(p / (1 - p))


def logistic(x):
    return 1 / (1 + np.exp(-np.clip(np.asarray(x, dtype=float), -40, 40)))


def pick(df, names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def fair_american(p):
    p = float(p)
    if not (0 < p < 1):
        return None
    return round(-100 * p / (1 - p)) if p > .5 else round(100 * (1 - p) / p)


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p):
    p = clamp(p)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))


def detect_total(df):
    c = pick(df, ["opening_total","dk_opening_total","draftkings_opening_total","pregame_opening_total","game_total_open","total_open","open_total"])
    if not c:
        raise ValueError("Opening full-game total column not found")
    return c


def detect_date(df):
    c = pick(df, ["game_date","date","gameDate"])
    if not c:
        raise ValueError("game_date/date column required for chronological validation")
    return c


def inning_runs(df, inn):
    full = pick(df, [f"i{inn}_runs", f"inning_{inn}_runs", f"inning{inn}_runs", f"runs_inning_{inn}"])
    if full:
        return pd.to_numeric(df[full], errors="coerce")
    away = pick(df, [f"away_i{inn}", f"away_inning_{inn}", f"away_inning{inn}_runs", f"away_runs_i{inn}"])
    home = pick(df, [f"home_i{inn}", f"home_inning_{inn}", f"home_inning{inn}_runs", f"home_runs_i{inn}"])
    if away and home:
        return pd.to_numeric(df[away], errors="coerce") + pd.to_numeric(df[home], errors="coerce")
    return None


def load_registry(path):
    reg = pd.read_csv(path)
    need = {"feature","family","status","inning_scope"}
    if not need.issubset(reg.columns):
        raise ValueError(f"Registry missing {sorted(need-set(reg.columns))}")
    return reg


def scope_has(scope, inn):
    s = str(scope).strip()
    if "-" in s:
        a,b = s.split("-",1)
        return int(a) <= inn <= int(b)
    return inn in [int(x) for x in s.split(",") if str(x).strip()]


def inning_features(df, reg, inn):
    r = reg[(reg.status.isin(["candidate","active_shadow"])) & reg.inning_scope.map(lambda s: scope_has(s, inn))]
    available = [f for f in r.feature if f in df.columns]
    missing = [f for f in r.feature if f not in df.columns]
    family = dict(zip(r.feature, r.family))
    return available, missing, family


def standardize_fit(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[(~np.isfinite(sd)) | (sd < 1e-9)] = 1
    return mu, sd


def standardize(X, mu, sd):
    z = (X-mu)/sd
    return np.where(np.isfinite(z), z, 0)


def fit_ridge_offset(X, y, offset, lam=5.0, max_iter=100):
    beta = np.zeros(X.shape[1])
    I = np.eye(X.shape[1]) * lam
    for _ in range(max_iter):
        eta = offset + X @ beta
        p = logistic(eta)
        w = np.clip(p*(1-p), 1e-6, None)
        grad = X.T @ (y-p) - I @ beta
        H = X.T @ (w[:,None]*X) + I
        step = np.linalg.solve(H, grad)
        new = beta + step
        if np.max(np.abs(new-beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def empirical_prior(train, total_col, y_col, strength=100.0):
    broad = float(train[y_col].mean())
    tab = train.groupby(total_col)[y_col].agg(["sum","count"])
    priors = {}
    for total,row in tab.iterrows():
        n = float(row["count"]); wins = float(row["sum"])
        priors[float(total)] = {"p": (wins + strength*broad)/(n+strength), "n": int(n)}
    return broad, priors


def baseline(df, total_col, broad, priors):
    ps=[]; ns=[]; used=[]
    keys=list(priors)
    for t in pd.to_numeric(df[total_col], errors="coerce"):
        if not np.isfinite(t) or not keys:
            ps.append(broad); ns.append(0); used.append(None); continue
        k = float(t) if float(t) in priors else min(keys, key=lambda z: abs(z-float(t)))
        ps.append(priors[k]["p"]); ns.append(priors[k]["n"]); used.append(k)
    return np.asarray(ps), np.asarray(ns), used


def season_folds(df, min_train=2):
    seasons=sorted(df.__season.dropna().astype(int).unique())
    for idx in range(min_train, len(seasons)):
        test=seasons[idx]
        train_seasons=seasons[:idx]
        tr=df[df.__season.isin(train_seasons)].copy(); te=df[df.__season==test].copy()
        if len(tr) and len(te):
            yield train_seasons,test,tr,te


def calibration_bins(y,p,n_bins=10):
    d=pd.DataFrame({"y":y,"p":p})
    try:
        d["bin"]=pd.qcut(d.p,q=n_bins,duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    return d.groupby("bin",observed=True).agg(n=("y","size"),mean_pred=("p","mean"),actual=("y","mean")).reset_index().assign(bin=lambda x:x.bin.astype(str))


def evaluate_inning(df, inn, total_col, features, family, prior_strength, ridge_lambda):
    fold_rows=[]; pred_rows=[]; coef_rows=[]; cal_rows=[]
    for train_seasons,test_season,tr,te in season_folds(df):
        broad,priors=empirical_prior(tr,total_col,"__y",prior_strength)
        p_tr,_,_=baseline(tr,total_col,broad,priors)
        p_te,n_te,bucket_te=baseline(te,total_col,broad,priors)
        pred=p_te.copy(); beta=np.array([])
        if features:
            Xtr0=tr[features].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            Xte0=te[features].apply(pd.to_numeric,errors="coerce").to_numpy(float)
            mu,sd=standardize_fit(Xtr0)
            Xtr=standardize(Xtr0,mu,sd); Xte=standardize(Xte0,mu,sd)
            beta=fit_ridge_offset(Xtr,tr.__y.to_numpy(float),logit(p_tr),ridge_lambda)
            pred=logistic(logit(p_te)+Xte@beta)
            for f,b in zip(features,beta):
                coef_rows.append({"inning":inn,"test_season":test_season,"feature":f,"family":family.get(f),"standardized_beta":float(b)})
        y=te.__y.to_numpy(float)
        fold_rows.append({"inning":inn,"train_seasons":",".join(map(str,train_seasons)),"test_season":test_season,"n":len(te),"feature_count":len(features),"baseline_brier":brier(y,p_te),"model_brier":brier(y,pred),"baseline_log_loss":log_loss(y,p_te),"model_log_loss":log_loss(y,pred)})
        pp=te[["__game_date","__season",total_col,"__runs","__y"]].copy()
        pp["inning"]=inn; pp["empirical_total_p"]=p_te; pp["prediction"]=pred; pp["matchup_delta"]=pred-p_te; pp["prior_n"]=n_te; pp["prior_bucket_used"]=bucket_te; pp["test_season"]=test_season
        pred_rows.append(pp)
        cb=calibration_bins(y,pred)
        if not cb.empty:
            cb["inning"]=inn; cb["test_season"]=test_season; cal_rows.append(cb)
    return pd.DataFrame(fold_rows), pd.concat(pred_rows,ignore_index=True) if pred_rows else pd.DataFrame(), pd.DataFrame(coef_rows), pd.concat(cal_rows,ignore_index=True) if cal_rows else pd.DataFrame()


def baseline_table(df,total_col,prior_strength):
    rows=[]
    for inn in range(1,10):
        s=inning_runs(df,inn)
        if s is None: continue
        tmp=df.copy(); tmp["__runs"]=s; tmp=tmp[tmp.__runs.notna()].copy(); tmp["__y"]=(tmp.__runs>=1).astype(int)
        broad=float(tmp.__y.mean())
        for total,g in tmp.groupby(total_col):
            n=len(g); wins=int(g.__y.sum()); raw=wins/n; shr=(wins+prior_strength*broad)/(n+prior_strength)
            rows.append({"inning":inn,"opening_total":total,"n":n,"wins":wins,"raw_p_over05":raw,"shrunk_p_over05":shr,"raw_p_under05":1-raw,"fair_over_raw":fair_american(raw),"fair_under_raw":fair_american(1-raw),"broad_inning_over":broad})
    return pd.DataFrame(rows)


def dispersion(preds,total_col):
    rows=[]
    if preds.empty: return pd.DataFrame()
    for (inn,total),g in preds.groupby(["inning",total_col]):
        d=g.matchup_delta.astype(float)
        rows.append({"inning":inn,"opening_total":total,"n_predictions":len(g),"mean_delta_pp":100*d.mean(),"sd_delta_pp":100*d.std(ddof=1) if len(g)>1 else np.nan,"p05_delta_pp":100*d.quantile(.05),"p25_delta_pp":100*d.quantile(.25),"p50_delta_pp":100*d.quantile(.5),"p75_delta_pp":100*d.quantile(.75),"p95_delta_pp":100*d.quantile(.95)})
    return pd.DataFrame(rows).sort_values(["inning","opening_total"])


def family_ablation(df,inn,total_col,features,family,prior_strength,ridge_lambda):
    if not features: return pd.DataFrame()
    full_folds,_,_,_=evaluate_inning(df,inn,total_col,features,family,prior_strength,ridge_lambda)
    base_ll=full_folds.model_log_loss.mean(); base_br=full_folds.model_brier.mean()
    rows=[]
    for fam in sorted(set(family.get(f) for f in features)):
        kept=[f for f in features if family.get(f)!=fam]
        folds,_,_,_=evaluate_inning(df,inn,total_col,kept,family,prior_strength,ridge_lambda)
        rows.append({"inning":inn,"removed_family":fam,"features_remaining":len(kept),"full_mean_log_loss":base_ll,"ablated_mean_log_loss":folds.model_log_loss.mean(),"delta_log_loss":folds.model_log_loss.mean()-base_ll,"full_mean_brier":base_br,"ablated_mean_brier":folds.model_brier.mean(),"delta_brier":folds.model_brier.mean()-base_br})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--registry",default="config/inning_matchup_feature_registry.csv")
    ap.add_argument("--output-dir",default="data/derived/i2/all_inning_matchup_research")
    ap.add_argument("--prior-strength",type=float,default=100.0)
    ap.add_argument("--ridge-lambda",type=float,default=5.0)
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.input)
    total_col=detect_total(df); date_col=detect_date(df)
    df[total_col]=pd.to_numeric(df[total_col],errors="coerce")
    df["__game_date"]=pd.to_datetime(df[date_col],errors="coerce"); df["__season"]=df.__game_date.dt.year
    df=df[df[total_col].notna() & df.__game_date.notna()].copy()
    reg=load_registry(args.registry)

    bt=baseline_table(df,total_col,args.prior_strength); bt.to_csv(out/"empirical_total_inning_baselines.csv",index=False)
    all_folds=[]; all_preds=[]; all_coef=[]; all_cal=[]; all_abl=[]; missing_by_inning={}
    for inn in range(1,10):
        runs=inning_runs(df,inn)
        if runs is None:
            missing_by_inning[str(inn)]={"status":"MISSING_INNING_OUTCOME"}; continue
        d=df.copy(); d["__runs"]=runs; d=d[d.__runs.notna()].copy(); d["__y"]=(d.__runs>=1).astype(float)
        features,missing,family=inning_features(d,reg,inn); missing_by_inning[str(inn)]={"available_features":features,"missing_features":missing}
        folds,preds,coef,cal=evaluate_inning(d,inn,total_col,features,family,args.prior_strength,args.ridge_lambda)
        abl=family_ablation(d,inn,total_col,features,family,args.prior_strength,args.ridge_lambda)
        all_folds.append(folds); all_preds.append(preds); all_coef.append(coef); all_cal.append(cal); all_abl.append(abl)
    folds=pd.concat(all_folds,ignore_index=True) if all_folds else pd.DataFrame(); preds=pd.concat(all_preds,ignore_index=True) if all_preds else pd.DataFrame(); coef=pd.concat(all_coef,ignore_index=True) if all_coef else pd.DataFrame(); cal=pd.concat(all_cal,ignore_index=True) if all_cal else pd.DataFrame(); abl=pd.concat(all_abl,ignore_index=True) if all_abl else pd.DataFrame()
    folds.to_csv(out/"walk_forward_metrics.csv",index=False); preds.to_csv(out/"walk_forward_predictions.csv",index=False); coef.to_csv(out/"standardized_coefficients.csv",index=False); cal.to_csv(out/"calibration_bins.csv",index=False); abl.to_csv(out/"feature_family_ablation.csv",index=False)
    disp=dispersion(preds,total_col); disp.to_csv(out/"matchup_delta_dispersion_by_total_inning.csv",index=False)
    summary=[]
    if not folds.empty:
        for inn,g in folds.groupby("inning"):
            summary.append({"inning":int(inn),"folds":len(g),"mean_baseline_log_loss":g.baseline_log_loss.mean(),"mean_model_log_loss":g.model_log_loss.mean(),"log_loss_improvement":g.baseline_log_loss.mean()-g.model_log_loss.mean(),"mean_baseline_brier":g.baseline_brier.mean(),"mean_model_brier":g.model_brier.mean(),"brier_improvement":g.baseline_brier.mean()-g.model_brier.mean()})
    pd.DataFrame(summary).to_csv(out/"cross_inning_incremental_value.csv",index=False)
    manifest={"status":"RESEARCH_COMPLETE" if len(folds) else "BLOCKED_NO_VALID_FOLDS","input":args.input,"rows":len(df),"seasons":sorted(int(x) for x in df.__season.dropna().unique()),"opening_total_column":total_col,"prior_strength":args.prior_strength,"ridge_lambda":args.ridge_lambda,"innings_requested":list(range(1,10)),"feature_availability":missing_by_inning,"governance":{"market_isolated":True,"chronological":True,"production_changed":False,"warning":"Final-feed historical lineup/starter features are discovery-only unless archived pregame timing is verified."}}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()

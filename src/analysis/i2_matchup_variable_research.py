#!/usr/bin/env python3
"""Research framework for I2 matchup-variable inclusion and weighting.

Governance intent:
- empirical pregame-total I2 probability is the anchor/offset;
- matchup variables may explain only residual variation around that anchor;
- all feature selection and weighting is chronological and market-isolated;
- candidate complexity must beat the empirical-total-only benchmark out of sample;
- variable effects are regularized and reliability-shrunk where supported.

Expected input is one row per historical game with leakage-free pregame features.
Required columns are detected from common aliases for date/season, opening total, and I2 runs.
Candidate feature columns are selected from config/i2_matchup_feature_registry.csv.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-9


def clamp_prob(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def logit(p):
    p = clamp_prob(p)
    return np.log(p / (1 - p))


def logistic(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def pick_col(df: pd.DataFrame, names: Iterable[str]):
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def detect_columns(df: pd.DataFrame):
    date_col = pick_col(df, ["game_date", "date", "gameDate"])
    season_col = pick_col(df, ["season", "year"])
    total_col = pick_col(df, [
        "opening_total", "dk_opening_total", "draftkings_opening_total",
        "pregame_opening_total", "game_total_open", "total_open", "open_total",
    ])
    i2_col = pick_col(df, ["i2_runs", "inning_2_runs", "inning2_runs", "runs_inning_2"])
    if not i2_col:
        away = pick_col(df, ["away_i2", "away_inning_2", "away_inning2_runs", "away_runs_i2"])
        home = pick_col(df, ["home_i2", "home_inning_2", "home_inning2_runs", "home_runs_i2"])
        if away and home:
            df = df.copy()
            df["__i2_runs"] = pd.to_numeric(df[away], errors="coerce") + pd.to_numeric(df[home], errors="coerce")
            i2_col = "__i2_runs"
    if not total_col or not i2_col:
        raise ValueError("Input must include an opening full-game total and I2 runs (full inning or away/home halves).")
    if not date_col and not season_col:
        raise ValueError("Input must include game_date/date or season/year for chronological validation.")
    return df, date_col, season_col, total_col, i2_col


def add_time_fields(df, date_col, season_col):
    out = df.copy()
    if date_col:
        out["__game_date"] = pd.to_datetime(out[date_col], errors="coerce")
        out["__season"] = out["__game_date"].dt.year
    else:
        out["__season"] = pd.to_numeric(out[season_col], errors="coerce")
        out["__game_date"] = pd.to_datetime(out["__season"].astype("Int64").astype(str) + "-07-01", errors="coerce")
    return out


def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = clamp_prob(p)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = clamp_prob(p)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_intercept_slope(y, p):
    """Fit y ~ intercept + slope*logit(p) with tiny ridge for numerical stability."""
    x = logit(p)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    for _ in range(50):
        eta = X @ beta
        mu = logistic(eta)
        w = np.clip(mu * (1 - mu), 1e-6, None)
        z = eta + (np.asarray(y) - mu) / w
        A = X.T @ (w[:, None] * X) + np.diag([1e-8, 1e-8])
        b = X.T @ (w * z)
        new = np.linalg.solve(A, b)
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    return float(beta[0]), float(beta[1])


def fit_empirical_total_prior(train: pd.DataFrame, total_col: str, y_col: str, prior_strength: float):
    broad = float(train[y_col].mean())
    grouped = train.groupby(total_col)[y_col].agg(["sum", "count"]).reset_index()
    priors = {}
    for _, row in grouped.iterrows():
        n = float(row["count"])
        wins = float(row["sum"])
        p = (wins + prior_strength * broad) / (n + prior_strength)
        priors[float(row[total_col])] = {"p": float(p), "n": int(n)}
    return broad, priors


def nearest_prior(total, priors, broad):
    if not priors:
        return broad, 0, None
    t = float(total)
    if t in priors:
        r = priors[t]
        return r["p"], r["n"], t
    nearest = min(priors, key=lambda k: abs(k - t))
    r = priors[nearest]
    return r["p"], r["n"], nearest


def baseline_for(df, total_col, broad, priors):
    vals = [nearest_prior(t, priors, broad) for t in df[total_col]]
    return np.array([v[0] for v in vals]), np.array([v[1] for v in vals]), [v[2] for v in vals]


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X):
        mean = np.nanmean(X, axis=0)
        scale = np.nanstd(X, axis=0)
        scale = np.where((~np.isfinite(scale)) | (scale < 1e-9), 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, X):
        Z = (X - self.mean) / self.scale
        return np.where(np.isfinite(Z), Z, 0.0)


def fit_ridge_logistic_offset(X, y, offset, lam=5.0, max_iter=100):
    """Penalized logistic regression with fixed empirical-total logit offset."""
    n, k = X.shape
    beta = np.zeros(k)
    ridge = np.eye(k) * float(lam)
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = logistic(eta)
        w = np.clip(mu * (1 - mu), 1e-6, None)
        grad = X.T @ (y - mu) - ridge @ beta
        hess_pos = X.T @ (w[:, None] * X) + ridge
        step = np.linalg.solve(hess_pos, grad)
        new = beta + step
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def load_registry(path: Path):
    reg = pd.read_csv(path)
    required = {"feature", "family", "status"}
    if not required.issubset(reg.columns):
        raise ValueError(f"Feature registry missing required columns: {sorted(required - set(reg.columns))}")
    return reg


def available_features(df, registry):
    allowed_status = {"candidate", "active_shadow"}
    rows = registry[registry["status"].isin(allowed_status)].copy()
    features = [f for f in rows["feature"].tolist() if f in df.columns]
    missing = [f for f in rows["feature"].tolist() if f not in df.columns]
    return features, missing


def season_folds(df, min_train_seasons=2):
    seasons = sorted(int(s) for s in df["__season"].dropna().unique())
    folds = []
    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        train = df[df["__season"].isin(train_seasons)].copy()
        test = df[df["__season"] == test_season].copy()
        if len(train) and len(test):
            folds.append((train_seasons, test_season, train, test))
    return folds


def evaluate_feature_set(df, features, total_col, y_col, prior_strength, ridge_lambda):
    fold_rows = []
    pred_frames = []
    coef_rows = []
    for train_seasons, test_season, train, test in season_folds(df):
        broad, priors = fit_empirical_total_prior(train, total_col, y_col, prior_strength)
        p_train, _, _ = baseline_for(train, total_col, broad, priors)
        p_test, prior_n, prior_bucket = baseline_for(test, total_col, broad, priors)
        baseline_pred = p_test.copy()
        if features:
            Xtr_raw = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            Xte_raw = test[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            scaler = Standardizer.fit(Xtr_raw)
            Xtr = scaler.transform(Xtr_raw)
            Xte = scaler.transform(Xte_raw)
            beta = fit_ridge_logistic_offset(
                Xtr,
                train[y_col].to_numpy(float),
                logit(p_train),
                lam=ridge_lambda,
            )
            pred = logistic(logit(p_test) + Xte @ beta)
            for f, b in zip(features, beta):
                coef_rows.append({"test_season": test_season, "feature": f, "standardized_beta": float(b)})
        else:
            pred = baseline_pred

        y = test[y_col].to_numpy(float)
        ci, cs = calibration_intercept_slope(y, pred)
        fold_rows.append({
            "train_seasons": ",".join(map(str, train_seasons)),
            "test_season": test_season,
            "n_test": len(test),
            "features": ",".join(features) if features else "EMPIRICAL_TOTAL_ONLY",
            "brier": brier(y, pred),
            "log_loss": log_loss(y, pred),
            "calibration_intercept": ci,
            "calibration_slope": cs,
            "baseline_brier": brier(y, baseline_pred),
            "baseline_log_loss": log_loss(y, baseline_pred),
        })
        pf = test[["__game_date", "__season", total_col, y_col]].copy()
        pf["empirical_total_p"] = p_test
        pf["prediction"] = pred
        pf["matchup_delta"] = pred - p_test
        pf["prior_n"] = prior_n
        pf["prior_bucket_used"] = prior_bucket
        pf["test_season"] = test_season
        pred_frames.append(pf)
    return pd.DataFrame(fold_rows), pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(), pd.DataFrame(coef_rows)


def univariate_diagnostics(df, features, total_col, y_col, prior_strength):
    rows = []
    # Use strictly prior-season anchors for the observed residual diagnostic.
    for train_seasons, test_season, train, test in season_folds(df):
        broad, priors = fit_empirical_total_prior(train, total_col, y_col, prior_strength)
        base, _, _ = baseline_for(test, total_col, broad, priors)
        tmp = test.copy()
        tmp["__base"] = base
        tmp["__resid"] = tmp[y_col] - tmp["__base"]
        for feature in features:
            x = pd.to_numeric(tmp[feature], errors="coerce")
            valid = x.notna()
            if valid.sum() < 100:
                continue
            try:
                bins = pd.qcut(x[valid], q=5, duplicates="drop")
            except ValueError:
                continue
            sub = tmp.loc[valid, [y_col, "__base", "__resid"]].copy()
            sub["bin"] = bins.astype(str).values
            sub["x"] = x[valid].values
            for label, g in sub.groupby("bin", observed=True):
                rows.append({
                    "test_season": test_season,
                    "feature": feature,
                    "bin": label,
                    "n": len(g),
                    "feature_mean": float(g["x"].mean()),
                    "expected_from_total": float(g["__base"].mean()),
                    "actual_i2_over": float(g[y_col].mean()),
                    "residual_pp": float(100 * g["__resid"].mean()),
                })
    return pd.DataFrame(rows)


def matchup_dispersion(predictions, total_col):
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for total, g in predictions.groupby(total_col):
        d = g["matchup_delta"].astype(float)
        rows.append({
            "opening_total": total,
            "n_predictions": len(g),
            "mean_matchup_delta_pp": 100 * float(d.mean()),
            "sd_matchup_delta_pp": 100 * float(d.std(ddof=1)) if len(g) > 1 else np.nan,
            "p05_delta_pp": 100 * float(d.quantile(0.05)),
            "p95_delta_pp": 100 * float(d.quantile(0.95)),
        })
    return pd.DataFrame(rows).sort_values("opening_total")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Leakage-free historical feature matrix CSV or CSV.GZ")
    ap.add_argument("--registry", default="config/i2_matchup_feature_registry.csv")
    ap.add_argument("--output-dir", default="data/derived/i2/matchup_variable_research")
    ap.add_argument("--prior-strength", type=float, default=100.0, help="Beta-binomial shrinkage strength for exact total bucket toward broad train prior")
    ap.add_argument("--ridge-lambda", type=float, default=5.0)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    df, date_col, season_col, total_col, i2_col = detect_columns(df)
    df = add_time_fields(df, date_col, season_col)
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df[i2_col] = pd.to_numeric(df[i2_col], errors="coerce")
    df["__i2_over"] = (df[i2_col] >= 1).astype(float)
    df = df[df[total_col].notna() & df[i2_col].notna() & df["__season"].notna()].copy()
    df["__season"] = df["__season"].astype(int)

    registry = load_registry(Path(args.registry))
    features, missing = available_features(df, registry)

    baseline_folds, baseline_preds, _ = evaluate_feature_set(
        df, [], total_col, "__i2_over", args.prior_strength, args.ridge_lambda
    )
    baseline_folds.to_csv(out / "m0_empirical_total_only_folds.csv", index=False)

    manifest = {
        "status": "PASS_BASELINE_ONLY" if not features else "RESEARCH_COMPLETE",
        "input": str(src),
        "rows": int(len(df)),
        "seasons": sorted(int(x) for x in df["__season"].unique()),
        "opening_total_column": total_col,
        "i2_runs_column": i2_col,
        "empirical_anchor": "training-only exact opening-total I2 Over rate, shrunk toward training broad prior",
        "prior_strength": args.prior_strength,
        "ridge_lambda": args.ridge_lambda,
        "available_candidate_features": features,
        "missing_candidate_features": missing,
        "market_isolation": "No sportsbook I2 price or market-agreement variable permitted in fitting or selection.",
        "governance": "Chronological season folds; no random split; complexity must beat M0 empirical-total-only benchmark.",
    }

    if not features:
        manifest["status"] = "BLOCKED_MATCHUP_FEATURES_MISSING"
        manifest["block_reason"] = "Historical input has no registered leakage-free matchup feature columns. Baseline testing is valid; matchup inclusion/weight fitting is not."
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2))
        return

    uni = univariate_diagnostics(df, features, total_col, "__i2_over", args.prior_strength)
    uni.to_csv(out / "univariate_residual_diagnostics.csv", index=False)

    # Sequential family build: each family is accepted provisionally only if aggregate OOS log loss improves.
    reg_avail = registry[registry["feature"].isin(features)].copy()
    selected = []
    stage_rows = []
    best_ll = float(baseline_folds["log_loss"].mean()) if len(baseline_folds) else float("inf")
    family_order = list(dict.fromkeys(reg_avail["family"].tolist()))
    final_preds = baseline_preds
    final_coefs = pd.DataFrame()

    for family in family_order:
        family_features = reg_avail.loc[reg_avail["family"] == family, "feature"].tolist()
        candidate = selected + family_features
        folds, preds, coefs = evaluate_feature_set(
            df, candidate, total_col, "__i2_over", args.prior_strength, args.ridge_lambda
        )
        ll = float(folds["log_loss"].mean()) if len(folds) else float("inf")
        br = float(folds["brier"].mean()) if len(folds) else float("inf")
        accepted = ll < best_ll - 1e-5
        stage_rows.append({
            "family": family,
            "candidate_features": ",".join(family_features),
            "accepted": accepted,
            "mean_log_loss": ll,
            "mean_brier": br,
            "prior_best_log_loss": best_ll,
            "log_loss_improvement": best_ll - ll,
        })
        if accepted:
            selected = candidate
            best_ll = ll
            final_preds = preds
            final_coefs = coefs

    pd.DataFrame(stage_rows).to_csv(out / "sequential_family_gate.csv", index=False)

    final_folds, final_preds, final_coefs = evaluate_feature_set(
        df, selected, total_col, "__i2_over", args.prior_strength, args.ridge_lambda
    )
    final_folds.to_csv(out / "final_walk_forward_folds.csv", index=False)
    final_preds.to_csv(out / "final_walk_forward_predictions.csv", index=False)
    final_coefs.to_csv(out / "final_standardized_coefficients_by_fold.csv", index=False)
    matchup_dispersion(final_preds, total_col).to_csv(out / "matchup_adjustment_dispersion_by_total.csv", index=False)

    # Ablation: remove one selected feature at a time from the accepted model.
    ablation_rows = []
    if selected:
        full_ll = float(final_folds["log_loss"].mean())
        full_br = float(final_folds["brier"].mean())
        for feature in selected:
            keep = [f for f in selected if f != feature]
            folds, _, _ = evaluate_feature_set(df, keep, total_col, "__i2_over", args.prior_strength, args.ridge_lambda)
            ablation_rows.append({
                "removed_feature": feature,
                "mean_log_loss_without": float(folds["log_loss"].mean()),
                "delta_log_loss_vs_full": float(folds["log_loss"].mean()) - full_ll,
                "mean_brier_without": float(folds["brier"].mean()),
                "delta_brier_vs_full": float(folds["brier"].mean()) - full_br,
            })
    pd.DataFrame(ablation_rows).to_csv(out / "feature_ablation.csv", index=False)

    manifest.update({
        "selected_features": selected,
        "m0_mean_log_loss": float(baseline_folds["log_loss"].mean()) if len(baseline_folds) else None,
        "final_mean_log_loss": float(final_folds["log_loss"].mean()) if len(final_folds) else None,
        "m0_mean_brier": float(baseline_folds["brier"].mean()) if len(baseline_folds) else None,
        "final_mean_brier": float(final_folds["brier"].mean()) if len(final_folds) else None,
        "promotion_status": "RESEARCH_ONLY_NOT_PRODUCTION",
        "next_gate": "Freeze chosen feature/regularization specification before any forward 2026 evaluation or production integration.",
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

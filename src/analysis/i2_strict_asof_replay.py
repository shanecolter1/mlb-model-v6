#!/usr/bin/env python3
"""Strict chronological I2 retrospective replay using only pregame-reconstructable data.

Generate frozen out-of-sample P(I2 Over 0.5) predictions for threshold calibration.
This first replay intentionally excludes retrospective final-feed lineup identities,
actual starter identities, same-game I1 outcomes, and all I2 derivative prices.

Allowed inputs:
- DraftKings opening full-game total point from the canonical historical master,
  used only as the approved structural run-environment anchor;
- team batting and pitching-allowed event rates constructed strictly from dates before
  the target game;
- game/date/team identifiers used only for joins;
- I2 outcome used only after prediction for grading.

This is a conservative STRICT-ASOF challenger, not a byte-for-byte replay of the
current live simulator. It establishes a leakage-safe historical prediction surface
and tests whether baseball context beyond the total anchor adds OOS discrimination.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-9
EVENTS = ["strikeout", "walk", "hit_by_pitch", "home_run", "hit", "xbh", "onbase", "contact"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logistic(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -40, 40)))


def brier(y, p):
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_total_prior(train, strength=100.0):
    broad = float(train["actual_over"].mean())
    g = train.groupby("opening_total")["actual_over"].agg(["sum", "count"])
    priors = {}
    for total, r in g.iterrows():
        n = float(r["count"])
        priors[float(total)] = (float(r["sum"]) + strength * broad) / (n + strength)
    return broad, priors


def map_total_prior(values, broad, priors):
    keys = list(priors)
    out, used = [], []
    for x in values:
        t = float(x)
        if t in priors:
            k = t
        elif keys:
            k = min(keys, key=lambda z: abs(z - t))
        else:
            k = None
        out.append(priors[k] if k is not None else broad)
        used.append(k)
    return np.asarray(out, float), used


class Standardizer:
    def fit(self, X):
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where((~np.isfinite(self.scale_)) | (self.scale_ < 1e-9), 1.0, self.scale_)
        return self

    def transform(self, X):
        Z = (X - self.mean_) / self.scale_
        return np.where(np.isfinite(Z), Z, 0.0)


def fit_ridge_offset(X, y, offset, lam=10.0, max_iter=100):
    beta = np.zeros(X.shape[1], dtype=float)
    ridge = np.eye(X.shape[1]) * float(lam)
    for _ in range(max_iter):
        eta = offset + X @ beta
        mu = logistic(eta)
        w = np.clip(mu * (1 - mu), 1e-6, None)
        grad = X.T @ (y - mu) - ridge @ beta
        h = X.T @ (w[:, None] * X) + ridge
        step = np.linalg.solve(h, grad)
        new = beta + step
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


def canonical_master(path: Path):
    m = pd.read_csv(path, low_memory=False)
    required = ["season", "game_date", "away_team_code", "home_team_code", "dk_total_open_total", "inning2_total_runs"]
    missing = [c for c in required if c not in m.columns]
    if missing:
        raise ValueError(f"Canonical master missing required columns: {missing}")
    if "benchmark_matched" in m.columns:
        m = m[m["benchmark_matched"] == True].copy()  # noqa: E712
    m["game_date"] = pd.to_datetime(m["game_date"], errors="coerce").dt.normalize()
    m["opening_total"] = pd.to_numeric(m["dk_total_open_total"], errors="coerce")
    m["i2_runs"] = pd.to_numeric(m["inning2_total_runs"], errors="coerce")
    m["actual_over"] = (m["i2_runs"] >= 1).astype(float)
    m["season"] = pd.to_numeric(m["season"], errors="coerce")
    m = m[m["opening_total"].notna() & m["i2_runs"].notna() & m["game_date"].notna()].copy()
    m["season"] = m["season"].astype(int)
    keep = ["season", "game_date", "game_number", "away_team_code", "home_team_code", "opening_total", "i2_runs", "actual_over"]
    if "retro_game_id" in m.columns:
        keep.append("retro_game_id")
    return m[[c for c in keep if c in m.columns]].copy()


def team_code_map():
    # StatsAPI abbreviations and full names -> canonical codes in joined master.
    return {
        "AZ":"AZ", "ARI":"AZ", "ARIZONA DIAMONDBACKS":"AZ",
        "ATH":"ATH", "OAK":"ATH", "OAKLAND ATHLETICS":"ATH", "ATHLETICS":"ATH",
        "ATL":"ATL", "ATLANTA BRAVES":"ATL", "BAL":"BAL", "BALTIMORE ORIOLES":"BAL",
        "BOS":"BOS", "BOSTON RED SOX":"BOS", "CHC":"CHC", "CHICAGO CUBS":"CHC",
        "CWS":"CHW", "CHW":"CHW", "CHICAGO WHITE SOX":"CHW", "CIN":"CIN", "CINCINNATI REDS":"CIN",
        "CLE":"CLE", "CLEVELAND GUARDIANS":"CLE", "CLEVELAND INDIANS":"CLE",
        "COL":"COL", "COLORADO ROCKIES":"COL", "DET":"DET", "DETROIT TIGERS":"DET",
        "HOU":"HOU", "HOUSTON ASTROS":"HOU", "KC":"KC", "KCR":"KC", "KANSAS CITY ROYALS":"KC",
        "LAA":"LAA", "LOS ANGELES ANGELS":"LAA", "LAD":"LAD", "LOS ANGELES DODGERS":"LAD",
        "MIA":"MIA", "MIAMI MARLINS":"MIA", "MIL":"MIL", "MILWAUKEE BREWERS":"MIL",
        "MIN":"MIN", "MINNESOTA TWINS":"MIN", "NYM":"NYM", "NEW YORK METS":"NYM",
        "NYY":"NYY", "NEW YORK YANKEES":"NYY", "PHI":"PHI", "PHILADELPHIA PHILLIES":"PHI",
        "PIT":"PIT", "PITTSBURGH PIRATES":"PIT", "SD":"SD", "SDP":"SD", "SAN DIEGO PADRES":"SD",
        "SEA":"SEA", "SEATTLE MARINERS":"SEA", "SF":"SF", "SFG":"SF", "SAN FRANCISCO GIANTS":"SF",
        "STL":"STL", "ST. LOUIS CARDINALS":"STL", "TB":"TB", "TBR":"TB", "TAMPA BAY RAYS":"TB",
        "TEX":"TEX", "TEXAS RANGERS":"TEX", "TOR":"TOR", "TORONTO BLUE JAYS":"TOR",
        "WSH":"WSH", "WSN":"WSH", "WASHINGTON NATIONALS":"WSH",
    }


def normalize_code(x):
    s = str(x).upper().strip()
    return team_code_map().get(s, s)


def build_game_key_from_normalized(games_path: Path):
    g = pd.read_parquet(games_path)
    need = ["game_id", "game_date", "away_team", "home_team", "away_team_id", "home_team_id"]
    missing = [c for c in need if c not in g.columns]
    if missing:
        raise ValueError(f"Normalized games missing columns: {missing}")
    g["game_date"] = pd.to_datetime(g["game_date"], errors="coerce").dt.normalize()
    g["away_code"] = g["away_team"].map(normalize_code)
    g["home_code"] = g["home_team"].map(normalize_code)
    g = g.sort_values(["game_date", "away_code", "home_code", "game_id"]).copy()
    g["game_number_join"] = g.groupby(["game_date", "away_code", "home_code"]).cumcount()
    return g[["game_id", "game_date", "away_code", "home_code", "game_number_join", "away_team_id", "home_team_id"]].copy()


def join_master_to_games(master, games):
    m = master.copy()
    if "game_number" in m.columns:
        m["game_number_join"] = pd.to_numeric(m["game_number"], errors="coerce").fillna(0).astype(int)
    else:
        m = m.sort_values(["game_date", "away_team_code", "home_team_code"]).copy()
        m["game_number_join"] = m.groupby(["game_date", "away_team_code", "home_team_code"]).cumcount()
    x = m.merge(
        games,
        left_on=["game_date", "away_team_code", "home_team_code", "game_number_join"],
        right_on=["game_date", "away_code", "home_code", "game_number_join"],
        how="left",
        validate="one_to_one",
    )
    miss = x["game_id"].isna()
    if miss.any():
        counts = games.groupby(["game_date", "away_code", "home_code"]).size().rename("n").reset_index()
        unique = games.merge(counts[counts["n"] == 1], on=["game_date", "away_code", "home_code"])
        fb = m.loc[miss, :].merge(
            unique,
            left_on=["game_date", "away_team_code", "home_team_code"],
            right_on=["game_date", "away_code", "home_code"],
            how="left",
        )
        for c in ["game_id", "away_team_id", "home_team_id"]:
            x.loc[miss, c] = fb[c].to_numpy()
    return x


def team_feature_table(path: Path):
    t = pd.read_parquet(path)
    t["as_of_date"] = pd.to_datetime(t["as_of_date"], errors="coerce").dt.normalize()
    wanted = [f"365d_ev_{e}_rate_shrunk" for e in EVENTS]
    missing = [c for c in wanted if c not in t.columns]
    if missing:
        raise ValueError(f"Team as-of store missing 365d rate columns: {missing}")
    keep = ["team_id", "team_role", "as_of_date", "365d_reliability"] + wanted
    return t[keep].copy(), wanted


def add_side_context(df, team, rate_cols):
    out = df.copy()
    specs = [
        ("away_team_id", "batting", "away_bat"),
        ("home_team_id", "batting", "home_bat"),
        ("away_team_id", "pitching_allowed", "away_pit"),
        ("home_team_id", "pitching_allowed", "home_pit"),
    ]
    for key, role, prefix in specs:
        sub = team[team["team_role"] == role].drop(columns=["team_role"]).copy()
        ren = {c: f"{prefix}_{c}" for c in ["365d_reliability"] + rate_cols}
        sub = sub.rename(columns=ren)
        out = out.merge(sub, left_on=[key, "game_date"], right_on=["team_id", "as_of_date"], how="left")
        out = out.drop(columns=["team_id", "as_of_date"], errors="ignore")

    features = []
    for event, c in zip(EVENTS, rate_cols):
        top = 0.5 * (pd.to_numeric(out[f"away_bat_{c}"], errors="coerce") + pd.to_numeric(out[f"home_pit_{c}"], errors="coerce"))
        bot = 0.5 * (pd.to_numeric(out[f"home_bat_{c}"], errors="coerce") + pd.to_numeric(out[f"away_pit_{c}"], errors="coerce"))
        out[f"ctx_{event}_mean"] = 0.5 * (top + bot)
        out[f"ctx_{event}_half_gap"] = (top - bot).abs()
        features.extend([f"ctx_{event}_mean", f"ctx_{event}_half_gap"])
    rel_cols = [f"{p}_365d_reliability" for p in ["away_bat", "home_bat", "away_pit", "home_pit"]]
    out["ctx_reliability_mean"] = out[rel_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    features.append("ctx_reliability_mean")
    return out, features


def replay(df, features, prior_strength=100.0, ridge_lambda=10.0):
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    pred_parts, metric_rows, coef_rows = [], [], []
    for test_season in seasons[1:]:
        train = df[df["season"] < test_season].copy()
        test = df[df["season"] == test_season].copy()
        if train.empty or test.empty:
            continue
        broad, priors = fit_total_prior(train, prior_strength)
        p_tr, _ = map_total_prior(train["opening_total"], broad, priors)
        p_te, buckets = map_total_prior(test["opening_total"], broad, priors)
        Xtr_raw = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        Xte_raw = test[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        sc = Standardizer().fit(Xtr_raw)
        Xtr = sc.transform(Xtr_raw)
        Xte = sc.transform(Xte_raw)
        beta = fit_ridge_offset(Xtr, train["actual_over"].to_numpy(float), logit(p_tr), ridge_lambda)
        pred = logistic(logit(p_te) + Xte @ beta)
        y = test["actual_over"].to_numpy(float)
        base_ll, ctx_ll = log_loss(y, p_te), log_loss(y, pred)
        base_br, ctx_br = brier(y, p_te), brier(y, pred)
        metric_rows.append({
            "test_season": test_season,
            "train_seasons": ",".join(str(s) for s in seasons if s < test_season),
            "n": len(test),
            "baseline_log_loss": base_ll,
            "context_log_loss": ctx_ll,
            "log_loss_improvement": base_ll - ctx_ll,
            "baseline_brier": base_br,
            "context_brier": ctx_br,
            "brier_improvement": base_br - ctx_br,
        })
        for f, b in zip(features, beta):
            coef_rows.append({"test_season": test_season, "feature": f, "standardized_beta": float(b)})
        p = test[["game_id", "game_date", "season", "away_team_code", "home_team_code", "opening_total", "i2_runs", "actual_over"]].copy()
        p["baseline_prediction"] = p_te
        p["prediction"] = pred
        p["matchup_delta"] = pred - p_te
        p["prior_bucket_used"] = buckets
        p["prediction_class"] = "STRICT_ASOF_TEAM_CONTEXT"
        pred_parts.append(p)
    return (
        pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame(),
        pd.DataFrame(metric_rows),
        pd.DataFrame(coef_rows),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--team-asof", required=True)
    ap.add_argument("--output-dir", default="data/derived/i2/strict_asof_replay")
    ap.add_argument("--prior-strength", type=float, default=100.0)
    ap.add_argument("--ridge-lambda", type=float, default=10.0)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    master = canonical_master(Path(args.master))
    games = build_game_key_from_normalized(Path(args.games))
    joined = join_master_to_games(master, games)
    before = len(joined)
    joined = joined[joined["game_id"].notna()].copy()
    join_rate = len(joined) / before if before else 0.0
    if join_rate < 0.98:
        raise SystemExit(f"Game join rate too low for threshold calibration: {join_rate:.3%}")

    team, rate_cols = team_feature_table(Path(args.team_asof))
    model_df, features = add_side_context(joined, team, rate_cols)
    model_df = model_df[model_df["ctx_reliability_mean"].notna()].copy()
    model_df = model_df[model_df["ctx_reliability_mean"] >= 0.10].copy()

    predictions, metrics, coefs = replay(model_df, features, args.prior_strength, args.ridge_lambda)
    if predictions.empty:
        raise SystemExit("No chronological OOS predictions generated.")

    predictions.to_csv(outdir / "strict_oos_predictions.csv", index=False)
    metrics.to_csv(outdir / "walk_forward_metrics.csv", index=False)
    coefs.to_csv(outdir / "coefficients_by_fold.csv", index=False)
    model_df[["game_id", "game_date", "season", "away_team_code", "home_team_code", "opening_total", "i2_runs", "actual_over", "ctx_reliability_mean"] + features].to_csv(
        outdir / "strict_asof_feature_matrix.csv.gz", index=False, compression="gzip"
    )

    manifest = {
        "status": "PASS",
        "prediction_class": "STRICT_ASOF_TEAM_CONTEXT",
        "market_isolation": {
            "full_game_total_point_used": True,
            "i2_derivative_prices_used": False,
            "moneyline_used": False,
            "runline_used": False,
            "juice_used": False,
        },
        "identity_inputs": {
            "historical_lineup_identity_used": False,
            "historical_actual_starter_identity_used": False,
        },
        "stat_cutoff": "team rates strictly prior-date; same-day prior games excluded by source builder",
        "master_game_rows": int(before),
        "joined_to_statsapi_games": int(len(joined)),
        "join_rate": join_rate,
        "model_rows_after_reliability_gate": int(len(model_df)),
        "oos_predictions": int(len(predictions)),
        "oos_seasons": sorted(int(s) for s in predictions["season"].unique()),
        "features": features,
        "prior_strength": args.prior_strength,
        "ridge_lambda": args.ridge_lambda,
        "note": "First leakage-safe historical replay layer. It sacrifices lineup/starter specificity rather than silently using retrospective identities as pregame-known.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Chronological direct-I2 feature ablation audit.

Purpose
-------
Measure whether each direct-I2 feature family improves held-out I2 PA event
prediction when the model is refit without that feature. This is a screening
stage for later full-I2 replay, not a production promotion test.

Governance
----------
- Baseball inputs only; no market data.
- Fixed governed hyperparameters (C=0.05, half-life=730 by default).
- Each target season is scored as one full-season holdout.
- Training uses only rows dated before January 1 of the target season.
- No target-season outcomes are used to select features or hyperparameters.
- 2026 is replication/sensitivity and must not select the final feature set.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import fit_i2_vnext as base


BASE_CATS = ["batter", "pitcher", "platoon", "home_team"]
BASE_NUMS = [f"arsenal_x_{p}" for p in base.PLATOONS]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--arsenal-dir", type=Path, required=True)
    p.add_argument("--target-season", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--half-life", type=float, default=730.0)
    p.add_argument("--c", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=5000)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def prepare(path: Path, arsenal_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="raise")
    df = df[df["event_class"].isin(base.EVENTS)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)

    numeric = df.copy()
    numeric["batter"] = pd.to_numeric(numeric["batter"], errors="raise").astype(int)
    numeric["pitcher"] = pd.to_numeric(numeric["pitcher"], errors="raise").astype(int)
    numeric = base.add_arsenal_feature(numeric, arsenal_dir)
    df["arsenal_matchup_xwoba"] = numeric["arsenal_matchup_xwoba"].to_numpy()

    df["batter"] = pd.to_numeric(df["batter"], errors="raise").astype(int).astype(str)
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="raise").astype(int).astype(str)
    df["platoon"] = df["platoon"].fillna("?v?").astype(str)
    df["home_team"] = df["home_team"].fillna("UNKNOWN").astype(str)
    for platoon in base.PLATOONS:
        df[f"arsenal_x_{platoon}"] = np.where(
            df["platoon"] == platoon,
            df["arsenal_matchup_xwoba"],
            0.0,
        )
    if df[BASE_NUMS].isna().any().any():
        raise ValueError("Missing leakage-safe arsenal feature")
    return df


def fit_model(train: pd.DataFrame, cats: list[str], nums: list[str], c: float,
              half_life: float, max_iter: int):
    transformers = []
    if cats:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
            cats,
        ))
    if nums:
        transformers.append(("num", StandardScaler(with_mean=False), nums))
    if not transformers:
        raise ValueError("At least one feature is required")

    prep = ColumnTransformer(transformers, sparse_threshold=1.0)
    x = prep.fit_transform(train[cats + nums])
    model = LogisticRegression(
        C=c,
        solver="saga",
        max_iter=max_iter,
        tol=1e-4,
        random_state=73,
    )
    model.fit(
        x,
        train["event_class"].astype(str),
        sample_weight=base.recency_weights(train["game_date"], half_life),
    )
    used = max(int(v) for v in np.atleast_1d(model.n_iter_))
    if used >= max_iter:
        raise RuntimeError(
            f"Nonconvergent ablation fit: cats={cats}, nums={nums}, "
            f"iterations={used}, max_iter={max_iter}"
        )
    return prep, model, used


def row_losses(prep, model, test: pd.DataFrame, cats: list[str], nums: list[str]):
    x = prep.transform(test[cats + nums])
    prob = model.predict_proba(x)
    classes = [str(x) for x in model.classes_]
    class_index = {c: i for i, c in enumerate(classes)}
    y = test["event_class"].astype(str).to_numpy()
    idx = np.asarray([class_index[v] for v in y], dtype=int)
    chosen = np.clip(prob[np.arange(len(test)), idx], 1e-12, 1.0)
    ll = -np.log(chosen)

    truth = np.zeros_like(prob)
    truth[np.arange(len(test)), idx] = 1.0
    bs = np.sum((prob - truth) ** 2, axis=1)
    return ll, bs


def clustered_bootstrap(delta: np.ndarray, dates: pd.Series, draws: int, seed: int):
    groups: dict[str, np.ndarray] = {}
    d = dates.astype(str).to_numpy()
    for day in np.unique(d):
        groups[str(day)] = delta[d == day]
    keys = sorted(groups)
    sums = np.asarray([groups[k].sum() for k in keys], dtype=float)
    counts = np.asarray([len(groups[k]) for k in keys], dtype=int)
    point = float(delta.mean())

    rng = np.random.default_rng(seed)
    sims = np.empty(draws, dtype=float)
    for i in range(draws):
        take = rng.integers(0, len(keys), size=len(keys))
        sims[i] = sums[take].sum() / counts[take].sum()
    lo, hi = np.quantile(sims, [0.025, 0.975])
    return {
        "estimate": point,
        "ci95": [float(lo), float(hi)],
        "dates": len(keys),
        "draws": draws,
        "method": "paired calendar-date cluster percentile bootstrap; conditional on fitted models",
    }


def main():
    a = parse_args()
    df = prepare(a.dataset, a.arsenal_dir)
    cutoff = pd.Timestamp(f"{a.target_season}-01-01")
    train = df[df["game_date"] < cutoff].copy()
    test = df[df["season"] == a.target_season].copy()
    if train.empty or test.empty:
        raise ValueError("Missing train or test rows")
    if train["game_date"].max() >= cutoff:
        raise ValueError("Training cutoff leakage")
    if test["game_date"].min() < cutoff:
        raise ValueError("Target-season test boundary failure")

    variants = {
        "baseline": (BASE_CATS, BASE_NUMS),
        "drop_batter_identity": ([x for x in BASE_CATS if x != "batter"], BASE_NUMS),
        "drop_pitcher_identity": ([x for x in BASE_CATS if x != "pitcher"], BASE_NUMS),
        "drop_batter_pitcher_identity": (
            [x for x in BASE_CATS if x not in {"batter", "pitcher"}],
            BASE_NUMS,
        ),
        "drop_platoon": ([x for x in BASE_CATS if x != "platoon"], BASE_NUMS),
        "drop_home_team_nuisance": ([x for x in BASE_CATS if x != "home_team"], BASE_NUMS),
        "drop_arsenal_all": (BASE_CATS, []),
    }
    for p in base.PLATOONS:
        variants[f"drop_arsenal_{p}"] = (BASE_CATS, [x for x in BASE_NUMS if x != f"arsenal_x_{p}"])

    fitted = {}
    for name, (cats, nums) in variants.items():
        prep, model, iterations = fit_model(
            train, list(cats), list(nums), a.c, a.half_life, a.max_iter
        )
        ll, bs = row_losses(prep, model, test, list(cats), list(nums))
        fitted[name] = {
            "cats": list(cats),
            "nums": list(nums),
            "iterations": iterations,
            "logloss_rows": ll,
            "brier_rows": bs,
            "logloss": float(ll.mean()),
            "brier": float(bs.mean()),
        }

    baseline = fitted["baseline"]
    results = {}
    for k, row in fitted.items():
        if k == "baseline":
            continue
        dll = row["logloss_rows"] - baseline["logloss_rows"]
        dbs = row["brier_rows"] - baseline["brier_rows"]
        results[k] = {
            "removed_feature_helps_if_positive": True,
            "candidate_minus_baseline_logloss": float(dll.mean()),
            "candidate_minus_baseline_brier": float(dbs.mean()),
            "logloss_uncertainty": clustered_bootstrap(
                dll, test["game_date"], a.bootstrap, 20261003 + a.target_season
            ),
            "brier_uncertainty": clustered_bootstrap(
                dbs, test["game_date"], a.bootstrap, 20261103 + a.target_season
            ),
            "candidate_logloss": row["logloss"],
            "candidate_brier": row["brier"],
            "iterations": row["iterations"],
            "active_categorical_features": row["cats"],
            "active_numeric_features": row["nums"],
        }

    payload = {
        "version": "i2-vnext-direct-feature-ablation-full-season-v1",
        "market_inputs_used": False,
        "target": "I2 terminal PA multiclass event",
        "target_season": a.target_season,
        "selection_role": "replication_only" if a.target_season == 2026 else "development",
        "training_cutoff_exclusive": cutoff.date().isoformat(),
        "training": {
            "n": int(len(train)),
            "start": train["game_date"].min().date().isoformat(),
            "end": train["game_date"].max().date().isoformat(),
            "seasons": sorted(int(x) for x in train["season"].unique()),
        },
        "test": {
            "n": int(len(test)),
            "start": test["game_date"].min().date().isoformat(),
            "end": test["game_date"].max().date().isoformat(),
            "season": a.target_season,
        },
        "hyperparameters": {
            "C": a.c,
            "half_life_days": a.half_life,
            "max_iter": a.max_iter,
            "reselected": False,
        },
        "baseline": {
            "logloss": baseline["logloss"],
            "brier": baseline["brier"],
            "iterations": baseline["iterations"],
            "categorical_features": BASE_CATS,
            "numeric_features": BASE_NUMS,
        },
        "ablations": results,
        "governance": {
            "target_season_outcomes_used_for_training": False,
            "target_season_outcomes_used_for_feature_selection_before_scoring": False,
            "market_inputs_used": False,
            "calendar_month_feature_or_control": False,
            "full_season_holdout": True,
            "note": "Screening stage only; full-I2 Under 0.5 replay remains the decision target.",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "target_season": a.target_season,
        "train_n": len(train),
        "test_n": len(test),
        "baseline_logloss": baseline["logloss"],
        "baseline_brier": baseline["brier"],
        "deltas": {k: {
            "logloss": v["candidate_minus_baseline_logloss"],
            "brier": v["candidate_minus_baseline_brier"],
        } for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()

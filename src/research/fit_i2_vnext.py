#!/usr/bin/env python3
"""Fit the lean direct-I2 vNext plate-appearance event model.

One jointly regularized multinomial model:
- batter MLBAM identity
- pitcher MLBAM identity
- batter/pitcher handedness interaction
- one pitcher-arsenal x batter pitch-type response feature

Recency half-life and L2 regularization are selected chronologically.
No PA-level post-calibration is applied; the only downstream shrinkage is the
final full-I2 OOS calibration curve.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder

EVENTS = [
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
]
CAT = ["batter", "pitcher", "platoon"]
NUM = ["arsenal_z"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/derived/i2_vnext/i2_pa_statcast.csv"))
    p.add_argument("--arsenal-dir", type=Path, default=Path("data/derived/i2_vnext/arsenal"))
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/i2_vnext_event_model.json"))
    p.add_argument("--live-arsenal-output", type=Path, default=Path("data/derived/i2_vnext/live_arsenal_profile.json"))
    p.add_argument("--half-lives", default="180,365,730,1460")
    p.add_argument("--c-grid", default="0.05,0.2,1.0")
    return p.parse_args()


def read_arsenal(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    needed = {"player_id", "pitch_type", "pitches", "est_woba"}
    if not needed.issubset(x.columns):
        raise RuntimeError(f"Arsenal file {path} missing {needed - set(x.columns)}")
    x["player_id"] = pd.to_numeric(x["player_id"], errors="coerce").astype("Int64")
    x["pitches"] = pd.to_numeric(x["pitches"], errors="coerce").fillna(0.0)
    x["est_woba"] = pd.to_numeric(x["est_woba"], errors="coerce")
    x["pitch_type"] = x["pitch_type"].astype(str)
    return x[x["player_id"].notna() & x["pitch_type"].notna()].copy()


def arsenal_maps(batter_file: Path, pitcher_file: Path):
    b = read_arsenal(batter_file)
    p = read_arsenal(pitcher_file)

    batter_x = b.dropna(subset=["est_woba"]).groupby(
        ["player_id", "pitch_type"], observed=True
    )["est_woba"].mean()
    league_x = (
        b.dropna(subset=["est_woba"])
        .groupby("pitch_type", observed=True)
        .apply(
            lambda g: np.average(g["est_woba"], weights=np.maximum(g["pitches"], 1)),
            include_groups=False,
        )
        .to_dict()
    )
    global_x = (
        float(np.average(b["est_woba"].dropna()))
        if b["est_woba"].notna().any()
        else 0.320
    )

    pg = p.groupby(["player_id", "pitch_type"], observed=True)["pitches"].sum().reset_index()
    totals = pg.groupby("player_id", observed=True)["pitches"].transform("sum")
    pg["usage"] = np.where(totals > 0, pg["pitches"] / totals, 0.0)
    pitcher_usage = {
        (int(r.player_id), str(r.pitch_type)): float(r.usage)
        for r in pg.itertuples(index=False)
    }
    types_by_pitcher: dict[int, list[str]] = {}
    for pid, group in pg.groupby("player_id", observed=True):
        types_by_pitcher[int(pid)] = list(group["pitch_type"].astype(str))

    league_usage_raw = p.groupby("pitch_type", observed=True)["pitches"].sum()
    total = float(league_usage_raw.sum())
    league_usage = {
        str(k): float(v / total) for k, v in league_usage_raw.items()
    } if total > 0 else {}

    return (
        {(int(pid), str(pt)): float(v) for (pid, pt), v in batter_x.items()},
        pitcher_usage,
        types_by_pitcher,
        league_x,
        global_x,
        league_usage,
    )


def matchup_score(batter: int, pitcher: int, maps) -> float:
    batter_x, pitcher_usage, types_by_pitcher, league_x, global_x, league_usage = maps
    pitch_types = types_by_pitcher.get(pitcher)
    if pitch_types:
        weights = [(pt, pitcher_usage.get((pitcher, pt), 0.0)) for pt in pitch_types]
    else:
        weights = list(league_usage.items())
    total = sum(w for _, w in weights)
    if total <= 0:
        return global_x
    return sum(
        (w / total) * batter_x.get((batter, pt), league_x.get(pt, global_x))
        for pt, w in weights
    )


def add_arsenal_feature(df: pd.DataFrame, arsenal_dir: Path) -> pd.DataFrame:
    out = df.copy()
    out["arsenal_matchup_xwoba"] = np.nan
    cache = {}
    for season in sorted(out["season"].unique()):
        prior = int(season) - 1
        batter_file = arsenal_dir / f"batter_{prior}.csv"
        pitcher_file = arsenal_dir / f"pitcher_{prior}.csv"
        if not batter_file.exists() or not pitcher_file.exists():
            raise RuntimeError(
                f"Missing leakage-safe prior-season arsenal files for {season}: "
                f"{batter_file}, {pitcher_file}"
            )
        maps = cache.setdefault(prior, arsenal_maps(batter_file, pitcher_file))
        mask = out["season"] == season
        pairs = out.loc[mask, ["batter", "pitcher"]]
        out.loc[mask, "arsenal_matchup_xwoba"] = [
            matchup_score(int(b), int(p), maps)
            for b, p in pairs.itertuples(index=False, name=None)
        ]
    return out


def recency_weights(dates: pd.Series, half_life: float) -> np.ndarray:
    d = pd.to_datetime(dates)
    age = (d.max() - d).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age / half_life)


def multiclass_brier(y: pd.Series, prob: np.ndarray, classes: list[str]) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    truth = np.zeros_like(prob)
    for row, value in enumerate(y.astype(str)):
        if value in idx:
            truth[row, idx[value]] = 1.0
    return float(np.mean(np.sum((prob - truth) ** 2, axis=1)))


def fit_one(train: pd.DataFrame, c: float, half_life: float):
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64), CAT),
            ("num", "passthrough", NUM),
        ],
        sparse_threshold=1.0,
    )
    x = prep.fit_transform(train[CAT + NUM])
    model = LogisticRegression(
        C=c,
        solver="saga",
        max_iter=800,
        tol=1e-4,
        random_state=73,
    )
    model.fit(
        x,
        train["event_class"].astype(str),
        sample_weight=recency_weights(train["game_date"], half_life),
    )
    return prep, model


def score(prep, model, test: pd.DataFrame) -> dict:
    x = prep.transform(test[CAT + NUM])
    p = model.predict_proba(x)
    classes = [str(x) for x in model.classes_]
    return {
        "n": int(len(test)),
        "logloss": float(log_loss(test["event_class"].astype(str), p, labels=model.classes_)),
        "brier_multiclass": multiclass_brier(test["event_class"], p, classes),
    }


def validation_folds(df: pd.DataFrame):
    folds = []
    for test_year in sorted(int(x) for x in df["season"].unique()):
        if test_year != 2024:
            continue
        train = df[df["season"] < test_year]
        test = df[df["season"] == test_year]
        if len(train) and len(test):
            folds.append((test_year, train, test))
    if not folds:
        raise RuntimeError("No chronological 2024 hyperparameter-selection fold available")
    return folds


def serialize_model(prep, model, mean: float, sd: float, selected: dict, trials: list[dict]) -> dict:
    names = [str(x) for x in prep.get_feature_names_out()]
    artifact = {
        "version": "i2-vnext-direct-talent-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_inputs_used": False,
        "target": "I2 terminal PA event class",
        "classes": [str(x) for x in model.classes_],
        "categorical_features": CAT,
        "numeric_features": NUM,
        "arsenal_feature": {
            "name": "arsenal_matchup_xwoba",
            "historical_source_rule": "prior-season Savant pitch-arsenal stats",
            "live_source_rule": "current YTD Savant pitch-arsenal snapshot at cutoff",
            "mean": mean,
            "sd": sd,
        },
        "selected": {**selected, "selection_year": 2024},
        "chronological_validation": trials,
        "final_calibration": {
            "status": "PENDING_FULL_I2_OOS_CURVE",
            "rule": (
                "No PA-level probability calibration; calibrate full I2 Under probability "
                "once after historical full-model replay."
            ),
        },
        "intercepts": {
            str(cls): float(model.intercept_[i])
            for i, cls in enumerate(model.classes_)
        },
        "coefficients": {},
    }
    for i, cls in enumerate(model.classes_):
        artifact["coefficients"][str(cls)] = {
            name: float(value)
            for name, value in zip(names, model.coef_[i])
            if abs(float(value)) > 1e-12
        }
    return artifact


def live_arsenal_payload(year: int, arsenal_dir: Path) -> dict:
    maps = arsenal_maps(
        arsenal_dir / f"batter_{year}.csv",
        arsenal_dir / f"pitcher_{year}.csv",
    )
    batter_x, pitcher_usage, types_by_pitcher, league_x, global_x, league_usage = maps
    batter_ids = sorted({pid for pid, _ in batter_x})
    return {
        "version": "i2-vnext-live-arsenal-v1",
        "season": year,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batter_xwoba_by_pitch": {
            str(pid): {
                pt: value for (p, pt), value in batter_x.items() if p == pid
            }
            for pid in batter_ids
        },
        "pitcher_usage_by_pitch": {
            str(pid): {
                pt: pitcher_usage.get((pid, pt), 0.0)
                for pt in types_by_pitcher.get(pid, [])
            }
            for pid in sorted(types_by_pitcher)
        },
        "league_xwoba_by_pitch": league_x,
        "league_global_xwoba": global_x,
        "league_pitch_usage": league_usage,
    }


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.dataset)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna() & df["event_class"].isin(EVENTS)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)

    # numeric IDs for arsenal join
    numeric = df.copy()
    numeric["batter"] = pd.to_numeric(numeric["batter"], errors="raise").astype(int)
    numeric["pitcher"] = pd.to_numeric(numeric["pitcher"], errors="raise").astype(int)
    numeric = add_arsenal_feature(numeric, args.arsenal_dir)
    df["arsenal_matchup_xwoba"] = numeric["arsenal_matchup_xwoba"].to_numpy()

    # categorical IDs for direct player effects
    df["batter"] = pd.to_numeric(df["batter"], errors="raise").astype(int).astype(str)
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="raise").astype(int).astype(str)
    df["platoon"] = df["platoon"].fillna("?v?").astype(str)

    mean = float(df["arsenal_matchup_xwoba"].mean())
    sd = float(df["arsenal_matchup_xwoba"].std(ddof=0)) or 1.0
    df["arsenal_z"] = (df["arsenal_matchup_xwoba"] - mean) / sd

    half_lives = [float(x) for x in args.half_lives.split(",") if x]
    c_grid = [float(x) for x in args.c_grid.split(",") if x]
    folds = validation_folds(df)

    trials = []
    best = None
    for half_life in half_lives:
        for c in c_grid:
            fold_scores = []
            for year, train, test in folds:
                prep, model = fit_one(train, c, half_life)
                fold_scores.append({"test_year": year, **score(prep, model, test)})
            weighted_ll = float(np.average(
                [x["logloss"] for x in fold_scores],
                weights=[x["n"] for x in fold_scores],
            ))
            weighted_bs = float(np.average(
                [x["brier_multiclass"] for x in fold_scores],
                weights=[x["n"] for x in fold_scores],
            ))
            row = {
                "half_life_days": half_life,
                "C": c,
                "weighted_logloss": weighted_ll,
                "weighted_brier": weighted_bs,
                "folds": fold_scores,
            }
            trials.append(row)
            key = (weighted_ll, weighted_bs, half_life, c)
            if best is None or key < best[0]:
                best = (key, half_life, c)

    if best is None:
        raise RuntimeError("Hyperparameter selection failed")

    selected = {
        "half_life_days": best[1],
        "C": best[2],
        "selection_metric": "chronological weighted multiclass log loss",
    }
    prep, model = fit_one(df, best[2], best[1])
    artifact = serialize_model(prep, model, mean, sd, selected, trials)
    artifact["training"] = {
        "start": df["game_date"].min().date().isoformat(),
        "end": df["game_date"].max().date().isoformat(),
        "n": int(len(df)),
        "seasons": sorted(int(x) for x in df["season"].unique()),
        "event_counts": {
            str(k): int(v) for k, v in df["event_class"].value_counts().to_dict().items()
        },
        "batters": int(df["batter"].nunique()),
        "pitchers": int(df["pitcher"].nunique()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")

    live_year = int(df["season"].max())
    args.live_arsenal_output.parent.mkdir(parents=True, exist_ok=True)
    args.live_arsenal_output.write_text(
        json.dumps(live_arsenal_payload(live_year, args.arsenal_dir), separators=(",", ":")),
        encoding="utf-8",
    )

    print(json.dumps({
        "training": artifact["training"],
        "selected": selected,
        "best_validation_logloss": best[0][0],
        "best_validation_brier": best[0][1],
        "full_i2_calibration": artifact["final_calibration"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()

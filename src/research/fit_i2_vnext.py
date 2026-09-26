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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EVENTS = [
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
]
CAT = ["batter", "pitcher", "platoon", "home_team"]
PLATOONS = ["LvL", "LvR", "RvL", "RvR"]
NUM = [f"arsenal_x_{p}" for p in PLATOONS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("data/derived/i2_vnext/i2_pa_statcast.csv"))
    p.add_argument("--arsenal-dir", type=Path, default=Path("data/derived/i2_vnext/arsenal"))
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/i2_vnext_event_model.json"))
    p.add_argument("--live-output", type=Path, default=Path("data/derived/i2_vnext/i2_vnext_event_model_live.json"))
    p.add_argument("--live-arsenal-output", type=Path, default=Path("data/derived/i2_vnext/live_arsenal_profile.json"))
    p.add_argument("--replay-arsenal-output", type=Path, default=Path("data/derived/i2_vnext/arsenal_profile_2024.json"))
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


def arsenal_maps(
    batter_file: Path,
    pitcher_file: Path,
    pitcher_side_file: Path | None = None,
):
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

    pitcher_side_usage = {}
    types_by_pitcher_side: dict[tuple[int, str], list[str]] = {}
    league_usage_by_side: dict[str, dict[str, float]] = {}
    if pitcher_side_file is not None and pitcher_side_file.exists():
        side = pd.read_csv(pitcher_side_file)
        needed = {"pitcher", "stand", "pitch_type", "pitches"}
        if needed.issubset(side.columns):
            side["pitcher"] = pd.to_numeric(side["pitcher"], errors="coerce").astype("Int64")
            side["pitches"] = pd.to_numeric(side["pitches"], errors="coerce").fillna(0.0)
            side["stand"] = side["stand"].astype(str)
            side["pitch_type"] = side["pitch_type"].astype(str)
            side = side[
                side["pitcher"].notna()
                & side["stand"].isin(["L", "R"])
                & (side["pitches"] > 0)
            ].copy()
            if not side.empty:
                stot = side.groupby(["pitcher", "stand"], observed=True)["pitches"].transform("sum")
                side["usage"] = np.where(stot > 0, side["pitches"] / stot, 0.0)
                pitcher_side_usage = {
                    (int(r.pitcher), str(r.stand), str(r.pitch_type)): float(r.usage)
                    for r in side.itertuples(index=False)
                }
                for (pid, stand), group in side.groupby(["pitcher", "stand"], observed=True):
                    types_by_pitcher_side[(int(pid), str(stand))] = list(group["pitch_type"].astype(str))
                for stand, group in side.groupby("stand", observed=True):
                    lg = group.groupby("pitch_type", observed=True)["pitches"].sum()
                    denom = float(lg.sum())
                    if denom > 0:
                        league_usage_by_side[str(stand)] = {
                            str(k): float(v / denom) for k, v in lg.items()
                        }

    return (
        {(int(pid), str(pt)): float(v) for (pid, pt), v in batter_x.items()},
        pitcher_usage,
        types_by_pitcher,
        pitcher_side_usage,
        types_by_pitcher_side,
        league_x,
        global_x,
        league_usage,
        league_usage_by_side,
    )


def matchup_score(
    batter: int,
    pitcher: int,
    maps,
    batter_side: str | None = None,
) -> float:
    (
        batter_x,
        pitcher_usage,
        types_by_pitcher,
        pitcher_side_usage,
        types_by_pitcher_side,
        league_x,
        global_x,
        league_usage,
        league_usage_by_side,
    ) = maps

    side = batter_side if batter_side in {"L", "R"} else None
    side_types = types_by_pitcher_side.get((pitcher, side)) if side else None
    if side_types:
        weights = [
            (pt, pitcher_side_usage.get((pitcher, side, pt), 0.0))
            for pt in side_types
        ]
    else:
        pitch_types = types_by_pitcher.get(pitcher)
        if pitch_types:
            weights = [(pt, pitcher_usage.get((pitcher, pt), 0.0)) for pt in pitch_types]
        else:
            weights = list((league_usage_by_side.get(side) or league_usage).items())

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
        side_file = arsenal_dir / f"pitcher_usage_side_{prior}.csv"
        maps = cache.setdefault(
            prior,
            arsenal_maps(
                batter_file,
                pitcher_file,
                side_file if side_file.exists() else None,
            ),
        )
        mask = out["season"] == season
        pairs = out.loc[mask, ["batter", "pitcher", "stand"]]
        out.loc[mask, "arsenal_matchup_xwoba"] = [
            matchup_score(int(b), int(p), maps, str(stand))
            for b, p, stand in pairs.itertuples(index=False, name=None)
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
            ("num", StandardScaler(with_mean=False), NUM),
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
    # One bounded chronological fold that actually contains both prior-season
    # and current-season evidence in training. This makes the fitted recency
    # half-life answer the April-vs-late-season weighting question rather than
    # merely reweighting a single prior season.
    cutoff = pd.Timestamp("2024-06-30")
    train = df[df["game_date"] <= cutoff]
    test = df[
        (df["game_date"] > cutoff)
        & (df["game_date"] < pd.Timestamp("2025-01-01"))
    ]
    if train.empty or test.empty:
        raise RuntimeError("No chronological 2024H2 hyperparameter-selection fold available")
    return [("2024H2", train, test)]


def serialize_model(prep, model, selected: dict, trials: list[dict]) -> dict:
    names = [str(x) for x in prep.get_feature_names_out()]
    artifact = {
        "version": "i2-vnext-direct-talent-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_inputs_used": False,
        "target": "I2 terminal PA event class",
        "classes": [str(x) for x in model.classes_],
        "categorical_features": CAT,
        "numeric_features": NUM,
        "nuisance_controls": {
            "home_team": "fit-only park confounder control; coefficient intentionally omitted at neutral live inference before Savant park is applied once"
        },
        "arsenal_feature": {
            "name": "pitcher_arsenal_x_batter_pitch_response_x_platoon",
            "historical_source_rule": "prior-season Savant pitch-arsenal stats",
            "live_source_rule": "current YTD Savant pitch-arsenal snapshot at cutoff",
            "platoon_interaction_features": NUM,
            "scales": {
                feature: float(scale)
                for feature, scale in zip(NUM, prep.named_transformers_["num"].scale_)
            },
        },
        "selected": {
            **selected,
            "selection_period": "train through 2024-06-30; test 2024-07-01 through season end",
        },
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
    side_file = arsenal_dir / f"pitcher_usage_side_{year}.csv"
    maps = arsenal_maps(
        arsenal_dir / f"batter_{year}.csv",
        arsenal_dir / f"pitcher_{year}.csv",
        side_file if side_file.exists() else None,
    )
    (
        batter_x,
        pitcher_usage,
        types_by_pitcher,
        pitcher_side_usage,
        types_by_pitcher_side,
        league_x,
        global_x,
        league_usage,
        league_usage_by_side,
    ) = maps
    batter_ids = sorted({pid for pid, _ in batter_x})
    side_pitchers = sorted({pid for pid, _, _ in pitcher_side_usage})
    return {
        "version": "i2-vnext-live-arsenal-v2",
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
        "pitcher_usage_by_side": {
            str(pid): {
                side: {
                    pt: pitcher_side_usage.get((pid, side, pt), 0.0)
                    for pt in types_by_pitcher_side.get((pid, side), [])
                }
                for side in ("L", "R")
                if (pid, side) in types_by_pitcher_side
            }
            for pid in side_pitchers
        },
        "league_xwoba_by_pitch": league_x,
        "league_global_xwoba": global_x,
        "league_pitch_usage": league_usage,
        "league_pitch_usage_by_side": league_usage_by_side,
        "side_specific_usage_source": (
            "I2 Statcast pitches by actual batter side"
            if side_file.exists()
            else "unavailable; overall pitcher arsenal fallback"
        ),
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
    df["home_team"] = df["home_team"].fillna("UNKNOWN").astype(str)
    for platoon in PLATOONS:
        df[f"arsenal_x_{platoon}"] = np.where(
            df["platoon"] == platoon,
            df["arsenal_matchup_xwoba"],
            0.0,
        )

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
    # Freeze the raw PA model before consuming the full-model calibration/validation periods.
    # 2023 is estimation; 2024 selects hyperparameters and is then included in the frozen
    # raw model fit. 2025 is reserved for the full-I2 calibration curve and 2026 for
    # untouched full-model validation. Do not fit player effects on either holdout here.
    fit_df = df[df["season"] <= 2024].copy()
    if fit_df.empty or 2024 not in set(fit_df["season"]):
        raise RuntimeError("Frozen PA fit requires 2023-2024 data with 2024 present")

    prep, model = fit_one(fit_df, best[2], best[1])
    artifact = serialize_model(prep, model, selected, trials)
    artifact["holdout_policy"] = {
        "raw_pa_estimation_years": sorted(int(x) for x in fit_df["season"].unique()),
        "hyperparameter_selection_period": "2024H2",
        "full_i2_calibration_year": 2025,
        "full_i2_calibration_fit_segment": "first chronological half",
        "full_i2_calibration_validation_segment": "second chronological half",
        "prospective_validation_start": "2026-09-26",
        "fit_uses_2025": False,
        "fit_uses_2026": False,
    }
    artifact["training"] = {
        "start": fit_df["game_date"].min().date().isoformat(),
        "end": fit_df["game_date"].max().date().isoformat(),
        "n": int(len(fit_df)),
        "seasons": sorted(int(x) for x in fit_df["season"].unique()),
        "event_counts": {
            str(k): int(v) for k, v in fit_df["event_class"].value_counts().to_dict().items()
        },
        "batters": int(fit_df["batter"].nunique()),
        "pitchers": int(fit_df["pitcher"].nunique()),
    }

    artifact["artifact_role"] = "FROZEN_HOLDOUT_EVALUATION"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")

    # Live production uses the identical frozen specification and hyperparameters,
    # refit on all baseball observations available through the dataset cutoff.
    live_prep, live_model = fit_one(df, best[2], best[1])
    live_artifact = serialize_model(live_prep, live_model, selected, trials)
    live_artifact["artifact_role"] = "LIVE_REFIT_FIXED_SPECIFICATION"
    live_artifact["holdout_policy"] = artifact["holdout_policy"]
    live_artifact["training"] = {
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
    args.live_output.parent.mkdir(parents=True, exist_ok=True)
    args.live_output.write_text(
        json.dumps(live_artifact, separators=(",", ":")), encoding="utf-8"
    )

    # 2024 prior-season arsenal is the leakage-safe feature snapshot for 2025
    # full-I2 calibration replay. The live profile remains current-season YTD.
    args.replay_arsenal_output.parent.mkdir(parents=True, exist_ok=True)
    args.replay_arsenal_output.write_text(
        json.dumps(live_arsenal_payload(2024, args.arsenal_dir), separators=(",", ":")),
        encoding="utf-8",
    )

    live_year = int(df["season"].max())
    args.live_arsenal_output.parent.mkdir(parents=True, exist_ok=True)
    args.live_arsenal_output.write_text(
        json.dumps(live_arsenal_payload(live_year, args.arsenal_dir), separators=(",", ":")),
        encoding="utf-8",
    )

    print(json.dumps({
        "frozen_training": artifact["training"],
        "live_training": live_artifact["training"],
        "selected": selected,
        "best_validation_logloss": best[0][0],
        "best_validation_brier": best[0][1],
        "full_i2_calibration": artifact["final_calibration"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()

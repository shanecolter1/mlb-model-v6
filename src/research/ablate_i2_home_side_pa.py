#!/usr/bin/env python3
"""Component-only home batting-side A/B; 2025 was already inspected.

The same frozen 2024-selected regularization and 2023–2024 training PAs are
used in both fits. No betting data, full-I2 calibration, or promotion is used.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from fit_i2_vnext import CAT, EVENTS, NUM, PLATOONS, add_arsenal_feature, fit_one


def prepare(dataset: Path, arsenal_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df[df["game_date"].notna() & df["event_class"].isin(EVENTS)].copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df = df[df["season"].isin((2023, 2024, 2025))].copy()
    if set(df["season"]) != {2023, 2024, 2025}:
        raise ValueError("Expected 2023–2025 Statcast I2 PAs")
    if not set(df["inning_topbot"]) <= {"Top", "Bot"}:
        raise ValueError("Invalid home batting-side field")
    numeric = df.copy()
    numeric["batter"] = pd.to_numeric(numeric["batter"], errors="raise").astype(int)
    numeric["pitcher"] = pd.to_numeric(numeric["pitcher"], errors="raise").astype(int)
    numeric = add_arsenal_feature(numeric, arsenal_dir)
    df["arsenal_matchup_xwoba"] = numeric["arsenal_matchup_xwoba"].to_numpy()
    df["batter"] = numeric["batter"].astype(str)
    df["pitcher"] = numeric["pitcher"].astype(str)
    df["platoon"] = df["platoon"].fillna("?v?").astype(str)
    df["home_team"] = df["home_team"].fillna("UNKNOWN").astype(str)
    df["home_batting_side"] = df["inning_topbot"].map({"Top": "AWAY", "Bot": "HOME"})
    for platoon in PLATOONS:
        df[f"arsenal_x_{platoon}"] = np.where(
            df["platoon"] == platoon, df["arsenal_matchup_xwoba"], 0.0
        )
    if df[NUM].isna().any().any():
        raise ValueError("Missing prior-season arsenal features")
    return df


def scored_rows(prep, model, test: pd.DataFrame, categorical: list[str]) -> tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(prep.transform(test[categorical + NUM]))
    labels = {str(value): i for i, value in enumerate(model.classes_)}
    indices = np.array([labels[str(value)] for value in test["event_class"]])
    loss = -np.log(np.clip(prob[np.arange(len(indices)), indices], 1e-12, 1))
    brier = np.sum(prob**2, axis=1) + 1 - 2 * prob[np.arange(len(indices)), indices]
    return loss, brier


def group_stats(frame: pd.DataFrame, index: np.ndarray) -> dict:
    chunk = frame.iloc[index]
    return {
        "n": len(chunk),
        "control_logloss": float(chunk["control_logloss"].mean()),
        "home_side_logloss": float(chunk["home_side_logloss"].mean()),
        "home_side_minus_control_logloss": float((chunk["home_side_logloss"] - chunk["control_logloss"]).mean()),
        "control_brier": float(chunk["control_brier"].mean()),
        "home_side_brier": float(chunk["home_side_brier"].mean()),
        "home_side_minus_control_brier": float((chunk["home_side_brier"] - chunk["control_brier"]).mean()),
    }


def compare(frame: pd.DataFrame) -> dict:
    groups = {"all": np.arange(len(frame))}
    for side in ("HOME", "AWAY"):
        groups[side.lower()] = np.flatnonzero(frame["home_batting_side"].to_numpy() == side)
    out = {name: group_stats(frame, ix) for name, ix in groups.items()}
    # Shared calendar dates cluster multiple PAs and games. CIs are descriptive
    # after the 2025 residual was inspected, conditional on fitted models.
    by_date: dict[str, list[int]] = defaultdict(list)
    for i, date in enumerate(frame["game_date"]):
        by_date[str(date.date())].append(i)
    delta = frame[["home_side_logloss", "home_side_brier"]].to_numpy() - frame[["control_logloss", "control_brier"]].to_numpy()
    totals = np.array([delta[ix].sum(axis=0) for ix in by_date.values()])
    counts = np.array([len(ix) for ix in by_date.values()])
    rng = np.random.default_rng(20260927)
    draws = np.empty((3000, 2))
    for i in range(3000):
        picks = rng.integers(0, len(counts), size=len(counts))
        draws[i] = totals[picks].sum(axis=0) / counts[picks].sum()
    interval = np.quantile(draws, [0.025, 0.975], axis=0)
    out["all"]["paired_date_cluster_ci95"] = {
        "logloss": interval[:, 0].tolist(), "brier": interval[:, 1].tolist()
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--arsenal-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/phase7d_home_side_pa_ablation.json"))
    args = p.parse_args()
    df = prepare(args.dataset, args.arsenal_dir)
    train = df[df["season"].isin((2023, 2024))].copy()
    test = df[df["season"] == 2025].copy()
    if train["game_date"].max() >= test["game_date"].min():
        raise ValueError("Train/test chronological split violated")
    candidates = {"control": CAT, "home_side": CAT + ["home_batting_side"]}
    for name, categorical in candidates.items():
        prep, model = fit_one(train, c=0.05, half_life=730, categorical_features=categorical)
        loss, brier = scored_rows(prep, model, test, categorical)
        test[f"{name}_logloss"] = loss
        test[f"{name}_brier"] = brier
    result = {
        "version": "i2-home-side-pa-ablation-v1",
        "market_inputs_used": False,
        "promotion_status": "SHADOW_COMPONENT_RESEARCH_ONLY",
        "training": {"seasons": [2023, 2024], "n": len(train), "end": str(train["game_date"].max().date())},
        "test": {"season": 2025, "n": len(test), "already_inspected": True},
        "fixed_hyperparameters": {"C": 0.05, "half_life_days": 730, "source": "2024 chronological selection; no refit based on this ablation"},
        "only_feature_difference": "Top/Bottom home batting-side categorical term inside the same joint multinomial PA model",
        "metrics": compare(test),
        "limitations": [
            "2025 outcomes were already inspected before this candidate was proposed; this is not an independent validation.",
            "PA event scoring does not establish full-I2 probability improvement or address park coverage.",
            "Frozen pre-2025 model tests the PA component; the deployed shadow runner uses monthly walk-forward refits.",
            "No calibration, thresholds, betting strategy or production model changed.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Walk-forward season-position candidate for full I2 Over 0.5.

The fixed early/late basis is fitted only on completed 2023–25 seasons.
Leave-one-season-out selects one ridge strength per candidate; 2026 is only
scored after candidate selection. Dates and outcomes never enter the pregame
base probabilities. This script does not alter production or select a ball
effect from a retrospectively discovered 2026 date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from audit_i2_date_regime import load_season


DEVELOPMENT = (2023, 2024, 2025)
HOLDOUT = 2026
RIDGE = (0.0, 0.0001, 0.001, 0.01, 0.1, 1.0)


def features(data: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "identity":
        return np.empty((len(data), 0))
    if mode == "constant":
        return np.ones((len(data), 1))
    if mode != "season_position":
        raise ValueError(mode)
    elapsed = data.elapsed.to_numpy(dtype=float)
    remaining = data.remaining.to_numpy(dtype=float)
    return np.column_stack((
        np.ones(len(data)),
        np.maximum(0.0, 1.0 - elapsed / 28.0),
        np.maximum(0.0, 1.0 - remaining / 42.0),
    ))


def fit(data: pd.DataFrame, mode: str, ridge: float) -> np.ndarray:
    x = features(data, mode)
    if x.shape[1] == 0:
        return np.empty(0)
    y = 1 - data.under_y.to_numpy()
    eta0 = logit(np.clip(1 - data.under_p.to_numpy(), 1e-9, 1 - 1e-9))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = eta0 + x @ beta
        p = expit(eta)
        value = float(np.mean(np.logaddexp(0, eta) - y * eta)
                      + ridge * np.sum(beta ** 2))
        gradient = (x.T @ (p - y)) / len(y) + 2 * ridge * beta
        return value, gradient

    result = minimize(objective, np.zeros(x.shape[1]), method="BFGS", jac=True,
                      options={"gtol": 1e-10})
    if not result.success and np.max(np.abs(objective(result.x)[1])) > 1e-7:
        raise RuntimeError(result.message)
    return result.x


def probabilities(data: pd.DataFrame, mode: str, beta: np.ndarray) -> np.ndarray:
    base = np.clip(1 - data.under_p.to_numpy(), 1e-9, 1 - 1e-9)
    if mode == "identity":
        return base
    return expit(logit(base) + features(data, mode) @ beta)


def scores(data: pd.DataFrame, p: np.ndarray) -> dict:
    y = 1 - data.under_y.to_numpy()
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {
        "games": len(y), "observed_over": float(y.mean()),
        "predicted_over": float(p.mean()),
        "actual_minus_predicted": float(np.mean(y - p)),
        "logloss": float(np.mean(np.logaddexp(0, logit(p)) - y * logit(p))),
        "brier": float(np.mean((y - p) ** 2)),
    }


def date_ci(data: pd.DataFrame, pred: np.ndarray, draws: int = 10000) -> dict:
    y = 1 - data.under_y.to_numpy()
    raw = np.clip(1 - data.under_p.to_numpy(), 1e-12, 1 - 1e-12)
    p = np.clip(pred, 1e-12, 1 - 1e-12)
    logdelta = (np.logaddexp(0, logit(p)) - y * logit(p)
                - np.logaddexp(0, logit(raw)) + y * logit(raw))
    brierdelta = (y - p) ** 2 - (y - raw) ** 2
    frame = pd.DataFrame({"date": data.date, "log": logdelta, "brier": brierdelta})
    daily = frame.groupby("date").agg(log=("log", "sum"),
                                       brier=("brier", "sum"), games=("log", "size"))
    clusters = daily.to_numpy()
    rng = np.random.default_rng(242026)
    samples = np.empty((draws, 2))
    for i in range(draws):
        sums = clusters[rng.integers(len(clusters), size=len(clusters))].sum(axis=0)
        samples[i] = sums[:2] / sums[2]
    return {"logloss_delta": float(logdelta.mean()),
            "brier_delta": float(brierdelta.mean()),
            "logloss_ci95_day_cluster": [float(v) for v in np.quantile(samples[:, 0], [.025, .975])],
            "brier_ci95_day_cluster": [float(v) for v in np.quantile(samples[:, 1], [.025, .975])]}


def read_data(bundles: dict[int, Path]) -> dict[int, pd.DataFrame]:
    years = {}
    for year, path in bundles.items():
        data = load_season(path, year)
        date = pd.to_datetime(data.date)
        data["elapsed"] = (date - date.min()).dt.days
        data["remaining"] = (date.max() - date).dt.days
        years[year] = data
    return years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="append", nargs=2, required=True,
                        metavar=("YEAR", "ZIP"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundles = {int(year): Path(path) for year, path in args.bundle}
    if set(bundles) != set(DEVELOPMENT + (HOLDOUT,)):
        raise ValueError("Requires complete 2023–2026 season bundles")
    years = read_data(bundles)
    raw_development = {}
    for year in DEVELOPMENT:
        data = years[year]
        raw_development[str(year)] = scores(data, probabilities(data, "identity", np.empty(0)))
    raw_pooled_logloss = sum(v["games"] * v["logloss"] for v in raw_development.values()) / sum(
        v["games"] for v in raw_development.values())
    selected = {}
    for mode in ("constant", "season_position"):
        choices = []
        for ridge in RIDGE:
            folds = {}
            total_loss = 0.0
            total_games = 0
            for test_year in DEVELOPMENT:
                train = pd.concat([years[year] for year in DEVELOPMENT if year != test_year])
                beta = fit(train, mode, ridge)
                score = scores(years[test_year], probabilities(years[test_year], mode, beta))
                folds[str(test_year)] = {"logloss": score["logloss"],
                                         "brier": score["brier"]}
                total_loss += score["logloss"] * score["games"]
                total_games += score["games"]
            choices.append({"ridge": ridge, "loso_logloss": total_loss / total_games,
                            "folds": folds})
        best = min(choices, key=lambda row: row["loso_logloss"])
        beta = fit(pd.concat([years[y] for y in DEVELOPMENT]), mode, best["ridge"])
        selected[mode] = {"development_choices": choices,
                          "selected_ridge": best["ridge"],
                          "selected_loso_logloss_delta_vs_raw": best["loso_logloss"] - raw_pooled_logloss,
                          "coefficients": [float(v) for v in beta]}

    holdout = years[HOLDOUT]
    candidates = {"identity": np.array([]),
                  **{mode: np.array(fit(pd.concat([years[y] for y in DEVELOPMENT]),
                                          mode, spec["selected_ridge"]))
                     for mode, spec in selected.items()}}
    comparisons = {}
    for mode, beta in candidates.items():
        pred = probabilities(holdout, mode, beta)
        result = {"full_season": scores(holdout, pred),
                  "versus_raw": date_ci(holdout, pred)}
        for name, mask in (
            ("before_may_25", holdout.date < "2026-05-25"),
            ("may_25_onward", holdout.date >= "2026-05-25"),
        ):
            result[name] = scores(holdout.loc[mask], pred[mask.to_numpy()])
        comparisons[mode] = result

    output = {
        "version": "i2-season-position-validation-v1",
        "market_inputs_used": False,
        "sample": "primary-home games; all contact types and scoring outcomes",
        "development_years": DEVELOPMENT, "replication_year": HOLDOUT,
        "basis": "logit(raw full-I2 Over) + intercept + early(max(0,1-elapsed/28)) + late(max(0,1-remaining/42))",
        "selection": "ridge by pooled 2023-25 leave-one-season-out log loss; 2026 scored after selection",
        "caution": "2026 outcomes were previously inspected; descriptive replication, not pristine holdout",
        "development_raw": {"by_year": raw_development,
                            "pooled_logloss": raw_pooled_logloss},
        "fits": selected, "replication": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

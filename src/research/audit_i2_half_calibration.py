#!/usr/bin/env python3
"""Estimate full-season top/bottom I2 calibration residuals.

Research-only diagnostic. The canonical replay probabilities are pregame and
market-isolated; observed half-inning runs enter only as targets. Calendar
month is never a feature, control, split, or adjustment variable.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit, logit


MATCHED = "RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cv-repeats", type=int, default=20)
    p.add_argument("--cv-folds", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=5000)
    return p.parse_args()


def build_rows(replay, inputs):
    if replay.get("market_inputs_used") is not False or inputs.get("market_inputs_used") is not False:
        raise ValueError("Market contamination")
    if replay.get("trials_per_game") != 10000:
        raise ValueError("Expected canonical 10,000-trial replay")
    if replay.get("n") != 2430 or replay.get("season") != 2025:
        raise ValueError("Expected complete 2025 replay")
    games = {g["gid"]: g for g in inputs["games"]}
    rows = []
    for pred in replay["predictions"]:
        game = games[pred["gid"]]
        for side, pkey, okey in (
            ("top", "top2_score_probability", "top2_runs"),
            ("bottom", "bottom2_score_probability", "bottom2_runs"),
        ):
            p = float(pred[pkey])
            if not 0 < p < 1:
                raise ValueError(f"Invalid probability: {pred['gid']}/{side}")
            rows.append({
                "gid": pred["gid"],
                "date": pred["date"],
                "side": side,
                "bottom": int(side == "bottom"),
                "p": p,
                "y": int(game["observed"][okey] > 0),
                "park_status": pred["park_status"],
                "raw_under05": float(pred["raw_under05"]),
                "observed_under05": int(pred["observed_under05"]),
            })
    frame = pd.DataFrame(rows)
    frame["x"] = logit(frame["p"].clip(1e-6, 1 - 1e-6))
    return frame


def fit(frame, mode):
    if mode == "bottom_only":
        X = frame[["bottom"]].to_numpy()
        offset = frame["x"].to_numpy()
        names = ["bottom_offset"]
    elif mode == "half_offset":
        X = np.column_stack((np.ones(len(frame)), frame["bottom"]))
        offset = frame["x"].to_numpy()
        names = ["top_offset", "bottom_minus_top_offset"]
    elif mode == "half_curve":
        xb = frame["x"] * frame["bottom"]
        X = np.column_stack((np.ones(len(frame)), frame["x"], frame["bottom"], xb))
        offset = None
        names = ["top_intercept", "top_slope", "bottom_intercept_delta", "bottom_slope_delta"]
    else:
        raise ValueError(mode)
    model = sm.GLM(frame["y"].to_numpy(), X, family=sm.families.Binomial(), offset=offset)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": frame["date"].to_numpy()})
    return result, names


def predict(result, frame, mode):
    params = np.asarray(result.params)
    x = frame["x"].to_numpy()
    bottom = frame["bottom"].to_numpy()
    if mode == "bottom_only":
        eta = x + params[0] * bottom
    elif mode == "half_offset":
        eta = x + params[0] + params[1] * bottom
    else:
        eta = params[0] + params[1] * x + params[2] * bottom + params[3] * x * bottom
    return expit(eta)


def metric(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return {
        "n": int(len(y)),
        "mean_prediction": float(p.mean()),
        "observed": float(y.mean()),
        "bias_pred_minus_actual": float((p - y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
    }


def losses(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return (p - y) ** 2, -y * np.log(p) - (1 - y) * np.log1p(-p)


def date_bootstrap(frame, delta_brier, delta_logloss, draws, seed):
    tmp = pd.DataFrame({
        "date": frame["date"].to_numpy(),
        "db": delta_brier,
        "dl": delta_logloss,
    })
    grouped = tmp.groupby("date").agg({"db": ["sum", "count"], "dl": "sum"})
    clusters = np.column_stack((
        grouped[("db", "sum")],
        grouped[("dl", "sum")],
        grouped[("db", "count")],
    ))
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, 2))
    for i in range(draws):
        picked = clusters[rng.integers(0, len(clusters), size=len(clusters))].sum(axis=0)
        samples[i] = (picked[0] / picked[2], picked[1] / picked[2])
    return {
        "brier": [float(x) for x in np.quantile(samples[:, 0], [0.025, 0.975])],
        "logloss": [float(x) for x in np.quantile(samples[:, 1], [0.025, 0.975])],
    }


def repeated_group_date_cv(frame, mode, repeats, folds, seed=1701):
    dates = np.array(sorted(frame["date"].unique()))
    if folds < 2 or folds > len(dates):
        raise ValueError("Invalid CV folds")
    rng = np.random.default_rng(seed)
    total = np.zeros(len(frame))
    count = np.zeros(len(frame))
    for _ in range(repeats):
        perm = dates.copy()
        rng.shuffle(perm)
        mapping = {date: i % folds for i, date in enumerate(perm)}
        assignment = frame["date"].map(mapping).to_numpy()
        for fold in range(folds):
            test = assignment == fold
            result, _ = fit(frame.loc[~test], mode)
            total[test] += predict(result, frame.loc[test], mode)
            count[test] += 1
    if not np.all(count == repeats):
        raise ValueError("Incomplete repeated CV")
    return total / count


def full_under_metrics(frame, q_score, replay, draws):
    temp = frame[["gid", "date", "side"]].copy()
    temp["q"] = q_score
    pivot = temp.pivot(index="gid", columns="side", values="q")
    meta = {p["gid"]: p for p in replay["predictions"]}
    gids = list(pivot.index)
    adjusted = np.asarray([(1 - pivot.loc[g, "top"]) * (1 - pivot.loc[g, "bottom"]) for g in gids])
    raw = np.asarray([meta[g]["raw_under05"] for g in gids])
    y = np.asarray([meta[g]["observed_under05"] for g in gids])
    dates = np.asarray([meta[g]["date"] for g in gids])
    b0, l0 = losses(y, raw)
    b1, l1 = losses(y, adjusted)
    date_frame = pd.DataFrame({"date": dates})
    return {
        "raw": metric(y, raw),
        "adjusted": metric(y, adjusted),
        "adjusted_minus_raw": {
            "brier": float(np.mean(b1 - b0)),
            "logloss": float(np.mean(l1 - l0)),
            "ci95_date_cluster": date_bootstrap(date_frame, b1 - b0, l1 - l0, draws, 1717),
        },
    }


def summarize_group(frame, replay, repeats, folds, draws):
    out = {
        "n_games": int(frame["gid"].nunique()),
        "n_dates": int(frame["date"].nunique()),
        "raw": {side: metric(g["y"], g["p"]) for side, g in frame.groupby("side")},
        "fits": {},
        "repeated_group_date_cv": {},
    }

    for mode in ("bottom_only", "half_offset", "half_curve"):
        result, names = fit(frame, mode)
        coeff = {
            name: {
                "coef": float(result.params[i]),
                "se_date_cluster": float(result.bse[i]),
                "z": float(result.tvalues[i]),
                "p": float(result.pvalues[i]),
            }
            for i, name in enumerate(names)
        }
        fit_summary = {
            "coefficients": coeff,
            "aic": float(result.aic),
            "deviance": float(result.deviance),
        }
        if mode == "half_offset":
            cov = np.asarray(result.cov_params())
            top = float(result.params[0])
            bottom = float(result.params[0] + result.params[1])
            bottom_se = math.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
            fit_summary["derived_offsets"] = {
                "top": top,
                "bottom": bottom,
                "bottom_ci95": [bottom - 1.96 * bottom_se, bottom + 1.96 * bottom_se],
            }
        if mode == "half_curve":
            fit_summary["derived_curves"] = {
                "top": {
                    "intercept": float(result.params[0]),
                    "slope": float(result.params[1]),
                },
                "bottom": {
                    "intercept": float(result.params[0] + result.params[2]),
                    "slope": float(result.params[1] + result.params[3]),
                },
            }
        out["fits"][mode] = fit_summary

        q = repeated_group_date_cv(frame, mode, repeats, folds)
        b0, l0 = losses(frame["y"], frame["p"])
        b1, l1 = losses(frame["y"], q)
        out["repeated_group_date_cv"][mode] = {
            "halves": metric(frame["y"], q),
            "vs_raw": {
                "brier": float(np.mean(b1 - b0)),
                "logloss": float(np.mean(l1 - l0)),
                "ci95_date_cluster": date_bootstrap(frame, b1 - b0, l1 - l0, draws, 1709),
            },
            "full_i2": full_under_metrics(frame, q, replay, draws),
        }

    bottom_fit, _ = fit(frame, "bottom_only")
    offset = float(bottom_fit.params[0])
    out["bottom_only_probability_map"] = [
        {
            "raw_score_probability": p,
            "adjusted_score_probability": float(expit(logit(p) + offset)),
        }
        for p in (0.15, 0.20, 0.25, 0.30, 0.35)
    ]
    return out


def main():
    args = parse_args()
    replay = json.loads(args.replay.read_text())
    inputs = json.loads(args.inputs.read_text())
    frame = build_rows(replay, inputs)
    matched = frame[frame["park_status"] == MATCHED].reset_index(drop=True)
    result = {
        "version": "i2-half-calibration-full-season-2025-v1",
        "status": "RESEARCH_ONLY_PREVIOUSLY_INSPECTED_2025",
        "market_inputs_used": False,
        "season": 2025,
        "canonical_replay": replay["model_version"],
        "trials_per_game": replay["trials_per_game"],
        "calendar_month_used": False,
        "method": (
            "Full-season half-inning score calibration. Binomial logit models use the "
            "canonical pregame half score probability as an offset/predictor; covariance "
            "is clustered by calendar date. Repeated CV randomizes whole dates across "
            "folds and never uses month as a feature or control."
        ),
        "full_slate": summarize_group(frame.reset_index(drop=True), replay, args.cv_repeats, args.cv_folds, args.bootstrap),
        "matched_home_venue": summarize_group(matched, replay, args.cv_repeats, args.cv_folds, args.bootstrap),
        "governance": {
            "live_model_changed": False,
            "production_calibration_changed": False,
            "market_workflow_changed": False,
            "2025_previously_inspected": True,
            "eligible_for_direct_promotion": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "matched_raw": result["matched_home_venue"]["raw"],
        "matched_fits": result["matched_home_venue"]["fits"],
        "matched_cv": result["matched_home_venue"]["repeated_group_date_cv"],
    }, indent=2))


if __name__ == "__main__":
    main()

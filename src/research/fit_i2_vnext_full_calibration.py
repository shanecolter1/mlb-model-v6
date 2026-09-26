#!/usr/bin/env python3
"""Fit the single final full-I2 calibration curve for I2 vNext.

The raw baseball model is frozen before this stage. 2025 is used only for
full-model calibration. A chronological 70/30 split tests whether a simple
two-parameter logit calibration improves both Brier score and log loss versus
no calibration. If it passes, the same specification is refit on all 2025
games. No market data is used.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/derived/i2_vnext/replay_2025_predictions.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/i2_vnext/full_i2_calibration.json"),
    )
    return p.parse_args()


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)


def logit(p: np.ndarray) -> np.ndarray:
    q = clip(p)
    return np.log(q / (1 - q))


def logistic(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1 / (1 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1 + ex)
    return out


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    q = clip(p)
    return {
        "n": int(len(y)),
        "brier": float(np.mean((q - y) ** 2)),
        "logloss": float(np.mean(-(y * np.log(q) + (1 - y) * np.log(1 - q)))),
        "mean_prediction": float(np.mean(q)),
        "realized_under_rate": float(np.mean(y)),
    }


def fit_logit_calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    x = logit(p)
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.array([0.0, 1.0], dtype=float)
    ridge = np.diag([1e-9, 1e-9])
    for _ in range(100):
        eta = X @ beta
        mu = logistic(eta)
        w = np.maximum(mu * (1 - mu), 1e-8)
        grad = X.T @ (y - mu) - ridge @ beta
        h = X.T @ (w[:, None] * X) + ridge
        step = np.linalg.solve(h, grad)
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-10:
            beta = beta_next
            break
        beta = beta_next
    return float(beta[0]), float(beta[1])


def apply_curve(p: np.ndarray, a: float, b: float) -> np.ndarray:
    return logistic(a + b * logit(p))


def bootstrap_delta(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, draws: int = 2000) -> dict:
    rng = np.random.default_rng(73)
    n = len(y)
    brier = np.empty(draws)
    ll = np.empty(draws)
    raw_c = clip(raw)
    cal_c = clip(cal)
    raw_ll = -(y * np.log(raw_c) + (1 - y) * np.log(1 - raw_c))
    cal_ll = -(y * np.log(cal_c) + (1 - y) * np.log(1 - cal_c))
    raw_bs = (raw_c - y) ** 2
    cal_bs = (cal_c - y) ** 2
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        brier[i] = float(np.mean(raw_bs[idx] - cal_bs[idx]))
        ll[i] = float(np.mean(raw_ll[idx] - cal_ll[idx]))
    return {
        "positive_means_calibration_better": True,
        "brier_improvement_mean": float(np.mean(brier)),
        "brier_improvement_ci95": [float(np.quantile(brier, 0.025)), float(np.quantile(brier, 0.975))],
        "logloss_improvement_mean": float(np.mean(ll)),
        "logloss_improvement_ci95": [float(np.quantile(ll, 0.025)), float(np.quantile(ll, 0.975))],
        "draws": draws,
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("market_inputs_used") is not False:
        raise RuntimeError("Calibration input is not explicitly market-isolated")
    rows = payload.get("predictions") or []
    if len(rows) < 1000:
        raise RuntimeError(f"Expected at least 1,000 2025 replay games; found {len(rows)}")

    rows = sorted(rows, key=lambda r: (r["date"], r["gid"]))
    dates = sorted({r["date"] for r in rows})
    split_idx = max(1, min(len(dates) - 1, int(math.floor(len(dates) * 0.70))))
    cutoff = dates[split_idx]

    train = [r for r in rows if r["date"] < cutoff]
    valid = [r for r in rows if r["date"] >= cutoff]
    if len(train) < 500 or len(valid) < 250:
        raise RuntimeError(f"Chronological calibration split too small: train={len(train)} valid={len(valid)}")

    def arrays(data):
        return (
            np.array([float(r["observed_under05"]) for r in data], dtype=float),
            np.array([float(r["raw_under05"]) for r in data], dtype=float),
        )

    y_t, p_t = arrays(train)
    y_v, p_v = arrays(valid)
    a_t, b_t = fit_logit_calibration(y_t, p_t)
    p_v_cal = apply_curve(p_v, a_t, b_t)

    raw_valid = metrics(y_v, p_v)
    cal_valid = metrics(y_v, p_v_cal)
    delta = bootstrap_delta(y_v, p_v, p_v_cal)

    passes = (
        cal_valid["brier"] < raw_valid["brier"]
        and cal_valid["logloss"] < raw_valid["logloss"]
    )

    y_all, p_all = arrays(rows)
    if passes:
        a_final, b_final = fit_logit_calibration(y_all, p_all)
        status = "LOGIT_CALIBRATION_SELECTED"
    else:
        a_final, b_final = 0.0, 1.0
        status = "NO_CALIBRATION_SELECTED"

    p_all_final = apply_curve(p_all, a_final, b_final)
    result = {
        "version": "i2-vnext-full-calibration-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_inputs_used": False,
        "target": "full I2 Under 0.5",
        "source_model_version": payload.get("model_version"),
        "source_replay_version": payload.get("version"),
        "selection_rule": (
            "Fit two-parameter logit calibration on first 70% of 2025 dates; "
            "select only if it improves both Brier and log loss on final 30%; "
            "if selected, refit same curve on all 2025. 2026 remains forward validation."
        ),
        "chronological_split": {
            "cutoff_date": cutoff,
            "train_n": len(train),
            "validation_n": len(valid),
        },
        "candidate_logit": {
            "fit_on_train": {"a": a_t, "b": b_t},
            "validation_raw": raw_valid,
            "validation_calibrated": cal_valid,
            "bootstrap_delta": delta,
            "passes_selection_gate": passes,
        },
        "selected": {
            "status": status,
            "formula": "logit(p_final) = a + b * logit(p_raw)",
            "a": a_final,
            "b": b_final,
        },
        "all_2025_diagnostics": {
            "raw": metrics(y_all, p_all),
            "selected_curve": metrics(y_all, p_all_final),
        },
        "forward_validation": {
            "season": 2026,
            "status": "RESERVED_UNTOUCHED_AFTER_FREEZE",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "selection": result["selected"],
        "cutoff": result["chronological_split"],
        "validation_raw": raw_valid,
        "validation_calibrated": cal_valid,
        "bootstrap_delta": delta,
    }, indent=2))


if __name__ == "__main__":
    main()

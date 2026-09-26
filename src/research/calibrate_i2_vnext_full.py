#!/usr/bin/env python3
"""Fit the single final full-I2 calibration curve for I2 vNext.

Input is the leakage-safe 2025 full-model replay produced by
run_i2_vnext_replay.mjs. The raw baseball model is already frozen.

Chronology:
- first chronological half of 2025: fit one sigmoid/logit candidate
- second chronological half of 2025: validate candidate vs identity
- adopt sigmoid only if BOTH Brier and log loss improve
- if adopted, refit that same one-dimensional curve on all 2025 replay games

No component-level calibration or additional probability shrinkage is allowed.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--replay",
        type=Path,
        default=Path("data/derived/i2_vnext/replay_2025_predictions.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/i2_vnext/i2_vnext_full_calibration.json"),
    )
    return p.parse_args()


def clip(p: float) -> float:
    return min(1 - 1e-9, max(1e-9, float(p)))


def logit_values(values) -> np.ndarray:
    p = np.asarray([clip(x) for x in values], dtype=float)
    return np.log(p / (1 - p)).reshape(-1, 1)


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray([clip(x) for x in p], dtype=float)
    return {
        "n": int(len(y)),
        "realized_under_rate": float(np.mean(y)),
        "predicted_under_mean": float(np.mean(p)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }


def sigmoid_prob(raw, intercept: float, slope: float) -> np.ndarray:
    z = intercept + slope * logit_values(raw).reshape(-1)
    return 1.0 / (1.0 + np.exp(-z))


def fit_sigmoid(raw, y):
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logit_values(raw), np.asarray(y, dtype=int))
    return model


def main() -> None:
    args = parse_args()
    replay = json.loads(args.replay.read_text())
    if replay.get("market_inputs_used") is not False:
        raise RuntimeError("Replay must be baseball-only")
    if replay.get("observed_i2_state_used_as_predictor") is not False:
        raise RuntimeError("Replay used observed I2 state as a predictor")

    rows = sorted(
        replay.get("predictions") or [],
        key=lambda r: (str(r.get("date") or ""), str(r.get("gid") or "")),
    )
    if len(rows) < 500:
        raise RuntimeError(f"Insufficient 2025 replay games: {len(rows)}")

    dates = sorted({str(r.get("date") or "") for r in rows})
    if len(dates) < 2:
        raise RuntimeError("2025 replay does not contain enough distinct dates")
    split_date = dates[len(dates) // 2]
    fit_rows = [r for r in rows if str(r.get("date") or "") < split_date]
    val_rows = [r for r in rows if str(r.get("date") or "") >= split_date]
    if len(fit_rows) < 250 or len(val_rows) < 250:
        raise RuntimeError(
            f"Chronological date split too small: fit={len(fit_rows)} validation={len(val_rows)}"
        )

    raw = np.asarray([clip(r["raw_under05"]) for r in rows], dtype=float)
    y = np.asarray([int(r["observed_under05"]) for r in rows], dtype=int)
    raw_fit = np.asarray([clip(r["raw_under05"]) for r in fit_rows], dtype=float)
    y_fit = np.asarray([int(r["observed_under05"]) for r in fit_rows], dtype=int)
    raw_val = np.asarray([clip(r["raw_under05"]) for r in val_rows], dtype=float)
    y_val = np.asarray([int(r["observed_under05"]) for r in val_rows], dtype=int)

    identity_val = metrics(y_val, raw_val)
    candidate = fit_sigmoid(raw_fit, y_fit)
    cand_intercept = float(candidate.intercept_[0])
    cand_slope = float(candidate.coef_[0, 0])
    candidate_val_p = sigmoid_prob(raw_val, cand_intercept, cand_slope)
    sigmoid_val = metrics(y_val, candidate_val_p)

    adopt = (
        sigmoid_val["brier"] < identity_val["brier"]
        and sigmoid_val["logloss"] < identity_val["logloss"]
    )

    if adopt:
        final_model = fit_sigmoid(raw, y)
        final_curve = {
            "type": "sigmoid",
            "intercept": float(final_model.intercept_[0]),
            "slope": float(final_model.coef_[0, 0]),
        }
    else:
        final_curve = {"type": "none", "intercept": 0.0, "slope": 1.0}

    payload = {
        "version": "i2-vnext-full-calibration-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_inputs_used": False,
        "target": "full I2 Under 0.5 probability",
        "source_replay_version": replay.get("version"),
        "source_model_version": replay.get("model_version"),
        "source_model_training": replay.get("model_training"),
        "holdout_policy": replay.get("holdout_policy"),
        "calibration_selection": {
            "fit_segment": f"2025 dates before {split_date}",
            "validation_segment": f"2025 dates on/after {split_date}",
            "split_date": split_date,
            "fit_games": int(len(fit_rows)),
            "validation_games": int(len(val_rows)),
            "candidate_models": ["identity", "single_sigmoid_logit"],
            "selection_rule": (
                "sigmoid must improve both Brier and log loss on later-2025 validation; "
                "otherwise identity"
            ),
            "identity_validation": identity_val,
            "sigmoid_validation": sigmoid_val,
            "sigmoid_candidate": {
                "intercept": cand_intercept,
                "slope": cand_slope,
            },
            "chosen": "sigmoid" if adopt else "identity",
        },
        "raw_all_2025": metrics(y, raw),
        "replay_games": int(len(rows)),
        "final_curve": final_curve,
        "governance": {
            "probability_calibration_layers": 1 if adopt else 0,
            "component_level_probability_calibration": False,
            "market_conditioning": False,
            "observed_i2_state_used_as_predictor": False,
            "prospective_validation_required": True,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "replay_games": payload["replay_games"],
        "raw_all_2025": payload["raw_all_2025"],
        "identity_validation": identity_val,
        "sigmoid_validation": sigmoid_val,
        "final_curve": final_curve,
    }, indent=2))


if __name__ == "__main__":
    main()

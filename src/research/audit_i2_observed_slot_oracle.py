#!/usr/bin/env python3
"""Target-only I2 start-slot oracle diagnostic; never a pregame predictor."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--replay-input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def metrics(y, p):
    return {
        "n": len(y), "observed_under": float(y.mean()),
        "predicted_under": float(p.mean()),
        "bias_pred_minus_actual": float(np.mean(p-y)),
        "brier": float(np.mean((p-y)**2)),
        "logloss": float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
    }


def main():
    a = arguments()
    if a.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    payload = json.loads(a.input.read_text())
    replay = json.loads(a.replay_input.read_text())
    if payload.get("season") != replay.get("season") or payload.get("season") != 2025:
        raise ValueError("Unexpected replay season")
    if payload.get("market_inputs_used") is not False or replay.get("market_inputs_used") is not False:
        raise ValueError("Market entered pregame replay")
    if payload.get("observed_i2_start_slot_used_as_predictor") is not False or payload.get("observed_slot_target_only_diagnostic") is not True:
        raise ValueError("Observed slot not governed as target-only")
    if not 0.9 < payload.get("park_match_rate", 0) < 1:
        raise ValueError("Park coverage failed")
    rows = payload["predictions"]
    games = {g["gid"]: g for g in replay["games"]}
    if len(rows) != len(games) or {r["gid"] for r in rows} != set(games):
        raise ValueError("Replay games do not match")
    y = np.asarray([r["observed_under05"] for r in rows], float)
    before = np.asarray([r["player_asof_under05"] for r in rows], float)
    oracle = np.asarray([r["oracle_slot_under05"] for r in rows], float)
    for v in (before, oracle):
        if not np.isfinite(v).all() or not ((0 < v) & (v < 1)).all():
            raise ValueError("Invalid probability")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid outcome")
    changes = np.column_stack((
        oracle-before,
        (oracle-y)**2-(before-y)**2,
        (-y*np.log(oracle)-(1-y)*np.log1p(-oracle)) - (-y*np.log(before)-(1-y)*np.log1p(-before)),
    ))
    by_date = defaultdict(list)
    for i, row in enumerate(rows):
        by_date[row["date"]].append(i)
    clusters = np.asarray([[*changes[idx].sum(axis=0), len(idx)] for idx in by_date.values()])
    rng = np.random.default_rng(1405)
    draws = clusters[rng.integers(0, len(clusters), size=(a.bootstrap, len(clusters)))].sum(axis=1)
    ci = {name: [float(v) for v in np.quantile(draws[:, j]/draws[:, 3], [0.025, 0.975])]
          for j, name in enumerate(("predicted_under", "brier", "logloss"))}

    halves = {}
    for side, observed_key in (("top", "top2_runs"), ("bottom", "bottom2_runs")):
        half_y = np.asarray([int(games[r["gid"]]["observed"][observed_key] == 0) for r in rows])
        halves[side] = {
            "baseline": metrics(half_y, np.asarray([r[f"player_{side}0"] for r in rows])),
            "oracle_slot": metrics(half_y, np.asarray([r[f"oracle_{side}0"] for r in rows])),
        }
    out = {
        "version": "i2-observed-slot-oracle-diagnostic-v1",
        "status": "RESEARCH_ONLY_TARGET_LEAKAGE_DIAGNOSTIC",
        "market_inputs_used": False,
        "observed_slot_used_for_pregame_prediction": False,
        "games": len(rows),
        "baseline": metrics(y, before),
        "oracle_slot": metrics(y, oracle),
        "oracle_minus_baseline": {
            "predicted_under": float(changes[:, 0].mean()),
            "brier": float(changes[:, 1].mean()),
            "logloss": float(changes[:, 2].mean()),
            "paired_calendar_date_ci95": ci,
        },
        "halves": halves,
        "park_matched": payload["park_match_rate"],
        "governance": {
            "diagnostic_only": True,
            "oracle_slot_is_observed_after_I1": True,
            "game_outcomes_only_for_scoring": True,
            "same_static_2024_trained_I2_model_and_transitions": True,
            "2025_walkforward_refits_used": False,
            "calibration_applied": False,
            "no_independent_promotion_test": True,
            "live_runner_changed": False,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("games", "baseline", "oracle_slot", "oracle_minus_baseline", "halves")}, indent=2))


if __name__ == "__main__":
    main()

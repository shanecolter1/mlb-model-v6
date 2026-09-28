#!/usr/bin/env python3
"""Score paired full-I2 Under forecasts from the frozen direct I1 research arm."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def main():
    a = arguments()
    if a.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    payload = json.loads(a.input.read_text())
    if payload.get("season") != 2025 or payload.get("market_inputs_used") is not False:
        raise ValueError("Unexpected replay season or market isolation")
    if payload.get("direct_i1_pa_model") != "i1-direct-pa-home-research-2022-2023-v1":
        raise ValueError("Unexpected I1 PA research model")
    if not 0.9 < payload.get("park_match_rate", 0) < 1:
        raise ValueError("Expected prior-season park coverage is missing")
    rows = payload["predictions"]
    if len(rows) != payload["games"] or len({r["gid"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate games")
    y = np.asarray([r["observed_under05"] for r in rows], dtype=float)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid Under target")

    def score(key):
        p = np.asarray([r[key] for r in rows], dtype=float)
        if not np.isfinite(p).all() or not ((0 < p) & (p < 1)).all():
            raise ValueError(f"Invalid probabilities: {key}")
        return p, (p-y)**2, -(y*np.log(p)+(1-y)*np.log1p(-p))

    old, old_brier, old_loss = score("player_asof_under05")
    new, new_brier, new_loss = score("direct_under05")
    dates = defaultdict(list)
    for i, row in enumerate(rows):
        dates[row["date"]].append(i)
    clustered = np.asarray([[float(np.sum((new_brier-old_brier)[idx])),
                             float(np.sum((new_loss-old_loss)[idx])),
                             float(np.sum((new-old)[idx])), len(idx)]
                            for idx in dates.values()])
    rng = np.random.default_rng(1305)
    samples = clustered[rng.integers(0, len(clustered), size=(a.bootstrap, len(clustered)))].sum(axis=1)
    intervals = {
        key: [float(v) for v in np.quantile(samples[:, j]/samples[:, 3], [0.025, 0.975])]
        for j, key in enumerate(("brier", "logloss", "predicted_under"))
    }
    out = {
        "version": "i2-direct-i1-exact-paired-2025-v1",
        "status": "RESEARCH_ONLY_NOT_PROMOTED",
        "games": len(rows),
        "market_inputs_used": False,
        "observed_i2_outcomes_used_as_predictors": False,
        "observed_under_rate": float(y.mean()),
        "baseline": {"predicted_under": float(old.mean()), "brier": float(old_brier.mean()), "logloss": float(old_loss.mean())},
        "direct_i1": {"predicted_under": float(new.mean()), "brier": float(new_brier.mean()), "logloss": float(new_loss.mean())},
        "direct_minus_baseline": {
            "predicted_under": float((new-old).mean()),
            "brier": float((new_brier-old_brier).mean()),
            "logloss": float((new_loss-old_loss).mean()),
            "paired_calendar_date_ci95": intervals,
        },
        "by_half": {
            side: {
                "baseline_scoreless_mean": float(np.mean([r[f"player_{key}0"] for r in rows])),
                "direct_scoreless_mean": float(np.mean([r[f"direct_{key}0"] for r in rows])),
            }
            for side, key in (("top", "top"), ("bottom", "bottom"))
        },
        "mean_i1_slot_total_variation": float(np.mean([
            (r["away_direct_slot_tv"]+r["home_direct_slot_tv"])/2 for r in rows
        ])),
        "governance": {
            "i1_fit_years": [2022, 2023],
            "downstream_i2_model": payload["downstream_i2_model"],
            "downstream_i2_training": payload["downstream_i2_training"],
            "downstream_i2_2025_walkforward_refits_used": False,
            "i1_environment": payload["i1_environment"],
            "park_rule_i2": payload["park_rule_i2"],
            "2025_outcomes_previously_inspected": True,
            "same_exact_i2_model_and_transitions_in_both_arms": True,
            "full_i2_calibration_applied": False,
            "live_runner_changed": False,
            "bootstrap_scope": "paired dates conditional on frozen model; no model-fit uncertainty",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("games", "observed_under_rate", "baseline", "direct_i1", "direct_minus_baseline", "by_half")}, indent=2))


if __name__ == "__main__":
    main()

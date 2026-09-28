#!/usr/bin/env python3
"""Paired exact I1 start-slot comparison for a frozen research PA model."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def args():
    p = argparse.ArgumentParser()
    for name in ("baseline-2024", "candidate-2024", "baseline-2025", "candidate-2025", "model", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def load(path, year, expected_model):
    payload = json.loads(path.read_text())
    if payload.get("season") != year or payload.get("market_inputs_used") is not False:
        raise ValueError(f"Wrong season or market isolation: {path}")
    if payload.get("observed_i2_start_slot_used_as_predictor") is not False:
        raise ValueError(f"Observed I2 slot leaked into prediction: {path}")
    if payload.get("i1_environment", "neutral") != "neutral":
        raise ValueError(f"I1 environment mismatch: {path}")
    if payload.get("i1_pa_model", "existing_50_50_formula") != expected_model:
        raise ValueError(f"PA model mismatch: {path}")
    rows = {(r["gid"], r["side"]): r for r in payload["rows"]}
    if len(rows) != len(payload["rows"]):
        raise ValueError(f"Duplicate half: {path}")
    return rows


def ci(rows, value, draws, seed):
    dates = defaultdict(lambda: [0.0, 0])
    for row in rows:
        dates[row["date"]][0] += value(row)
        dates[row["date"]][1] += 1
    values = np.asarray(list(dates.values()))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sums = values[indices].sum(axis=1)
    return [float(x) for x in np.quantile(sums[:, 0] / sums[:, 1], [0.025, 0.975])]


def summarize(base, candidate, draws, seed):
    if base.keys() != candidate.keys():
        raise ValueError("Unmatched I1 halves")
    paired = []
    for key, before in base.items():
        after = candidate[key]
        if before["date"] != after["date"] or before["observed_slot"] != after["observed_slot"]:
            raise ValueError(f"Target changed for {key}")
        def arm(row):
            score = row["player_asof"]
            return {
                "late": sum(score["distribution"][str(s)] for s in (7, 8, 9)),
                "logloss": score["logloss"], "brier": score["brier"],
            }
        paired.append({
            "gid": key[0], "side": key[1], "date": before["date"],
            "observed_late": int(before["observed_slot"] in (7, 8, 9)),
            "baseline": arm(before), "candidate": arm(after),
        })
    result = {}
    for label in ("all", "top", "bottom"):
        group = paired if label == "all" else [r for r in paired if r["side"] == label]
        n = len(group)
        result[label] = {
            "n_halves": n,
            "observed_late": sum(r["observed_late"] for r in group) / n,
            "arms": {
                arm: {
                    "predicted_late": sum(r[arm]["late"] for r in group) / n,
                    "late_pred_minus_actual": sum(r[arm]["late"] - r["observed_late"] for r in group) / n,
                    "slot_logloss": sum(r[arm]["logloss"] for r in group) / n,
                    "slot_brier": sum(r[arm]["brier"] for r in group) / n,
                }
                for arm in ("baseline", "candidate")
            },
            "candidate_minus_baseline": {
                "slot_logloss": sum(r["candidate"]["logloss"] - r["baseline"]["logloss"] for r in group) / n,
                "slot_brier": sum(r["candidate"]["brier"] - r["baseline"]["brier"] for r in group) / n,
                "slot_logloss_ci95": ci(
                    group, lambda r: r["candidate"]["logloss"] - r["baseline"]["logloss"], draws, seed
                ),
            },
        }
    return result


def main():
    a = args()
    if a.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    model = json.loads(a.model.read_text())
    if model.get("market_inputs_used") is not False or model.get("observed_2024_outcomes_used_for_fit") is not False:
        raise ValueError("Candidate model not training-isolated")
    if model.get("training_years") != [2022, 2023] or model.get("status") != "RESEARCH_ONLY_NOT_PROMOTED":
        raise ValueError("Unexpected candidate training or status")
    result = {
        "version": "i1-direct-pa-exact-slot-comparison-v1",
        "status": "SHADOW_ONLY_NOT_PROMOTED",
        "market_inputs_used": False,
        "observed_i2_slots_used_as_predictors": False,
        "candidate_model_version": model["version"],
        "2024_inspected": summarize(
            load(a.baseline_2024, 2024, "existing_50_50_formula"),
            load(a.candidate_2024, 2024, model["version"]), a.bootstrap, 2024
        ),
        "2025_inspected": summarize(
            load(a.baseline_2025, 2025, "existing_50_50_formula"),
            load(a.candidate_2025, 2025, model["version"]), a.bootstrap, 2025
        ),
        "governance": {
            "candidate_fit_years": [2022, 2023],
            "2024_and_2025_outcomes_previously_inspected": True,
            "full_i2_probability_scored": False,
            "live_runner_changed": False,
            "interval_method": f"{a.bootstrap} paired calendar-date cluster draws conditional on fitted parameters",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        year: {side: result[year][side]["candidate_minus_baseline"] for side in ("all", "top", "bottom")}
        for year in ("2024_inspected", "2025_inspected")
    }, indent=2))


if __name__ == "__main__":
    main()

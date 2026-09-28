#!/usr/bin/env python3
"""Localize bottom-I2 scoreless error in the frozen canonical 2025 replay."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def summarize(rows, draws, seed):
    if not rows:
        raise ValueError("Empty residual group")
    bottom = np.asarray([r["bottom_p"] - r["bottom_y"] for r in rows])
    top = np.asarray([r["top_p"] - r["top_y"] for r in rows])
    dates = defaultdict(list)
    for i, row in enumerate(rows):
        dates[row["date"]].append(i)
    clustered = np.asarray([[bottom[ix].sum(), (bottom-top)[ix].sum(), len(ix)] for ix in dates.values()])
    rng = np.random.default_rng(seed)
    sample = clustered[rng.integers(0, len(clustered), size=(draws, len(clustered)))].sum(axis=1)
    return {
        "n_games": len(rows), "n_dates": len(dates),
        "bottom_predicted_scoreless": float(np.mean([r["bottom_p"] for r in rows])),
        "bottom_observed_scoreless": float(np.mean([r["bottom_y"] for r in rows])),
        "bottom_bias": float(bottom.mean()),
        "bottom_bias_date_ci95": [float(v) for v in np.quantile(sample[:, 0]/sample[:, 2], [.025, .975])],
        "top_bias": float(top.mean()),
        "bottom_minus_top_bias": float((bottom-top).mean()),
        "bottom_minus_top_date_ci95": [float(v) for v in np.quantile(sample[:, 1]/sample[:, 2], [.025, .975])],
    }


def main():
    a = args()
    if a.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    replay = json.loads(a.replay.read_text())
    inputs = json.loads(a.inputs.read_text())
    if replay.get("market_inputs_used") is not False or inputs.get("market_inputs_used") is not False:
        raise ValueError("Market contamination")
    if replay.get("observed_i2_state_used_as_predictor") is not False or replay.get("i1_state_mode") != "player_asof":
        raise ValueError("Replay governance mismatch")
    if replay.get("trials_per_game") != 10000 or replay.get("model_version") != "i2-vnext-walkforward-2025-v1":
        raise ValueError("Expected canonical precision replay")
    games = {g["gid"]: g for g in inputs["games"]}
    rows = []
    for p in replay["predictions"]:
        g = games[p["gid"]]
        if p["date"] != g["date"] or p["observed_under05"] != g["observed"]["under05"]:
            raise ValueError(f"Target mismatch: {p['gid']}")
        rows.append({
            "date": p["date"], "park": p["park_status"],
            "top_p": 1-p["top2_score_probability"],
            "bottom_p": 1-p["bottom2_score_probability"],
            "top_y": int(g["observed"]["top2_runs"] == 0),
            "bottom_y": int(g["observed"]["bottom2_runs"] == 0),
            "away_starter_began_i2": g["audit"]["away_starter_began_i2"] == 1,
        })
    if len(rows) != len(games) or len(rows) != 2430:
        raise ValueError("Game coverage mismatch")
    matched = [r for r in rows if r["park"] == "RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT"]
    neutral = [r for r in rows if r["park"] == "EXPLICIT_2025_SITE_NEUTRAL"]
    if len(matched) != 2264 or len(neutral) != 166:
        raise ValueError("Park coverage mismatch")
    cuts = np.quantile([r["bottom_p"] for r in matched], [0, .2, .4, .6, .8, 1])
    buckets = {}
    for i in range(5):
        group = [r for r in matched if (cuts[i] <= r["bottom_p"] < cuts[i+1] if i < 4
                                        else cuts[i] <= r["bottom_p"] <= cuts[i+1])]
        buckets[str(i+1)] = {"probability_range": [float(cuts[i]), float(cuts[i+1])],
                           **summarize(group, a.bootstrap, 1500+i)}
    periods = {
        "mar_apr": [r for r in matched if r["date"] < "20250501"],
        "may_jun": [r for r in matched if "20250501" <= r["date"] < "20250701"],
        "jul_sep": [r for r in matched if r["date"] >= "20250701"],
    }
    result = {
        "version": "i2-bottom-half-residual-localization-v1",
        "status": "SHADOW_DIAGNOSTIC_ONLY",
        "market_inputs_used": False,
        "canonical_replay": replay["model_version"],
        "full_slate": summarize(rows, a.bootstrap, 150),
        "matched_park": summarize(matched, a.bootstrap, 151),
        "neutral_venue": summarize(neutral, a.bootstrap, 152),
        "matched_park_starter_continuation_observed_only": {
            "same_starter": summarize([r for r in matched if r["away_starter_began_i2"]], a.bootstrap, 153),
            "changed": summarize([r for r in matched if not r["away_starter_began_i2"]], a.bootstrap, 154),
        },
        "matched_park_predicted_bottom_scoreless_quintiles": buckets,
        "matched_park_calendar_segments": {name: summarize(group, a.bootstrap, 160+i)
                                            for i, (name, group) in enumerate(periods.items())},
        "governance": {
            "2025_previously_inspected": True,
            "actual_starter_continuation_used_as_predictor": False,
            "quintiles_defined_by_pregame_probability_only": True,
            "calendar_segments_descriptive_not_fit": True,
            "bias_intervals_conditional_on_frozen_predictions": True,
            "live_model_changed": False,
            "bottom_cause_identified": False,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("matched_park", "matched_park_predicted_bottom_scoreless_quintiles", "matched_park_calendar_segments")}, indent=2))


if __name__ == "__main__":
    main()

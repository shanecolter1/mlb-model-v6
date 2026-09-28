#!/usr/bin/env python3
"""Audit frozen 2025 I1 slot forecasts and I2 venue residuals, without refitting."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slot-rows", type=Path, required=True)
    parser.add_argument("--precision-replay", type=Path, required=True)
    parser.add_argument("--replay-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def load(path):
    return json.loads(path.read_text())


def unique_by(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def interval(rows, numerator, *, draws, seed):
    """Calendar-date cluster percentile interval for a mean prediction error."""
    dates = defaultdict(lambda: [0.0, 0])
    for row in rows:
        dates[row["date"]][0] += numerator(row)
        dates[row["date"]][1] += 1
    counts = np.asarray(list(dates.values()), dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(draws, len(counts)))
    sampled = counts[indices].sum(axis=1)
    return [float(v) for v in np.quantile(sampled[:, 0] / sampled[:, 1], [0.025, 0.975])]


def slot_summary(rows, draws, seed):
    n = len(rows)
    slots = {}
    for slot in range(1, 10):
        predicted = sum(r["player_asof"]["distribution"][str(slot)] for r in rows) / n
        actual = sum(r["observed_slot"] == slot for r in rows) / n
        slots[str(slot)] = {
            "predicted": predicted,
            "actual": actual,
            "pred_minus_actual": predicted - actual,
        }

    def late_error(row):
        predicted = sum(row["player_asof"]["distribution"][str(s)] for s in (7, 8, 9))
        return predicted - int(row["observed_slot"] in (7, 8, 9))

    late = sum(late_error(r) for r in rows) / n
    return {
        "n_halves": n,
        "player_asof_slot_logloss": sum(r["player_asof"]["logloss"] for r in rows) / n,
        "league_slot_logloss": sum(r["league"]["logloss"] for r in rows) / n,
        "player_asof_slot_brier": sum(r["player_asof"]["brier"] for r in rows) / n,
        "league_slot_brier": sum(r["league"]["brier"] for r in rows) / n,
        "slots": slots,
        "slot_7_to_9_pred_minus_actual": late,
        "slot_7_to_9_error_ci95": interval(rows, late_error, draws=draws, seed=seed),
    }


def venue_summary(rows, draws, seed):
    n = len(rows)
    expected = sum(r["raw_under05"] for r in rows)
    actual = sum(r["observed_under05"] for r in rows)
    result = {
        "n_games": n,
        "predicted_under": expected / n,
        "actual_under": actual / n,
        "under_pred_minus_actual": (expected - actual) / n,
        "excess_expected_unders": expected - actual,
        "under_error_ci95": interval(
            rows, lambda r: r["raw_under05"] - r["observed_under05"], draws=draws, seed=seed
        ) if n >= 20 else None,
    }
    for side in ("top", "bottom"):
        predicted_scoreless = sum(1 - r[f"{side}2_score_probability"] for r in rows) / n
        actual_scoreless = sum(r[f"observed_{side}_scoreless"] for r in rows) / n
        result[f"{side}_scoreless"] = {
            "predicted": predicted_scoreless,
            "actual": actual_scoreless,
            "pred_minus_actual": predicted_scoreless - actual_scoreless,
        }
    return result


def main():
    args = arguments()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    slot_file = load(args.slot_rows)
    replay = load(args.precision_replay)
    inputs = load(args.replay_input)
    for name, payload in (("slot rows", slot_file), ("precision replay", replay), ("replay input", inputs)):
        if payload.get("market_inputs_used") is not False:
            raise ValueError(f"{name} is not market-isolated")
    if slot_file.get("observed_i2_start_slot_used_as_predictor") is not False:
        raise ValueError("Observed I2 slot used as predictor")
    if replay.get("observed_i2_state_used_as_predictor") is not False:
        raise ValueError("Observed I2 state used as predictor in precision replay")
    if replay.get("i1_state_mode") != "player_asof" or replay.get("trials_per_game") != 10000:
        raise ValueError("Expected frozen 10,000-trial player-asof I1 replay")
    if inputs.get("i1_state_model", {}).get("player_specific_i1_talent_used") is not True:
        raise ValueError("Expected governed pregame player-asof I1 inputs")
    games = unique_by(inputs["games"], "gid")
    predictions = unique_by(replay["predictions"], "gid")
    slots = unique_by(
        ({**row, "gid_side": f"{row['gid']}|{row['side']}"} for row in slot_file["rows"]),
        "gid_side",
    )
    if len(games) != 2430 or games.keys() != predictions.keys() or len(slots) != 2 * len(games):
        raise ValueError("2025 matched-game or paired-half coverage mismatch")

    rows = []
    for gid, game in games.items():
        pred = predictions[gid]
        if pred["date"] != game["date"] or pred["observed_under05"] != game["observed"]["under05"]:
            raise ValueError(f"Replay target or date mismatch: {gid}")
        for side in ("top", "bottom"):
            slot = slots[f"{gid}|{side}"]
            if slot["date"] != game["date"]:
                raise ValueError(f"Slot date mismatch: {gid}|{side}")
        rows.append({
            **pred,
            "site": game["site"],
            "observed_top_scoreless": int(game["observed"]["top2_runs"] == 0),
            "observed_bottom_scoreless": int(game["observed"]["bottom2_runs"] == 0),
        })
    all_slots = list(slots.values())
    by_park = defaultdict(list)
    by_site = defaultdict(list)
    for row in rows:
        by_park[row["park_status"]].append(row)
        by_site[row["site"]].append(row)

    neutral = by_park["EXPLICIT_2025_SITE_NEUTRAL"]
    matched = by_park["RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT"]
    if len(neutral) + len(matched) != len(rows):
        raise ValueError("Unexpected park status")
    total_excess = sum(r["raw_under05"] - r["observed_under05"] for r in rows)
    result = {
        "version": "i2-vnext-i1-venue-diagnosis-v1",
        "season": 2025,
        "status": "SHADOW_ONLY_DESCRIPTIVE",
        "market_inputs_used": False,
        "observed_i2_state_used_as_predictor": False,
        "method": "Exact pregame I1 start-slot distribution versus later observed slots; frozen precision I2 replay versus later outcomes",
        "source_transition_model": replay["transition_model"],
        "slot_evaluation": {
            "all": slot_summary(all_slots, args.bootstrap, 803),
            "top": slot_summary([r for r in all_slots if r["side"] == "top"], args.bootstrap, 804),
            "bottom": slot_summary([r for r in all_slots if r["side"] == "bottom"], args.bootstrap, 805),
        },
        "venue_evaluation": {
            "all": venue_summary(rows, args.bootstrap, 806),
            "matched_prior_venue": venue_summary(matched, args.bootstrap, 807),
            "neutral_actual_venue": venue_summary(neutral, args.bootstrap, 808),
            "neutral_sites": {
                site: venue_summary(group, args.bootstrap, 809 + i)
                for i, (site, group) in enumerate(sorted(by_site.items()))
                if group[0]["park_status"] == "EXPLICIT_2025_SITE_NEUTRAL"
            },
            "athletics_as_visitor": venue_summary(
                [r for r in rows if r["away_team_retro"] == "ATH"], args.bootstrap, 821
            ),
            "later_2025": venue_summary([r for r in rows if r["date"] >= "20250625"], args.bootstrap, 820),
            "neutral_share_of_total_excess_expected_unders": sum(
                r["raw_under05"] - r["observed_under05"] for r in neutral
            ) / total_excess,
        },
        "uncertainty": f"{args.bootstrap} calendar-date cluster bootstrap draws; descriptive after the 2025 outcomes were inspected, no multiple-comparison adjustment",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "games": len(rows),
        "halves": len(slots),
        "all_under_error": result["venue_evaluation"]["all"]["under_pred_minus_actual"],
        "neutral_under_error": result["venue_evaluation"]["neutral_actual_venue"]["under_pred_minus_actual"],
        "matched_under_error": result["venue_evaluation"]["matched_prior_venue"]["under_pred_minus_actual"],
        "slot_7_to_9_error": result["slot_evaluation"]["all"]["slot_7_to_9_pred_minus_actual"],
    }, indent=2))


if __name__ == "__main__":
    main()

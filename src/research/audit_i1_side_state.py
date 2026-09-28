#!/usr/bin/env python3
"""Compare pregame I1 start-slot forecasts by half and fixed mechanism ablations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def args():
    p = argparse.ArgumentParser()
    for name in ("rows-2024", "rows-2025", "rows-2025-distinct", "rows-2025-park"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--compact-dir", type=Path, default=Path("data/derived/i2"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def load_rows(path, year, environment="neutral"):
    payload = json.loads(path.read_text())
    if payload.get("season") != year or payload.get("market_inputs_used") is not False:
        raise ValueError(f"Invalid season or market isolation: {path}")
    if payload.get("observed_i2_start_slot_used_as_predictor") is not False:
        raise ValueError(f"Observed slot used as predictor: {path}")
    if payload.get("i1_environment", "neutral") != environment:
        raise ValueError(f"Unexpected I1 environment in {path}")
    rows = {(r["gid"], r["side"]): r for r in payload["rows"]}
    if len(rows) != len(payload["rows"]):
        raise ValueError(f"Duplicate game/side in {path}")
    return rows


def late_probability(row):
    return sum(row["player_asof"]["distribution"][str(slot)] for slot in (7, 8, 9))


def ci_by_date(items, value, draws=2000, seed=1):
    by_date = defaultdict(lambda: [0.0, 0])
    for item in items:
        by_date[item["date"]][0] += value(item)
        by_date[item["date"]][1] += 1
    matrix = np.asarray(list(by_date.values()), dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matrix), size=(draws, len(matrix)))
    totals = matrix[indices].sum(axis=1)
    return [float(x) for x in np.quantile(totals[:, 0] / totals[:, 1], (0.025, 0.975))]


def observed_history(compact_dir, year):
    with (compact_dir / f"i2_state_compact_{year}.csv").open(newline="") as f:
        raw = list(csv.DictReader(f))
    halves = {(r["gid"], r["half"]): r for r in raw}
    if len(halves) != len(raw):
        raise ValueError(f"Duplicate compact half in {year}")
    games = []
    for gid in sorted({r["gid"] for r in raw}):
        top, bottom = halves.get((gid, "top")), halves.get((gid, "bottom"))
        if not top or not bottom or top["date"] != bottom["date"]:
            continue
        games.append({
            "gid": gid, "date": top["date"],
            "top_late": int(int(top["i2_start_slot"]) in (7, 8, 9)),
            "bottom_late": int(int(bottom["i2_start_slot"]) in (7, 8, 9)),
            "top_pa": int(top["i1_pa"]), "bottom_pa": int(bottom["i1_pa"]),
        })
    return games


def historical_summary(games, draws, seed):
    n = len(games)
    return {
        "n_games": n,
        "top_late_rate": sum(g["top_late"] for g in games) / n,
        "bottom_late_rate": sum(g["bottom_late"] for g in games) / n,
        "bottom_minus_top_late": sum(g["bottom_late"] - g["top_late"] for g in games) / n,
        "bottom_minus_top_late_ci95": ci_by_date(
            games, lambda g: g["bottom_late"] - g["top_late"], draws, seed
        ),
        "top_mean_i1_pa": sum(g["top_pa"] for g in games) / n,
        "bottom_mean_i1_pa": sum(g["bottom_pa"] for g in games) / n,
    }


def score_rows(rows, draws, seed):
    by_side = {}
    for side in ("top", "bottom"):
        group = [r for (gid, s), r in rows.items() if s == side]
        n = len(group)
        error = lambda r: late_probability(r) - int(r["observed_slot"] in (7, 8, 9))
        by_side[side] = {
            "n_halves": n,
            "predicted_late": sum(late_probability(r) for r in group) / n,
            "observed_late": sum(r["observed_slot"] in (7, 8, 9) for r in group) / n,
            "late_pred_minus_actual": sum(error(r) for r in group) / n,
            "late_error_ci95": ci_by_date(group, error, draws, seed + (side == "bottom")),
            "slot_logloss": sum(r["player_asof"]["logloss"] for r in group) / n,
            "slot_brier": sum(r["player_asof"]["brier"] for r in group) / n,
        }
    pairs = []
    for (gid, side), top in rows.items():
        if side != "top" or (gid, "bottom") not in rows:
            continue
        bottom = rows[(gid, "bottom")]
        if top["date"] != bottom["date"]:
            continue
        pairs.append({
            "date": top["date"],
            "predicted_gap": late_probability(bottom) - late_probability(top),
            "observed_gap": int(bottom["observed_slot"] in (7, 8, 9)) - int(top["observed_slot"] in (7, 8, 9)),
        })
    n = len(pairs)
    by_side["paired_bottom_minus_top"] = {
        "n_games": n,
        "predicted_late_gap": sum(r["predicted_gap"] for r in pairs) / n,
        "observed_late_gap": sum(r["observed_gap"] for r in pairs) / n,
        "gap_error": sum(r["predicted_gap"] - r["observed_gap"] for r in pairs) / n,
        "gap_error_ci95": ci_by_date(
            pairs, lambda r: r["predicted_gap"] - r["observed_gap"], draws, seed + 2
        ),
    }
    return by_side


def compare_arm(baseline, candidate, draws, seed):
    if baseline.keys() != candidate.keys():
        raise ValueError("Mechanism ablation has unmatched half-inning coverage")
    changes = []
    for key, before in baseline.items():
        after = candidate[key]
        if before["observed_slot"] != after["observed_slot"] or before["date"] != after["date"]:
            raise ValueError(f"Mechanism ablation target mismatch: {key}")
        changes.append({
            "date": before["date"], "side": before["side"],
            "late": late_probability(after) - late_probability(before),
            "logloss": after["player_asof"]["logloss"] - before["player_asof"]["logloss"],
            "brier": after["player_asof"]["brier"] - before["player_asof"]["brier"],
        })
    result = {}
    for side in ("all", "top", "bottom"):
        group = changes if side == "all" else [r for r in changes if r["side"] == side]
        result[side] = {
            "n_halves": len(group),
            "delta_predicted_late": sum(r["late"] for r in group) / len(group),
            "delta_slot_logloss": sum(r["logloss"] for r in group) / len(group),
            "delta_slot_brier": sum(r["brier"] for r in group) / len(group),
            "delta_logloss_ci95": ci_by_date(group, lambda r: r["logloss"], draws, seed),
        }
    return result


def main():
    a = args()
    if a.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    r24 = load_rows(a.rows_2024, 2024)
    r25 = load_rows(a.rows_2025, 2025)
    distinct = load_rows(a.rows_2025_distinct, 2025)
    park = load_rows(a.rows_2025_park, 2025, "prior_season_park")
    history = {str(y): observed_history(a.compact_dir, y) for y in range(2021, 2026)}
    result = {
        "version": "i2-vnext-i1-side-state-audit-v1",
        "status": "SHADOW_ONLY_COMPONENT_RESEARCH",
        "market_inputs_used": False,
        "observed_slots_used_as_predictors": False,
        "historical_observed": {
            str(y): historical_summary(history[str(y)], a.bootstrap, 700 + y)
            for y in range(2021, 2026)
        },
        "historical_2021_to_2023": historical_summary(
            [r for y in range(2021, 2024) for r in history[str(y)]], a.bootstrap, 711
        ),
        "exact_pregame_i1_state": {
            "2024_selection_sample": score_rows(r24, a.bootstrap, 721),
            "2025_inspected_sample": score_rows(r25, a.bootstrap, 731),
        },
        "2025_fixed_mechanism_ablations": {
            "distinct_strikeout_bip_transitions_minus_pooled": compare_arm(r25, distinct, a.bootstrap, 741),
            "prior_season_park_in_i1_minus_neutral": compare_arm(r25, park, a.bootstrap, 751),
        },
        "governance": {
            "2024_transition_table_training_years": [2021, 2022, 2023],
            "distinct_transition_fit_years": [2021, 2022, 2023, 2024],
            "park_profile_source_year": 2024,
            "2025_outcomes_previously_inspected": True,
            "paired_calendar_date_bootstrap_draws": a.bootstrap,
            "causal_attribution": "NONE; park and K/BIP ablations test only two fixed mechanisms",
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "2024_gap_error": result["exact_pregame_i1_state"]["2024_selection_sample"]["paired_bottom_minus_top"]["gap_error"],
        "2025_gap_error": result["exact_pregame_i1_state"]["2025_inspected_sample"]["paired_bottom_minus_top"]["gap_error"],
        "park_delta_late": result["2025_fixed_mechanism_ablations"]["prior_season_park_in_i1_minus_neutral"]["all"]["delta_predicted_late"],
        "distinct_delta_late": result["2025_fixed_mechanism_ablations"]["distinct_strikeout_bip_transitions_minus_pooled"]["all"]["delta_predicted_late"],
    }, indent=2))


if __name__ == "__main__":
    main()

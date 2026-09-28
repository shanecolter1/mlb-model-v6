#!/usr/bin/env python3
"""Research-only direct I1 PA model with a jointly fitted home-side feature."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from audit_i1_opening_pa import read_first_innings
from build_i1_state_ab_inputs import EVENTS

EVENT_INDEX = {event: i for i, event in enumerate(EVENTS)}
REACH_INDEX = [EVENT_INDEX[e] for e in ("single", "double", "triple", "home_run", "walk", "hit_by_pitch")]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--retrosheet-dir", type=Path, required=True)
    p.add_argument("--vector-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def load_year(year, root, vector_root):
    games = read_first_innings(root / f"{year}plays.zip", year)
    artifact = json.loads((vector_root / f"i1_opening_pa_{year}_vectors.json").read_text())
    if artifact.get("season") != year or artifact.get("market_inputs_used") is not False:
        raise ValueError(f"Invalid pregame vector artifact for {year}")
    if artifact.get("observed_pa_outcomes_used_as_predictors") is not False:
        raise ValueError(f"Observed PA outcome entered pregame vector artifact in {year}")
    league = artifact["league_event_rates"]
    vectors = {(r["gid"], r["side"], r["slot"]): r for r in artifact["rows"]}
    if len(vectors) != artifact["n"]:
        raise ValueError(f"Duplicate pregame rows in {year}")
    rows = []
    excluded = Counter()
    for gid in {key[0] for key in vectors}:
        game = games.get(gid)
        if not game:
            excluded["MISSING_PAIRED_GAME"] += 1
            continue
        matched = []
        changed = False
        for side in ("top", "bottom"):
            for slot, pa in enumerate(game[side]["pa"][:3], 1):
                v = vectors[(gid, side, slot)]
                if v["batter_id"] != pa["batter_id"] or v["pitcher_id"] != pa["pitcher_id"]:
                    changed = True
                    break
                matched.append((side, slot, pa, v))
            if changed:
                break
        if changed or len(matched) != 6:
            excluded["OPENING_IDENTITY_CHANGED"] += 1
            continue
        for side, slot, pa, v in matched:
            rows.append({
                "gid": gid, "date": game["date"], "side": side, "slot": slot,
                "event": pa["event"], "baseline": [v["probabilities"][e] for e in EVENTS],
                "batter": [v["batter_rates"][e] for e in EVENTS],
                "pitcher": [v["pitcher_rates"][e] for e in EVENTS],
                "league": [league[e] for e in EVENTS],
            })
    return rows, dict(excluded)


def arrays(rows):
    batter = np.asarray([r["batter"] for r in rows], dtype=float)
    pitcher = np.asarray([r["pitcher"] for r in rows], dtype=float)
    league = np.asarray([r["league"] for r in rows], dtype=float)
    base = np.log(league)
    logit = lambda x: np.log(x / (1 - x))
    return {
        "base": base,
        "batter_delta": logit(batter) - logit(league),
        "pitcher_delta": logit(pitcher) - logit(league),
        "home": np.asarray([r["side"] == "bottom" for r in rows], dtype=float),
        "y": np.asarray([EVENT_INDEX[r["event"]] for r in rows], dtype=int),
        "baseline": np.asarray([r["baseline"] for r in rows], dtype=float),
        "dates": np.asarray([r["date"] for r in rows]),
    }


def probabilities(params, data, use_home):
    logits = data["base"] + params[0] * data["batter_delta"] + params[1] * data["pitcher_delta"]
    logits = logits + np.r_[params[2:9], 0]
    if use_home:
        logits = logits + data["home"][:, None] * np.r_[params[9:16], 0]
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit(data, use_home):
    n = len(data["y"])
    start = np.r_[0.5, 0.5, np.zeros(14 if use_home else 7)]
    truth = np.zeros((n, len(EVENTS)))
    truth[np.arange(n), data["y"]] = 1

    def objective(params):
        pred = probabilities(params, data, use_home)
        loss = -np.log(np.maximum(pred[np.arange(n), data["y"]], 1e-15)).mean()
        error = (pred - truth) / n
        grad = [
            np.sum(error * data["batter_delta"]),
            np.sum(error * data["pitcher_delta"]),
            *np.sum(error[:, :7], axis=0),
        ]
        if use_home:
            grad.extend(np.sum(error[:, :7] * data["home"][:, None], axis=0))
        return loss, np.asarray(grad)

    result = minimize(objective, start, jac=True, method="BFGS", options={"gtol": 1e-8, "maxiter": 500})
    if not result.success and np.linalg.norm(result.jac) > 1e-6:
        raise RuntimeError(f"I1 PA fit failed: {result.message}, gradient {np.linalg.norm(result.jac)}")
    return result.x, float(result.fun)


def metrics(data, pred):
    n = len(data["y"])
    truth = np.zeros_like(pred)
    truth[np.arange(n), data["y"]] = 1
    reach = pred[:, REACH_INDEX].sum(axis=1)
    actual_reach = np.isin(data["y"], REACH_INDEX)
    return {
        "n_pa": n,
        "multiclass_logloss": float(-np.log(np.maximum(pred[np.arange(n), data["y"]], 1e-15)).mean()),
        "multiclass_brier": float(np.sum((pred - truth) ** 2, axis=1).mean()),
        "modeled_reach_brier": float(np.mean((reach - actual_reach) ** 2)),
        "by_half": {
            label: {
                "n": int((data["home"] == home).sum()),
                "predicted_reach": float(reach[data["home"] == home].mean()),
                "actual_reach": float(actual_reach[data["home"] == home].mean()),
            }
            for label, home in (("top", 0), ("bottom", 1))
        },
    }


def paired_ci(data, before, after, draws, seed):
    n = len(data["y"])
    difference = -np.log(np.maximum(after[np.arange(n), data["y"]], 1e-15)) + np.log(
        np.maximum(before[np.arange(n), data["y"]], 1e-15)
    )
    by_date = defaultdict(lambda: [0.0, 0])
    for date, value in zip(data["dates"], difference):
        by_date[date][0] += float(value)
        by_date[date][1] += 1
    counts = np.asarray(list(by_date.values()))
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(counts), (draws, len(counts)))
    samples = counts[index].sum(axis=1)
    return [float(x) for x in np.quantile(samples[:, 0] / samples[:, 1], (0.025, 0.975))]


def evaluate(train_rows, test_rows, draws, seed):
    training, test = arrays(train_rows), arrays(test_rows)
    no_home, no_home_train_loss = fit(training, False)
    with_home, with_home_train_loss = fit(training, True)
    p_no_home = probabilities(no_home, test, False)
    p_with_home = probabilities(with_home, test, True)
    return {
        "train_n_pa": len(train_rows),
        "test_n_pa": len(test_rows),
        "training_logloss": {"no_home": no_home_train_loss, "with_home": with_home_train_loss},
        "test": {
            "existing_formula": metrics(test, test["baseline"]),
            "direct_no_home": metrics(test, p_no_home),
            "direct_with_home": metrics(test, p_with_home),
        },
        "paired_date_cluster_logloss_ci95": {
            "with_home_minus_existing": paired_ci(test, test["baseline"], p_with_home, draws, seed),
            "with_home_minus_no_home": paired_ci(test, p_no_home, p_with_home, draws, seed + 1),
        },
        "model_parameters": {
            "no_home": {"batter_weight": float(no_home[0]), "pitcher_weight": float(no_home[1]),
                        "event_intercepts": {e: float(no_home[i + 2]) for i, e in enumerate(EVENTS[:7])}},
            "with_home": {"batter_weight": float(with_home[0]), "pitcher_weight": float(with_home[1]),
                          "event_intercepts": {e: float(with_home[i + 2]) for i, e in enumerate(EVENTS[:7])},
                          "home_event_terms": {e: float(with_home[i + 9]) for i, e in enumerate(EVENTS[:7])}},
        },
    }


def main():
    args = arguments()
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    rows, exclusions = {}, {}
    for year in (2022, 2023, 2024):
        rows[year], exclusions[str(year)] = load_year(year, args.retrosheet_dir, args.vector_dir)
    result = {
        "version": "i1-direct-pa-home-candidate-v1",
        "status": "RESEARCH_ONLY_NOT_PROMOTED",
        "market_inputs_used": False,
        "observed_pa_outcomes_used_as_predictors": False,
        "model": "single multinomial softmax; log prior-season league event rate offset, fitted shared batter and pitcher logit deltas, event intercepts, optional home-side event terms; no additional shrinkage or PA calibration",
        "event_order": EVENTS,
        "reference_event": "ball_in_play_out",
        "chronological_2023_check": evaluate(rows[2022], rows[2023], args.bootstrap, 523),
        "descriptive_2024_check": evaluate(rows[2022] + rows[2023], rows[2024], args.bootstrap, 524),
        "excluded_games": exclusions,
        "governance": {
            "first_three_i1_pa_only": True,
            "2024_previously_inspected": True,
            "full_i1_slot_evaluation": False,
            "full_i2_evaluation": False,
            "live_model_changed": False,
            "candidate_selection": "NONE; diagnostic arms retained for comparison",
            "bootstrap_scope": "paired calendar date conditional on fitted model, no fit uncertainty",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    frozen = {
        "version": "i1-direct-pa-home-research-2022-2023-v1",
        "status": "RESEARCH_ONLY_NOT_PROMOTED",
        "market_inputs_used": False,
        "observed_2024_outcomes_used_for_fit": False,
        "training_years": [2022, 2023],
        "training_pa": result["descriptive_2024_check"]["train_n_pa"],
        "event_order": EVENTS,
        "reference_event": "ball_in_play_out",
        "parameters": result["descriptive_2024_check"]["model_parameters"]["with_home"],
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({
        "2023": {k:v["multiclass_logloss"] for k,v in result["chronological_2023_check"]["test"].items()},
        "2024": {k:v["multiclass_logloss"] for k,v in result["descriptive_2024_check"]["test"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()

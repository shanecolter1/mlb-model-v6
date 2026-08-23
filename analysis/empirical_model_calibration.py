#!/usr/bin/env python3
"""Empirical calibration audit for the MLB live half-inning model.

Uses only historical baseball data already stored in the repository:
- data/derived/i2/i2_play_calibration.json (2021-2025 Retrosheet PA transitions)
- data/derived/i2/i2_state_compact_2021.csv ... _2025.csv (I1->I2 pitcher continuation)

No sportsbook or market information is read.

Outputs are written to data/derived/model_calibration/:
- runner_transition_summary.csv
- pitcher_i2_continuation_by_pitch_count.csv
- pitcher_i2_continuation_by_runs.csv
- pitcher_i2_continuation_logit.json
- empirical_model_calibration.json
"""
from __future__ import annotations

import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "data/derived/i2/i2_play_calibration.json"
I2_GLOB = str(ROOT / "data/derived/i2/i2_state_compact_20*.csv")
OUT = ROOT / "data/derived/model_calibration"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def weighted_probability(rows, predicate):
    n = sum(int(r.get("n", 0)) for r in rows)
    if not n:
        return {"n": 0, "p": None}
    hit = sum(int(r.get("n", 0)) for r in rows if predicate(r))
    return {"n": n, "p": hit / n}


def state_rows(transitions, event, outs, mask):
    return transitions.get(f"{event}|{outs}|{mask}", [])


def runner_metrics(transitions):
    out = []
    legacy = {
        "runner_1b_scores_on_double": 0.45,
        "runner_2b_scores_on_single": 0.60,
        "runner_1b_to_3b_on_single": 0.30,
    }
    for outs in (0, 1, 2):
        # Runner on first, double. Any run on the play means the original runner scored;
        # two-run anomalies are retained because they are observed historical transitions.
        rows = state_rows(transitions, "double", outs, 1)
        m = weighted_probability(rows, lambda r: int(r.get("runs", 0)) >= 1)
        out.append({"metric": "runner_1b_scores_on_double", "outs": outs,
                    "legacy_p": legacy["runner_1b_scores_on_double"], **m})

        # Runner on second, single.
        rows = state_rows(transitions, "single", outs, 2)
        m = weighted_probability(rows, lambda r: int(r.get("runs", 0)) >= 1)
        out.append({"metric": "runner_2b_scores_on_single", "outs": outs,
                    "legacy_p": legacy["runner_2b_scores_on_single"], **m})

        # Runner on first, single -> original runner reaches third while batter occupies first.
        # Post mask bit 4 means third base occupied. Exclude plays adding an out so the measure
        # corresponds to clean first-to-third advancement rather than advancement with a putout.
        rows = state_rows(transitions, "single", outs, 1)
        m = weighted_probability(rows, lambda r: int(r.get("outs_added", 0)) == 0 and (int(r.get("post_mask", 0)) & 4))
        out.append({"metric": "runner_1b_to_3b_on_single", "outs": outs,
                    "legacy_p": legacy["runner_1b_to_3b_on_single"], **m})
    return out


def read_i2_rows():
    rows = []
    for path in sorted(glob.glob(I2_GLOB)):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append({
                        "season": int(r["season"]),
                        "i1_pa": float(r["i1_pa"]),
                        "i1_runs": float(r["i1_runs"]),
                        "i1_pitches": float(r["i1_pitches"]),
                        "i2_start_slot": float(r["i2_start_slot"]),
                        "same_pitcher_i2": int(r["same_pitcher_i2"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    return rows


def bin_summary(rows, field, bins):
    result = []
    for label, lo, hi in bins:
        sample = [r for r in rows if r[field] >= lo and (hi is None or r[field] <= hi)]
        if not sample:
            continue
        y = sum(r["same_pitcher_i2"] for r in sample)
        result.append({"bin": label, "n": len(sample), "same_pitcher_n": y, "p": y / len(sample)})
    return result


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logit(rows, iterations=12000, lr=0.03, l2=0.002):
    """Small dependency-free logistic model for I1->I2 continuation.

    Features are standardized: i1_pitches, i1_runs, i1_pa, i2_start_slot.
    This model is intentionally scoped to the first-to-second inning decision and must not
    be treated as a universal late-game pitcher continuation model.
    """
    names = ["i1_pitches", "i1_runs", "i1_pa", "i2_start_slot"]
    means = {k: sum(r[k] for r in rows) / len(rows) for k in names}
    sds = {}
    for k in names:
        v = sum((r[k] - means[k]) ** 2 for r in rows) / max(1, len(rows) - 1)
        sds[k] = math.sqrt(v) or 1.0
    X = [[1.0] + [(r[k] - means[k]) / sds[k] for k in names] for r in rows]
    y = [r["same_pitcher_i2"] for r in rows]
    w = [0.0] * len(X[0])
    n = len(rows)
    for _ in range(iterations):
        g = [0.0] * len(w)
        for xi, yi in zip(X, y):
            p = sigmoid(sum(a * b for a, b in zip(w, xi)))
            d = p - yi
            for j in range(len(w)):
                g[j] += d * xi[j]
        for j in range(len(w)):
            g[j] /= n
            if j:
                g[j] += l2 * w[j]
            w[j] -= lr * g[j]
    probs = [sigmoid(sum(a * b for a, b in zip(w, xi))) for xi in X]
    brier = sum((p - yi) ** 2 for p, yi in zip(probs, y)) / n
    eps = 1e-12
    logloss = -sum(yi * math.log(max(eps, p)) + (1 - yi) * math.log(max(eps, 1 - p)) for p, yi in zip(probs, y)) / n
    return {
        "scope": "first_inning_to_second_inning_only",
        "n": n,
        "base_rate": sum(y) / n,
        "features": names,
        "means": means,
        "sds": sds,
        "coefficients_standardized": {"intercept": w[0], **{k: w[i + 1] for i, k in enumerate(names)}},
        "brier_in_sample": brier,
        "logloss_in_sample": logloss,
        "warning": "Descriptive/fitted calibration on the same 2021-2025 sample. Validate out-of-sample before production replacement.",
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_json(CAL)
    transitions = cal["base_transitions"]
    runner = runner_metrics(transitions)
    write_csv(OUT / "runner_transition_summary.csv", runner,
              ["metric", "outs", "legacy_p", "n", "p"])

    i2 = read_i2_rows()
    pitch_bins = [("0-12", 0, 12), ("13-17", 13, 17), ("18-22", 18, 22),
                  ("23-27", 23, 27), ("28+", 28, None)]
    run_bins = [("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3+", 3, None)]
    pitch_summary = bin_summary(i2, "i1_pitches", pitch_bins)
    run_summary = bin_summary(i2, "i1_runs", run_bins)
    write_csv(OUT / "pitcher_i2_continuation_by_pitch_count.csv", pitch_summary,
              ["bin", "n", "same_pitcher_n", "p"])
    write_csv(OUT / "pitcher_i2_continuation_by_runs.csv", run_summary,
              ["bin", "n", "same_pitcher_n", "p"])

    logit = fit_logit(i2) if i2 else {"error": "No I2 rows found"}
    with (OUT / "pitcher_i2_continuation_logit.json").open("w", encoding="utf-8") as f:
        json.dump(logit, f, indent=2)

    report = {
        "version": "empirical-calibration-v1",
        "source_seasons": cal.get("seasons"),
        "regular_season_plate_appearances": cal.get("regular_season_plate_appearances"),
        "market_inputs_used": False,
        "runner_transition_replacements": runner,
        "pitcher_i2_continuation": {
            "n": len(i2),
            "by_i1_pitch_count": pitch_summary,
            "by_i1_runs": run_summary,
            "logit": logit,
            "production_scope_limit": "Only first-inning to second-inning continuation is identified by these compact files. Do not extrapolate this fit to later innings without a full appearance-level historical dataset."
        },
        "not_yet_identified": {
            "batter_pitcher_blend": "Requires historical PA rows with batter ID, pitcher ID, event outcome, event date, and leakage-safe pre-PA/pregame player histories.",
            "live_pitcher_deterioration": "Requires appearance/pitch-level historical data linking pitch count, TTO, velocity, contact quality, command and subsequent PA outcomes.",
            "confidence_shrinkage": "Requires out-of-sample prediction ledger containing model probability, realized outcome, data coverage and projection/live flags."
        }
    }
    with (OUT / "empirical_model_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

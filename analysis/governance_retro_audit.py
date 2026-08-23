#!/usr/bin/env python3
"""Retroactive anti-overfitting governance audit for empirical MLB calibration.

Historical baseball data only. No sportsbook/market inputs.

This audit does NOT silently approve model components. It writes explicit PASS/WARNING/BLOCKED
findings and blocks raw sparse transition states from production eligibility.

Outputs:
  data/derived/model_calibration/governance_retro_audit.json
  data/derived/model_calibration/transition_state_support.csv
  data/derived/model_calibration/pitcher_continuation_2025_benchmark.json
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

# Statistically motivated raw-state support gate:
# worst-case Bernoulli 95% margin of error <= 5 percentage points requires n >= 385.
# This is a governance diagnostic, not a fitted baseball coefficient.
RAW_STATE_MIN_N = 385


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def state_rows(transitions, event, outs, mask):
    return transitions.get(f"{event}|{outs}|{mask}", [])


def state_n(rows):
    return sum(int(r.get("n", 0)) for r in rows)


def support_audit(transitions):
    event_map = {
        "bb": ["walk"],
        "single": ["single"],
        "double": ["double"],
        "triple": ["triple"],
        "hr": ["home_run"],
        "out": ["strikeout", "ball_in_play_out"],
    }
    rows = []
    for model_event, hist_events in event_map.items():
        for outs in (0, 1, 2):
            for mask in range(8):
                n = sum(state_n(state_rows(transitions, he, outs, mask)) for he in hist_events)
                status = "PASS" if n >= RAW_STATE_MIN_N else "BLOCKED"
                rows.append({
                    "state": f"{model_event}|{outs}|{mask}",
                    "model_event": model_event,
                    "outs": outs,
                    "base_mask": mask,
                    "n": n,
                    "raw_probability_status": status,
                    "reason": "raw state support adequate" if status == "PASS" else "raw state too sparse; empirical hierarchical shrinkage required",
                })
    return rows


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
                        "y": int(r["same_pitcher_i2"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
    return rows


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1 / (1 + e)
    e = math.exp(z)
    return e / (1 + e)


def fit_logit(train, l2=0.002, iterations=12000, lr=0.03):
    names = ["i1_pitches", "i1_runs", "i1_pa", "i2_start_slot"]
    means = {k: sum(r[k] for r in train) / len(train) for k in names}
    sds = {}
    for k in names:
        v = sum((r[k] - means[k]) ** 2 for r in train) / max(1, len(train) - 1)
        sds[k] = math.sqrt(v) or 1.0
    X = [[1.0] + [(r[k] - means[k]) / sds[k] for k in names] for r in train]
    y = [r["y"] for r in train]
    w = [0.0] * (len(names) + 1)
    n = len(train)
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
    return {"names": names, "means": means, "sds": sds, "w": w, "base_rate": sum(y) / n}


def predict(model, rows):
    out = []
    for r in rows:
        x = [1.0] + [(r[k] - model["means"][k]) / model["sds"][k] for k in model["names"]]
        out.append(sigmoid(sum(a * b for a, b in zip(model["w"], x))))
    return out


def score(probs, y):
    eps = 1e-12
    n = len(y)
    return {
        "n": n,
        "brier": sum((p - yi) ** 2 for p, yi in zip(probs, y)) / n,
        "logloss": -sum(yi * math.log(max(eps, p)) + (1 - yi) * math.log(max(eps, 1 - p)) for p, yi in zip(probs, y)) / n,
        "predicted_mean": sum(probs) / n,
        "actual_mean": sum(y) / n,
    }


def continuation_benchmark(rows):
    train = [r for r in rows if 2021 <= r["season"] <= 2024]
    test = [r for r in rows if r["season"] == 2025]
    if not train or not test:
        return {"status": "BLOCKED", "reason": "2021-2024 training or 2025 validation sample missing"}
    model = fit_logit(train)
    y = [r["y"] for r in test]
    p_model = predict(model, test)
    p_base = [model["base_rate"]] * len(test)
    sm = score(p_model, y)
    sb = score(p_base, y)
    brier_improvement = sb["brier"] - sm["brier"]
    logloss_improvement = sb["logloss"] - sm["logloss"]
    status = "PASS" if brier_improvement > 0 and logloss_improvement > 0 else "BLOCKED"
    return {
        "status": status,
        "train_seasons": [2021, 2022, 2023, 2024],
        "validation_season": 2025,
        "warning": "2025 has been viewed in prior development and is not a pristine final holdout.",
        "model": sm,
        "constant_training_base_rate_benchmark": sb,
        "brier_improvement": brier_improvement,
        "logloss_improvement": logloss_improvement,
        "acceptance_rule": "complex model must beat constant training-base-rate benchmark on both Brier score and log loss",
    }


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if fields:
            w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_json(CAL)
    transitions = cal["base_transitions"]
    support = support_audit(transitions)
    write_csv(OUT / "transition_state_support.csv", support)

    total = len(support)
    passed = sum(r["raw_probability_status"] == "PASS" for r in support)
    blocked = total - passed

    i2 = read_i2_rows()
    cont = continuation_benchmark(i2)
    with (OUT / "pitcher_continuation_2025_benchmark.json").open("w", encoding="utf-8") as f:
        json.dump(cont, f, indent=2)

    required_generated = [
        "production_pa_transition_table.json",
        "pitcher_i2_continuation_logit.json",
        "pitcher_i2_continuation_oos_by_season.csv",
        "empirical_model_calibration.json",
    ]
    missing = [x for x in required_generated if not (OUT / x).exists()]

    findings = [
        {
            "component": "raw event/out/base transition table",
            "status": "PASS" if blocked == 0 else "BLOCKED",
            "rule": "minimum sample support + hierarchical shrinkage",
            "finding": f"{passed}/{total} states meet n>={RAW_STATE_MIN_N}; {blocked} raw states require shrinkage before production use.",
            "remedy": "fit an empirical hierarchical shrinkage rule on chronological development data; do not invent a fixed pooling weight.",
            "holdout_effect": "none from this diagnostic alone",
        },
        {
            "component": "pitcher I1->I2 continuation",
            "status": cont.get("status", "BLOCKED"),
            "rule": "complexity gate + chronological validation",
            "finding": "2021-2024 fitted model compared with a constant training-base-rate benchmark on 2025.",
            "remedy": "promote only if both Brier and log loss improve; retain 2026 prospective data for forward testing.",
            "holdout_effect": "2025 is already partially consumed; 2026 remains the stronger forward test if frozen before evaluation.",
        },
        {
            "component": "calibration artifact pipeline",
            "status": "PASS" if not missing else "BLOCKED",
            "rule": "reproducibility / no silent failures",
            "finding": "all expected generated outputs present" if not missing else f"missing generated outputs: {', '.join(missing)}",
            "remedy": "fix/run workflow and require artifacts before any production integration.",
            "holdout_effect": "none",
        },
        {
            "component": "season stability of runner transitions",
            "status": "WARNING",
            "rule": "chronological validation",
            "finding": "current i2_play_calibration.json aggregates 2021-2025 transitions, so season-by-season transition stability cannot be audited from this artifact alone.",
            "remedy": "regenerate event/out/base transition tables by season or from retained event-level Retrosheet data before claiming temporal stability.",
            "holdout_effect": "2025 should not be redefined as pristine; use season splits as development diagnostics and preserve future 2026 observations for forward evaluation.",
        },
    ]

    overall = "BLOCKED" if any(x["status"] == "BLOCKED" for x in findings) else ("WARNING" if any(x["status"] == "WARNING" for x in findings) else "PASS")
    audit = {
        "version": "governance-retro-audit-v1",
        "overall_status": overall,
        "market_inputs_used": False,
        "raw_state_support_gate": {
            "min_n": RAW_STATE_MIN_N,
            "basis": "worst-case Bernoulli 95% margin of error <= 5 percentage points",
            "states_total": total,
            "states_pass": passed,
            "states_blocked_raw": blocked,
        },
        "findings": findings,
        "pitcher_continuation_benchmark": cont,
        "required_generated_artifacts_missing_at_runtime": missing,
        "production_promotion_allowed": overall == "PASS",
    }
    with (OUT / "governance_retro_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

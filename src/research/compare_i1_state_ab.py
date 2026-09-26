#!/usr/bin/env python3
"""Paired statistical comparison for league vs player-as-of I1 state generators."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def summary(rows, arm):
    ll = np.array([r[arm]["logloss"] for r in rows], dtype=float)
    bs = np.array([r[arm]["brier"] for r in rows], dtype=float)
    op = np.array([r[arm]["observed_probability"] for r in rows], dtype=float)
    acc = np.array([r[arm]["top1"] for r in rows], dtype=float)
    return {
        "n": int(len(rows)),
        "slot_logloss": float(ll.mean()),
        "slot_brier": float(bs.mean()),
        "mean_probability_on_observed_slot": float(op.mean()),
        "top1_accuracy": float(acc.mean()),
    }


def main():
    args = parse_args()
    x = json.loads(args.input.read_text())
    if x.get("market_inputs_used") is not False:
        raise RuntimeError("A/B evaluation must be baseball-only")
    if x.get("observed_i2_start_slot_used_as_predictor") is not False:
        raise RuntimeError("Observed I2 start slot leaked into predictors")

    rows = x["rows"]
    if len(rows) < 4000:
        raise RuntimeError(f"Insufficient paired half-innings: {len(rows)}")

    league = summary(rows, "league")
    player = summary(rows, "player_asof")

    dll = np.array(
        [r["player_asof"]["logloss"] - r["league"]["logloss"] for r in rows],
        dtype=float,
    )
    dbs = np.array(
        [r["player_asof"]["brier"] - r["league"]["brier"] for r in rows],
        dtype=float,
    )

    rng = np.random.default_rng(20240926)
    n = len(rows)
    boot_ll = np.empty(args.bootstrap)
    boot_bs = np.empty(args.bootstrap)
    for i in range(args.bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_ll[i] = dll[idx].mean()
        boot_bs[i] = dbs[idx].mean()

    ll_ci = [float(v) for v in np.quantile(boot_ll, [0.025, 0.975])]
    bs_ci = [float(v) for v in np.quantile(boot_bs, [0.025, 0.975])]

    mean_dll = float(dll.mean())
    mean_dbs = float(dbs.mean())

    # Precommitted conservative rule:
    # player-specific must improve both paired means, and the log-loss
    # bootstrap interval must be entirely below zero. Otherwise keep the
    # simpler league-average generator.
    player_pass = mean_dll < 0 and mean_dbs < 0 and ll_ci[1] < 0
    selected = "player_asof" if player_pass else "league"

    payload = {
        "version": "i1-state-ab-selection-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selection_year": int(x["season"]),
        "market_inputs_used": False,
        "target": "actual I2 starting batting slot",
        "evaluation_method": x.get("evaluation_method"),
        "arms": {
            "league": league,
            "player_asof": player,
        },
        "paired_deltas_player_minus_league": {
            "slot_logloss": mean_dll,
            "slot_brier": mean_dbs,
            "slot_logloss_bootstrap_95pct": ll_ci,
            "slot_brier_bootstrap_95pct": bs_ci,
        },
        "selection_rule": (
            "player_asof must lower both paired mean slot logloss and Brier, "
            "with logloss bootstrap 95% CI entirely below zero; otherwise league"
        ),
        "selected": selected,
        "governance": {
            "2025_full_i2_holdout_consumed": False,
            "observed_i2_runs_used": False,
            "observed_i2_start_slot_used_as_predictor": False,
            "sportsbook_or_market_inputs_used": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

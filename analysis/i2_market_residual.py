#!/usr/bin/env python3
"""Market-pricing residual for I2 research signals.

This is a POST-PREDICTION / research-market layer. It does not alter baseball
features or probabilities. Pregame full-game total is used only as a proxy for
how much of an I2 run environment the market is likely already pricing.

Residual lift = observed/estimated signal I2-under probability
                - historical I2-under probability at the same DK opening total.
"""
from __future__ import annotations
import csv
import argparse
from pathlib import Path

DEFAULT_PRIOR = Path("data/derived/i2/total_bucket_i2_fair_values_2021_2025.csv")


def american_from_probability(p: float) -> float:
    if not 0 < p < 1:
        raise ValueError("probability must be between 0 and 1")
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p


def load_total_prior(path: Path = DEFAULT_PRIOR) -> dict[float, float]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[float(r["opening_total"])] = float(r["p_i2_no_run"])
    return out


def residual(signal_under: float, opening_total: float, prior: dict[float, float]):
    if opening_total not in prior:
        raise KeyError(f"No historical I2 prior for opening total {opening_total}")
    market_proxy = prior[opening_total]
    return {
        "opening_total": opening_total,
        "signal_under_probability": signal_under,
        "market_proxy_under_probability": market_proxy,
        "residual_lift_pp": 100 * (signal_under - market_proxy),
        "signal_fair_under": american_from_probability(signal_under),
        "market_proxy_fair_under": american_from_probability(market_proxy),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-under", type=float, required=True,
                    help="Signal I2-under probability, e.g. 0.67857")
    ap.add_argument("--total", type=float, required=True,
                    help="DraftKings opening full-game total")
    ap.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    a = ap.parse_args()
    x = residual(a.signal_under, a.total, load_total_prior(a.prior))
    for k, v in x.items():
        print(f"{k},{v}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit historical top/bottom I2 outcomes without fitting the vNext model.

The 2021–2024 records provide a prior-period description of the half-inning
gap noticed in the already inspected 2025 replay. All I1/I2 states and
pitcher continuation fields here are observed outcomes, never predictors.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def read_pairs(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    games: dict[tuple[int, str], dict] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                year = int(row["season"])
                gid = row["gid"]
                half = row["half"]
                if half not in ("top", "bottom"):
                    raise ValueError(f"Invalid half {half}: {gid}")
                key = (year, gid)
                pair = games.setdefault(key, {})
                if half in pair:
                    raise ValueError(f"Duplicate {half} half: {year}/{gid}")
                pair[half] = row

    result = []
    excluded = []
    for (year, gid), pair in sorted(games.items()):
        if set(pair) != {"top", "bottom"}:
            raise ValueError(f"Incomplete game: {year}/{gid}")
        top, bottom = pair["top"], pair["bottom"]
        if top["date"] != bottom["date"]:
            # Suspended games can resume months later. The two halves did not
            # share a calendar-date cluster or pregame information cutoff.
            excluded.append({"season": year, "gid": gid, "reason": "HALVES_PLAYED_ON_DIFFERENT_DATES"})
            continue
        if not top["date"].startswith(str(year)):
            raise ValueError(f"Mismatched season and date: {year}/{gid}")
        result.append({
            "season": year,
            "gid": gid,
            "date": top["date"],
            "top_under": int(int(top["i2_runs"]) == 0),
            "bottom_under": int(int(bottom["i2_runs"]) == 0),
            "both_same_pitcher_i2": int(
                int(top["same_pitcher_i2"]) == 1
                and int(bottom["same_pitcher_i2"]) == 1
            ),
            "same_start_slot": int(top["i2_start_slot"] == bottom["i2_start_slot"]),
        })
    return result, excluded


def describe(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty half-inning comparison")
    n = len(rows)
    top = sum(r["top_under"] for r in rows)
    bottom = sum(r["bottom_under"] for r in rows)
    top_only = sum(r["top_under"] and not r["bottom_under"] for r in rows)
    bottom_only = sum(r["bottom_under"] and not r["top_under"] for r in rows)
    assert bottom - top == bottom_only - top_only
    return {
        "games": n,
        "top_under_rate": top / n,
        "bottom_under_rate": bottom / n,
        "bottom_minus_top_under": (bottom - top) / n,
        "discordant_top_only_under": top_only,
        "discordant_bottom_only_under": bottom_only,
    }


def date_cluster_ci(rows: list[dict], draws: int = 5000, seed: int = 20260927) -> list[float]:
    by_date: dict[tuple[int, str], list[int]] = defaultdict(list)
    for r in rows:
        by_date[(r["season"], r["date"])].append(r["bottom_under"] - r["top_under"])
    clusters = [(sum(values), len(values)) for values in by_date.values()]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        picks = rng.choices(clusters, k=len(clusters))
        means.append(sum(s for s, _ in picks) / sum(n for _, n in picks))
    means.sort()
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def audit(rows: list[dict], excluded: list[dict] | None = None, draws: int = 5000) -> dict:
    historical = [r for r in rows if r["season"] in (2021, 2022, 2023, 2024)]
    inspected = [r for r in rows if r["season"] == 2025]
    if len({r["season"] for r in historical}) != 4 or not inspected:
        raise ValueError("Expected complete 2021–2025 input years")
    groups = {
        "both_starters_began_i2": [r for r in historical if r["both_same_pitcher_i2"]],
        "at_least_one_starter_changed": [r for r in historical if not r["both_same_pitcher_i2"]],
        "same_i2_start_slot": [r for r in historical if r["same_start_slot"]],
        "different_i2_start_slots": [r for r in historical if not r["same_start_slot"]],
    }
    return {
        "version": "i2-half-asymmetry-audit-v1",
        "market_inputs_used": False,
        "model_fitted": False,
        "promotion_status": "SHADOW_ONLY",
        "excluded_games": excluded or [],
        "hypothesis_source": "Retrospective after observing 2025 bottom-half bias; 2021–2024 is historical context, not an independent preregistered test.",
        "prior_2021_2024": {
            **describe(historical),
            "bottom_minus_top_ci95_date_cluster": date_cluster_ci(historical, draws),
            "seasons": {str(y): describe([r for r in historical if r["season"] == y]) for y in range(2021, 2025)},
            "descriptive_subgroups": {name: describe(group) for name, group in groups.items()},
        },
        "inspected_2025_context_only": describe(inspected),
        "limitations": [
            "This audits outcomes, not matched game-level model residuals in 2021–2024.",
            "Starter continuation and I2 start slot are observed after first pitch; subgroup results cannot enter pregame prediction.",
            "The paired comparison shares a park per game, but does not isolate home batting context from team strength, pitcher quality, or lineup state.",
            "Calendar-date bootstrap conditions on observed seasons and does not account for model training or hypothesis selection.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("data/derived/i2"))
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/phase7d_half_asymmetry.json"))
    args = p.parse_args()
    paths = [args.input_dir / f"i2_state_compact_{year}.csv" for year in range(2021, 2026)]
    rows, excluded = read_pairs(paths)
    result = audit(rows, excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

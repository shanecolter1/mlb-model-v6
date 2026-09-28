#!/usr/bin/env python3
"""First-three-batter I1 PA audit with pregame 2024 vectors as a diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_i1_state_ab_inputs import EVENTS, event_class

REACH = ("single", "double", "triple", "home_run", "walk", "hit_by_pitch")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--retrosheet-dir", type=Path, required=True)
    p.add_argument("--pregame-vectors-2024", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def read_first_innings(path, year):
    halves = defaultdict(lambda: {"pa": [], "non_pa_outs": 0, "date": None})
    with zipfile.ZipFile(path) as zf:
        with zf.open(f"{year}plays.csv") as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            for row in reader:
                if row["gametype"] != "regular" or row["inning"] != "1":
                    continue
                side = "top" if row["top_bot"] == "0" else "bottom"
                half = halves[(row["gid"], side)]
                half["date"] = row["date"]
                outs_added = max(0, int(row["outs_post"]) - int(row["outs_pre"]))
                if row["pa"] != "1":
                    half["non_pa_outs"] += outs_added
                    continue
                event = event_class(row)
                if event not in EVENTS:
                    raise ValueError(f"Unclassified PA: {year} {row['gid']}")
                half["pa"].append({
                    "event": event, "outs_added": outs_added,
                    "batter_id": row["batter"], "pitcher_id": row["pitcher"],
                })
    games = {}
    for gid in {key[0] for key in halves}:
        top, bottom = halves.get((gid, "top")), halves.get((gid, "bottom"))
        if not top or not bottom or top["date"] != bottom["date"]:
            continue
        if len(top["pa"]) < 3 or len(bottom["pa"]) < 3:
            continue
        games[gid] = {"date": top["date"], "top": top, "bottom": bottom}
    return games


def ci_by_date(items, value, draws, seed):
    by_date = defaultdict(lambda: [0.0, 0])
    for item in items:
        by_date[item["date"]][0] += value(item)
        by_date[item["date"]][1] += 1
    values = np.asarray(list(by_date.values()), dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    sums = values[indices].sum(axis=1)
    return [float(x) for x in np.quantile(sums[:, 0] / sums[:, 1], [0.025, 0.975])]


def rate(half, events):
    return sum(pa["event"] in events for pa in half["pa"][:3]) / 3


def observed_summary(games, draws, seed):
    pairs = list(games.values())
    by_side = {}
    for side in ("top", "bottom"):
        first_three = [pa for g in pairs for pa in g[side]["pa"][:3]]
        n = len(first_three)
        counts = Counter(pa["event"] for pa in first_three)
        conditional_outs = {}
        for event in EVENTS:
            event_rows = [pa for pa in first_three if pa["event"] == event]
            conditional_outs[event] = sum(pa["outs_added"] for pa in event_rows) / len(event_rows) if event_rows else None
        by_side[side] = {
            "n_opening_pa": n,
            "event_rates": {event: counts[event] / n for event in EVENTS},
            "modeled_reach_rate": sum(counts[e] for e in REACH) / n,
            "strikeout_rate": counts["strikeout"] / n,
            "mean_outs_added_per_opening_pa": sum(pa["outs_added"] for pa in first_three) / n,
            "conditional_outs_added_by_event": conditional_outs,
            "non_pa_outs_per_i1_half": sum(g[side]["non_pa_outs"] for g in pairs) / len(pairs),
        }
    n = len(pairs)
    gap = lambda group: sum(rate(g["bottom"], group) - rate(g["top"], group) for g in pairs) / n
    by_side["paired_bottom_minus_top"] = {
        "n_games": n,
        "reach_rate_gap": gap(REACH),
        "reach_gap_ci95": ci_by_date(
            pairs, lambda g: rate(g["bottom"], REACH) - rate(g["top"], REACH), draws, seed
        ),
        "strikeout_rate_gap": gap(("strikeout",)),
        "strikeout_gap_ci95": ci_by_date(
            pairs, lambda g: rate(g["bottom"], ("strikeout",)) - rate(g["top"], ("strikeout",)), draws, seed + 1
        ),
    }
    top, bottom = by_side["top"], by_side["bottom"]
    # Exact symmetric Oaxaca/Kitagawa identity for observed outs per opening PA.
    # Event mix and conditional-outs terms are descriptive; the latter also
    # contains base-state and lineup-selection differences.
    mix = sum(
        (bottom["event_rates"][e] - top["event_rates"][e]) *
        (bottom["conditional_outs_added_by_event"][e] + top["conditional_outs_added_by_event"][e]) / 2
        for e in EVENTS
    )
    within = sum(
        (bottom["conditional_outs_added_by_event"][e] - top["conditional_outs_added_by_event"][e]) *
        (bottom["event_rates"][e] + top["event_rates"][e]) / 2
        for e in EVENTS
    )
    total = bottom["mean_outs_added_per_opening_pa"] - top["mean_outs_added_per_opening_pa"]
    if abs(mix + within - total) > 1e-10:
        raise ValueError("Outs-per-PA descriptive decomposition failed")
    by_side["outs_gap_decomposition"] = {
        "bottom_minus_top_outs_per_pa": total,
        "event_mix_component": mix,
        "conditional_outs_component": within,
        "interpretation": "Descriptive symmetric identity, not causal; conditional term includes base-state differences",
    }
    return by_side


def pregame_2024(games, vectors, draws):
    if vectors.get("season") != 2024 or vectors.get("market_inputs_used") is not False:
        raise ValueError("Pregame 2024 vectors invalid")
    if vectors.get("observed_pa_outcomes_used_as_predictors") is not False:
        raise ValueError("Observed PA outcomes leaked into pregame vectors")
    keyed = {(r["gid"], r["side"], r["slot"]): r for r in vectors["rows"]}
    if len(keyed) != vectors["n"]:
        raise ValueError("Duplicate pregame vector")
    matched = []
    exclusions = Counter()
    for gid in {key[0] for key in keyed}:
        game = games.get(gid)
        if not game:
            exclusions["MISSING_MATCHED_GAME"] += 1
            continue
        observations = []
        identity_changed = False
        for side in ("top", "bottom"):
            for slot, pa in enumerate(game[side]["pa"][:3], 1):
                vector = keyed[(gid, side, slot)]
                if vector["batter_id"] != pa["batter_id"] or vector["pitcher_id"] != pa["pitcher_id"]:
                    exclusions["OPENING_IDENTITY_CHANGED"] += 1
                    identity_changed = True
                    break
                observations.append({"side": side, "slot": slot, "actual": pa["event"], "pred": vector["probabilities"]})
            if identity_changed:
                break
        if identity_changed or len(observations) != 6:
            continue
        matched.append({"date": game["date"], "gid": gid, "pa": observations})

    result = {"n_games": len(matched), "n_opening_pa": 6 * len(matched), "exclusions": dict(exclusions)}
    for side in ("top", "bottom"):
        observations = [pa for g in matched for pa in g["pa"] if pa["side"] == side]
        n = len(observations)
        score = {}
        for label, events in (("modeled_reach", REACH), ("strikeout", ("strikeout",)), ("ball_in_play_out", ("ball_in_play_out",))):
            predicted = sum(sum(pa["pred"][e] for e in events) for pa in observations) / n
            actual = sum(pa["actual"] in events for pa in observations) / n
            score[label] = {"predicted": predicted, "actual": actual, "pred_minus_actual": predicted - actual}
        result[side] = score

    def game_gap(g):
        def side_error(side):
            obs = [pa for pa in g["pa"] if pa["side"] == side]
            return sum(sum(pa["pred"][e] for e in REACH) - int(pa["actual"] in REACH) for pa in obs) / 3
        return side_error("bottom") - side_error("top")

    result["bottom_minus_top_reach_error_gap"] = sum(game_gap(g) for g in matched) / len(matched)
    result["reach_error_gap_ci95"] = ci_by_date(matched, game_gap, draws, 2024)
    return result


def main():
    args = parse_args()
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws required")
    history = {
        str(y): read_first_innings(args.retrosheet_dir / f"{y}plays.zip", y)
        for y in range(2021, 2025)
    }
    vectors = json.loads(args.pregame_vectors_2024.read_text())
    result = {
        "version": "i1-opening-pa-side-audit-v1",
        "status": "SHADOW_ONLY_DESCRIPTIVE",
        "market_inputs_used": False,
        "observed_pa_outcomes_used_as_predictors": False,
        "scope": "Opening three first-inning PAs per half; Retrosheet model-compatible event taxonomy",
        "observed_by_season": {
            str(y): observed_summary(history[str(y)], args.bootstrap, 600 + y)
            for y in range(2021, 2025)
        },
        "pregame_2024_opening_vectors": pregame_2024(history["2024"], vectors, args.bootstrap),
        "governance": {
            "source": "Retrosheet parsed regular-season play-by-play, 2021-2024",
            "source_archive_sha256": {
                str(y): hashlib.sha256((args.retrosheet_dir / f"{y}plays.zip").read_bytes()).hexdigest()
                for y in range(2021, 2025)
            },
            "2024_hypothesis_post_selection": True,
            "candidate_fit": False,
            "paired_calendar_date_bootstrap_draws": args.bootstrap,
            "full_i2_replay": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "historical_reach_gaps": {y:result["observed_by_season"][y]["paired_bottom_minus_top"]["reach_rate_gap"] for y in result["observed_by_season"]},
        "pregame_2024_games": result["pregame_2024_opening_vectors"]["n_games"],
        "2024_reach_residual_gap": result["pregame_2024_opening_vectors"]["bottom_minus_top_reach_error_gap"],
    }, indent=2))


if __name__ == "__main__":
    main()

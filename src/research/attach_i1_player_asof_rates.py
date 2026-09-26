#!/usr/bin/env python3
"""Attach leakage-safe season-to-date player I1 event rates to 2025 replay inputs.

This reconstructs the old live I1 state-generator inputs using Retrosheet PAs
strictly before each game date. Hitter rates use a 100-PA league prior and
pitcher rates a 180-BF league prior, matching run_i2_vnext_today.mjs.

For same-day doubleheaders, both games deliberately use the start-of-day
snapshot. This is conservative and prevents first-game outcomes from leaking
into a second-game prediction when exact pregame timing is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

EVENTS = [
    "single","double","triple","home_run","walk","hit_by_pitch",
    "strikeout","ball_in_play_out",
]


def as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def event_class(row: dict) -> str | None:
    if as_int(row.get("pa")) != 1:
        return None
    if as_int(row.get("single")):
        return "single"
    if as_int(row.get("double")):
        return "double"
    if as_int(row.get("triple")):
        return "triple"
    if as_int(row.get("hr")):
        return "home_run"
    if as_int(row.get("hbp")):
        return "hit_by_pitch"
    if as_int(row.get("walk")):
        return "walk"
    if as_int(row.get("k")):
        return "strikeout"
    return "ball_in_play_out"


def date_key(value: object) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8]


def smooth(counts: Counter, league: dict[str, float], strength: float) -> tuple[dict[str, float], int]:
    n = int(sum(counts.values()))
    denom = n + float(strength)
    rates = {
        k: (float(counts.get(k, 0)) + float(league[k]) * float(strength)) / denom
        for k in EVENTS
    }
    total = sum(rates.values())
    return ({k: rates[k] / total for k in EVENTS}, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--retrosheet-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    replay = json.loads(args.input.read_text())
    if replay.get("market_inputs_used") is not False:
        raise RuntimeError("Replay input is not market-isolated")
    league = replay.get("i1_state_model", {}).get("event_rates")
    if not league or any(k not in league for k in EVENTS):
        raise RuntimeError("Replay input missing league I1 event rates")

    games_by_date: dict[str, list[dict]] = defaultdict(list)
    for game in replay.get("games", []):
        games_by_date[date_key(game.get("date"))].append(game)

    season = int(replay["season"])
    rows_by_date: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(args.retrosheet_zip) as zf:
        member = f"{season}plays.csv"
        with zf.open(member) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")):
                if row.get("gametype") != "regular":
                    continue
                ev = event_class(row)
                if ev is None:
                    continue
                rows_by_date[date_key(row.get("date"))].append(row)

    batter_hist: dict[str, Counter] = defaultdict(Counter)
    pitcher_hist: dict[str, Counter] = defaultdict(Counter)
    attached_games = 0
    zero_batter_support = 0
    zero_pitcher_support = 0

    all_dates = sorted(set(rows_by_date) | set(games_by_date))
    for d in all_dates:
        # Snapshot before ingesting any outcomes from this date.
        for game in games_by_date.get(d, []):
            for side in ("away_lineup", "home_lineup"):
                for hitter in game.get(side, []):
                    retro = str(hitter.get("retro") or "")
                    rates, n = smooth(batter_hist[retro], league, 100.0)
                    hitter["i1_event_rates_asof"] = rates
                    hitter["i1_asof_pa"] = n
                    if n == 0:
                        zero_batter_support += 1
            for side in ("away_starter", "home_starter"):
                pitcher = game.get(side) or {}
                retro = str(pitcher.get("retro") or "")
                rates, n = smooth(pitcher_hist[retro], league, 180.0)
                pitcher["i1_event_rates_asof"] = rates
                pitcher["i1_asof_bf"] = n
                if n == 0:
                    zero_pitcher_support += 1
            attached_games += 1

        for row in rows_by_date.get(d, []):
            ev = event_class(row)
            if ev is None:
                continue
            batter = str(row.get("batter") or "")
            pitcher = str(row.get("pitcher") or "")
            if batter:
                batter_hist[batter][ev] += 1
            if pitcher:
                pitcher_hist[pitcher][ev] += 1

    if attached_games != len(replay.get("games", [])):
        raise RuntimeError(f"Attached {attached_games} games; expected {len(replay.get('games', []))}")

    replay["i1_player_asof_model"] = {
        "status": "AVAILABLE_FOR_AB_TEST",
        "source": "Retrosheet regular-season PAs strictly before game date",
        "hitter_prior_pa": 100,
        "pitcher_prior_bf": 180,
        "prior_distribution": "same prior-season league event vector used by league-average arm",
        "same_day_doubleheader_policy": "start-of-day snapshot for both games",
        "market_inputs_used": False,
        "games": attached_games,
        "zero_support_lineup_slots": zero_batter_support,
        "zero_support_starters": zero_pitcher_support,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, separators=(",", ":")))
    print(json.dumps(replay["i1_player_asof_model"], indent=2))


if __name__ == "__main__":
    main()

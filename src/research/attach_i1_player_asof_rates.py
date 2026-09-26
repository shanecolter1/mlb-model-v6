#!/usr/bin/env python3
"""Attach leakage-safe season-to-date player I1 event rates to replay inputs.

This reconstructs the selected live I1 state-generator inputs using Retrosheet
PAs strictly before each game. Hitter rates use a 100-PA league prior and
pitcher rates a 180-BF league prior, matching the 2024 A/B selection test and
run_i2_vnext_today.mjs.

Same-day doubleheaders are ordered by Retrosheet game identity/order, so Game 1
outcomes are available to Game 2 exactly as they would be pregame.
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


def game_order(gid: str, rows: list[dict]) -> tuple[str, int, str]:
    first = rows[0]
    date = date_key(first.get("date"))
    game_num = as_int(first.get("number") or first.get("game_num") or first.get("dh") or 0)
    return (date, game_num, gid)


def smooth(
    counts: Counter,
    league: dict[str, float],
    strength: float,
) -> tuple[dict[str, float], int]:
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

    replay_by_gid = {str(g.get("gid") or ""): g for g in replay.get("games", [])}
    if len(replay_by_gid) != len(replay.get("games", [])):
        raise RuntimeError("Replay contains duplicate or blank Retrosheet game IDs")

    season = int(replay["season"])
    source_games: dict[str, list[dict]] = defaultdict(list)
    with zipfile.ZipFile(args.retrosheet_zip) as zf:
        member = f"{season}plays.csv"
        if member not in zf.namelist():
            raise RuntimeError(f"{args.retrosheet_zip} does not contain {member}")
        with zf.open(member) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")):
                if row.get("gametype") != "regular":
                    continue
                gid = str(row.get("gid") or "").strip()
                if gid:
                    source_games[gid].append(row)

    batter_hist: dict[str, Counter] = defaultdict(Counter)
    pitcher_hist: dict[str, Counter] = defaultdict(Counter)
    attached_games = 0
    zero_batter_support = 0
    zero_pitcher_support = 0
    same_day_later_games = 0
    seen_dates: Counter = Counter()

    for gid, rows in sorted(source_games.items(), key=lambda kv: game_order(kv[0], kv[1])):
        date = date_key(rows[0].get("date"))
        game = replay_by_gid.get(gid)

        if game is not None:
            seen_dates[date] += 1
            if seen_dates[date] > 1:
                same_day_later_games += 1

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

        # Ingest this game's PAs only after snapshotting its pregame inputs.
        # Excluded replay games still contribute to future as-of statistics.
        for row in rows:
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
        missing = sorted(set(replay_by_gid) - set(source_games))
        raise RuntimeError(
            f"Attached {attached_games} games; expected {len(replay.get('games', []))}; "
            f"missing source gids={missing[:10]}"
        )

    replay["i1_player_asof_model"] = {
        "status": "SELECTED_BY_2024_OOS_AB",
        "source": "Retrosheet regular-season PAs strictly before each game",
        "hitter_prior_pa": 100,
        "pitcher_prior_bf": 180,
        "prior_distribution": "same prior-season league event vector used by league-average arm",
        "same_day_doubleheader_policy": "game-by-game chronological snapshot; earlier game contributes to later game",
        "same_day_later_games": same_day_later_games,
        "market_inputs_used": False,
        "games": attached_games,
        "zero_support_lineup_slots": zero_batter_support,
        "zero_support_starters": zero_pitcher_support,
        "selection_evidence": "2024 I1 starting-slot A/B selected player_asof",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(replay, separators=(",", ":")))
    print(json.dumps(replay["i1_player_asof_model"], indent=2))


if __name__ == "__main__":
    main()

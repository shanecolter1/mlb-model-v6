#!/usr/bin/env python3
"""Build leakage-safe 2024 inputs for the I1-state generator A/B test.

The two arms are fixed before evaluation:
A) prior-season league-average PA event vector;
B) current-season-to-date player event rates, shrunk exactly like the old live
   I1 engine (100 PA hitter prior, 180 BF pitcher prior).

Only pregame information is attached to each game. Actual I2 starting slots are
retained strictly as evaluation targets.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

EVENTS = [
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
]


def as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2024)
    p.add_argument("--retrosheet-zip", type=Path, required=True)
    p.add_argument("--prior-retrosheet-zip", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/derived/i2_vnext/i1_state_ab_2024_inputs.json"))
    return p.parse_args()


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


def member_rows(zip_path: Path, season: int):
    with zipfile.ZipFile(zip_path) as zf:
        member = f"{season}plays.csv"
        if member not in zf.namelist():
            raise RuntimeError(f"{zip_path} does not contain {member}")
        with zf.open(member) as raw:
            yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))


def league_event_rates(zip_path: Path, season: int) -> dict[str, float]:
    counts = Counter()
    for row in member_rows(zip_path, season):
        if row.get("gametype") != "regular":
            continue
        event = event_class(row)
        if event:
            counts[event] += 1
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("No prior-season regular-season PA events")
    return {k: counts[k] / total for k in EVENTS}


def load_games(zip_path: Path, season: int) -> dict[str, list[dict]]:
    games: dict[str, list[dict]] = defaultdict(list)
    for row in member_rows(zip_path, season):
        if row.get("gametype") != "regular":
            continue
        gid = str(row.get("gid") or "").strip()
        if gid:
            games[gid].append(row)
    return games


def lineup_from_first(row: dict) -> list[str]:
    return [str(row.get(f"l{i}") or "").strip() for i in range(1, 10)]


def shrunk_rates(counts: Counter, denom: int, league: dict[str, float], strength: float) -> dict[str, float]:
    d = max(0, int(denom))
    raw = {k: (counts[k] + league[k] * strength) / (d + strength) for k in EVENTS}
    total = sum(raw.values())
    return {k: raw[k] / total for k in EVENTS}


def slot_of(player: str, lineup: list[str]) -> int | None:
    try:
        return lineup.index(player) + 1
    except ValueError:
        return None


def game_order(gid: str, rows: list[dict]):
    first = rows[0]
    date = str(first.get("date") or "")
    game_num = as_int(first.get("number") or first.get("game_num") or first.get("dh") or 0)
    return (date, game_num, gid)


def main() -> None:
    args = parse_args()
    prior = args.season - 1
    league = league_event_rates(args.prior_retrosheet_zip, prior)
    games = load_games(args.retrosheet_zip, args.season)

    hitter_counts: dict[str, Counter] = defaultdict(Counter)
    pitcher_counts: dict[str, Counter] = defaultdict(Counter)
    hitter_pa: Counter = Counter()
    pitcher_bf: Counter = Counter()

    output = []
    exclusions = Counter()
    slot_targets = 0

    for gid, rows in sorted(games.items(), key=lambda kv: game_order(kv[0], kv[1])):
        top1 = [r for r in rows if as_int(r.get("inning")) == 1 and as_int(r.get("top_bot")) == 0]
        bot1 = [r for r in rows if as_int(r.get("inning")) == 1 and as_int(r.get("top_bot")) == 1]
        top2 = [r for r in rows if as_int(r.get("inning")) == 2 and as_int(r.get("top_bot")) == 0]
        bot2 = [r for r in rows if as_int(r.get("inning")) == 2 and as_int(r.get("top_bot")) == 1]

        record = None
        if not top1 or not bot1 or not top2 or not bot2:
            exclusions["MISSING_I1_OR_I2"] += 1
        else:
            away_lineup = lineup_from_first(top1[0])
            home_lineup = lineup_from_first(bot1[0])
            home_starter = str(top1[0].get("pitcher") or "").strip()
            away_starter = str(bot1[0].get("pitcher") or "").strip()
            if any(not x for x in away_lineup + home_lineup) or not home_starter or not away_starter:
                exclusions["MISSING_STARTING_INPUT"] += 1
            else:
                top_slot = slot_of(str(top2[0].get("batter") or "").strip(), away_lineup)
                bottom_slot = slot_of(str(bot2[0].get("batter") or "").strip(), home_lineup)
                slot_targets += int(top_slot is not None) + int(bottom_slot is not None)
                record = {
                    "gid": gid,
                    "date": str(top1[0].get("date") or ""),
                    "away_lineup": [
                        {
                            "id": p,
                            "i1_event_rates_asof": shrunk_rates(hitter_counts[p], hitter_pa[p], league, 100.0),
                            "season_pa_before_game": int(hitter_pa[p]),
                        }
                        for p in away_lineup
                    ],
                    "home_lineup": [
                        {
                            "id": p,
                            "i1_event_rates_asof": shrunk_rates(hitter_counts[p], hitter_pa[p], league, 100.0),
                            "season_pa_before_game": int(hitter_pa[p]),
                        }
                        for p in home_lineup
                    ],
                    "away_starter": {
                        "id": away_starter,
                        "i1_event_rates_asof": shrunk_rates(pitcher_counts[away_starter], pitcher_bf[away_starter], league, 180.0),
                        "season_bf_before_game": int(pitcher_bf[away_starter]),
                    },
                    "home_starter": {
                        "id": home_starter,
                        "i1_event_rates_asof": shrunk_rates(pitcher_counts[home_starter], pitcher_bf[home_starter], league, 180.0),
                        "season_bf_before_game": int(pitcher_bf[home_starter]),
                    },
                    "observed": {
                        "top2_start_slot": top_slot,
                        "bottom2_start_slot": bottom_slot,
                    },
                }
                output.append(record)

        # Update season-to-date statistics only AFTER snapshotting this game's
        # pregame inputs. Excluded replay games still contribute to future stats.
        for row in rows:
            event = event_class(row)
            if not event:
                continue
            batter = str(row.get("batter") or "").strip()
            pitcher = str(row.get("pitcher") or "").strip()
            if batter:
                hitter_counts[batter][event] += 1
                hitter_pa[batter] += 1
            if pitcher:
                pitcher_counts[pitcher][event] += 1
                pitcher_bf[pitcher] += 1

    payload = {
        "version": "i1-state-ab-inputs-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "selection_year": args.season,
        "prior_league_source_season": prior,
        "market_inputs_used": False,
        "observed_i2_start_slot_used_as_predictor": False,
        "league_event_rates": league,
        "player_asof_rule": {
            "hitter_prior_strength_pa": 100,
            "pitcher_prior_strength_bf": 180,
            "prior_mean": "prior-season league event vector",
            "current_season_stats": "strictly before each game",
        },
        "games_total": len(games),
        "games_eligible": len(output),
        "slot_targets": slot_targets,
        "exclusions": dict(exclusions),
        "games": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()

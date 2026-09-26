#!/usr/bin/env python3
"""Build leakage-safe 2025 full-I2 replay inputs from Retrosheet.

This phase reconstructs only information that would have been available
pregame: starting batting orders, starting pitchers, handedness, venue/site,
and the eventual full-I2 outcome used only as the calibration target.

Retrosheet player IDs are mapped to MLBAM IDs through the public Chadwick
register. No fuzzy name matching is allowed.
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

RETROSHEET_ATTRIBUTION = (
    "The information used here was obtained free of charge from and is copyrighted "
    "by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., "
    "Newark, DE 19711."
)


def as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--retrosheet-zip", type=Path, required=True)
    p.add_argument("--prior-retrosheet-zip", type=Path, required=True)
    p.add_argument("--chadwick-dir", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/i2_vnext/replay_2025_inputs.json"),
    )
    return p.parse_args()


def load_chadwick(directory: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    conflicts: set[str] = set()
    files = sorted(directory.glob("people-*.csv"))
    if not files:
        raise RuntimeError(f"No Chadwick people-*.csv files found in {directory}")
    for path in files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                retro = str(row.get("key_retro") or "").strip()
                raw_mlbam = str(row.get("key_mlbam") or "").strip()
                if not retro or not raw_mlbam:
                    continue
                try:
                    mlbam = int(float(raw_mlbam))
                except ValueError:
                    continue
                if mlbam <= 0:
                    continue
                if retro in mapping and mapping[retro] != mlbam:
                    conflicts.add(retro)
                    continue
                mapping[retro] = mlbam
    for retro in conflicts:
        mapping.pop(retro, None)
    return mapping



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


def league_event_rates(zip_path: Path, season: int) -> dict[str, float]:
    counts = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        member = f"{season}plays.csv"
        if member not in zf.namelist():
            raise RuntimeError(f"{zip_path} does not contain {member}")
        with zf.open(member) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")):
                if row.get("gametype") != "regular":
                    continue
                event = event_class(row)
                if event:
                    counts[event] += 1
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError(f"No regular-season PA events found in {zip_path}")
    keys = [
        "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
        "strikeout", "ball_in_play_out",
    ]
    return {k: counts[k] / total for k in keys}



def shrunk_event_rates(
    counts: Counter,
    denom: int,
    league: dict[str, float],
    strength: float,
) -> dict[str, float]:
    d = max(0, int(denom))
    raw = {
        k: (counts[k] + league[k] * strength) / (d + strength)
        for k in league
    }
    total = sum(raw.values())
    if total <= 0:
        raise RuntimeError("Invalid shrunk I1 event-rate vector")
    return {k: raw[k] / total for k in league}


def game_sort_key(item):
    gid, rows = item
    first = rows[0]
    date = str(first.get("date") or "")
    game_num = as_int(
        first.get("number")
        or first.get("game_num")
        or first.get("dh")
        or 0
    )
    return (date, game_num, gid)


def attach_i1_asof_rates(
    game: dict,
    hitter_counts: dict[str, Counter],
    hitter_pa: Counter,
    pitcher_counts: dict[str, Counter],
    pitcher_bf: Counter,
    league: dict[str, float],
) -> None:
    for side in ("away_lineup", "home_lineup"):
        for player in game[side]:
            retro = player["retro"]
            player["i1_event_rates_asof"] = shrunk_event_rates(
                hitter_counts[retro],
                hitter_pa[retro],
                league,
                100.0,
            )
            player["i1_season_pa_before_game"] = int(hitter_pa[retro])

    for side in ("away_starter", "home_starter"):
        pitcher = game[side]
        retro = pitcher["retro"]
        pitcher["i1_event_rates_asof"] = shrunk_event_rates(
            pitcher_counts[retro],
            pitcher_bf[retro],
            league,
            180.0,
        )
        pitcher["i1_season_bf_before_game"] = int(pitcher_bf[retro])


def update_i1_asof_counts(
    rows: list[dict],
    hitter_counts: dict[str, Counter],
    hitter_pa: Counter,
    pitcher_counts: dict[str, Counter],
    pitcher_bf: Counter,
) -> None:
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


def load_plays(zip_path: Path, season: int):
    if not zip_path.exists():
        raise RuntimeError(f"Missing Retrosheet ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        member = f"{season}plays.csv"
        if member not in zf.namelist():
            raise RuntimeError(f"{zip_path} does not contain {member}")
        with zf.open(member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            games: dict[str, list[dict]] = defaultdict(list)
            bat_hands: dict[str, Counter] = defaultdict(Counter)
            pitch_hands: dict[str, Counter] = defaultdict(Counter)
            for row in reader:
                if row.get("gametype") != "regular":
                    continue
                gid = str(row.get("gid") or "").strip()
                if not gid:
                    continue
                games[gid].append(row)
                batter = str(row.get("batter") or "").strip()
                pitcher = str(row.get("pitcher") or "").strip()
                bh = str(row.get("bathand") or "").strip().upper()
                ph = str(row.get("pithand") or "").strip().upper()
                if batter and bh in {"B", "L", "R"}:
                    bat_hands[batter][bh] += 1
                if pitcher and ph in {"L", "R"}:
                    pitch_hands[pitcher][ph] += 1
    return games, bat_hands, pitch_hands


def modal(counter: Counter, allowed: set[str]) -> str | None:
    values = [(n, k) for k, n in counter.items() if k in allowed]
    if not values:
        return None
    values.sort(reverse=True)
    return values[0][1]


def lineup_from_first(row: dict) -> list[str]:
    return [str(row.get(f"l{i}") or "").strip() for i in range(1, 10)]


def side_record(retro_id: str, id_map: dict[str, int], bat_hands: dict[str, Counter]) -> dict | None:
    mlbam = id_map.get(retro_id)
    bats = modal(bat_hands.get(retro_id, Counter()), {"B", "L", "R"})
    if not mlbam or not bats:
        return None
    return {"mlbam": mlbam, "retro": retro_id, "bats": bats}


def pitcher_record(retro_id: str, id_map: dict[str, int], pitch_hands: dict[str, Counter]) -> dict | None:
    mlbam = id_map.get(retro_id)
    throws = modal(pitch_hands.get(retro_id, Counter()), {"L", "R"})
    if not mlbam or not throws:
        return None
    return {"mlbam": mlbam, "retro": retro_id, "throws": throws}


def build_game(
    gid: str,
    rows: list[dict],
    id_map: dict[str, int],
    bat_hands: dict[str, Counter],
    pitch_hands: dict[str, Counter],
) -> tuple[dict | None, str | None]:
    top1 = [r for r in rows if as_int(r.get("inning")) == 1 and as_int(r.get("top_bot")) == 0]
    bot1 = [r for r in rows if as_int(r.get("inning")) == 1 and as_int(r.get("top_bot")) == 1]
    top2 = [r for r in rows if as_int(r.get("inning")) == 2 and as_int(r.get("top_bot")) == 0]
    bot2 = [r for r in rows if as_int(r.get("inning")) == 2 and as_int(r.get("top_bot")) == 1]
    if not top1 or not bot1 or not top2 or not bot2:
        return None, "MISSING_I1_OR_I2"

    away_lineup_retro = lineup_from_first(top1[0])
    home_lineup_retro = lineup_from_first(bot1[0])
    if any(not x for x in away_lineup_retro + home_lineup_retro):
        return None, "MISSING_STARTING_LINEUP"

    away_lineup = [side_record(x, id_map, bat_hands) for x in away_lineup_retro]
    home_lineup = [side_record(x, id_map, bat_hands) for x in home_lineup_retro]
    home_starter_retro = str(top1[0].get("pitcher") or "").strip()
    away_starter_retro = str(bot1[0].get("pitcher") or "").strip()
    home_starter = pitcher_record(home_starter_retro, id_map, pitch_hands)
    away_starter = pitcher_record(away_starter_retro, id_map, pitch_hands)

    if any(x is None for x in away_lineup + home_lineup):
        return None, "UNRESOLVED_LINEUP_ID_OR_HAND"
    if home_starter is None or away_starter is None:
        return None, "UNRESOLVED_STARTER_ID_OR_HAND"

    top2_runs = sum(as_int(r.get("runs")) for r in top2)
    bot2_runs = sum(as_int(r.get("runs")) for r in bot2)
    first_top2_pitcher = str(top2[0].get("pitcher") or "").strip()
    first_bot2_pitcher = str(bot2[0].get("pitcher") or "").strip()

    return {
        "gid": gid,
        "date": str(top1[0].get("date") or ""),
        "site": str(top1[0].get("site") or ""),
        "away_team_retro": str(top1[0].get("batteam") or ""),
        "home_team_retro": str(bot1[0].get("batteam") or ""),
        "away_lineup": away_lineup,
        "home_lineup": home_lineup,
        "away_starter": away_starter,
        "home_starter": home_starter,
        "observed": {
            "top2_runs": top2_runs,
            "bottom2_runs": bot2_runs,
            "full_i2_runs": top2_runs + bot2_runs,
            "under05": int(top2_runs + bot2_runs == 0),
        },
        "audit": {
            "away_starter_began_i2": int(first_bot2_pitcher == away_starter_retro),
            "home_starter_began_i2": int(first_top2_pitcher == home_starter_retro),
            "observed_i2_pitcher_identity_used_as_predictor": False,
            "observed_i2_start_slot_used_as_predictor": False,
        },
    }, None


def main() -> None:
    args = parse_args()
    id_map = load_chadwick(args.chadwick_dir)
    prior_year = args.season - 1
    i1_rates = league_event_rates(args.prior_retrosheet_zip, prior_year)
    games, bat_hands, pitch_hands = load_plays(args.retrosheet_zip, args.season)

    output = []
    exclusions = Counter()
    continuation = []

    hitter_counts: dict[str, Counter] = defaultdict(Counter)
    pitcher_counts: dict[str, Counter] = defaultdict(Counter)
    hitter_pa: Counter = Counter()
    pitcher_bf: Counter = Counter()

    for gid, rows in sorted(games.items(), key=game_sort_key):
        game, reason = build_game(gid, rows, id_map, bat_hands, pitch_hands)
        if game is None:
            exclusions[reason or "UNKNOWN"] += 1
        else:
            # Snapshot only statistics available before this game, using the
            # exact I1 rule selected on the untouched 2024 A/B experiment.
            attach_i1_asof_rates(
                game,
                hitter_counts,
                hitter_pa,
                pitcher_counts,
                pitcher_bf,
                i1_rates,
            )
            output.append(game)
            continuation.extend([
                game["audit"]["away_starter_began_i2"],
                game["audit"]["home_starter_began_i2"],
            ])

        # Update current-season totals after the pregame snapshot. Even games
        # excluded from the full-I2 replay still contribute to subsequent
        # season-to-date statistics, matching live season-stat semantics.
        update_i1_asof_counts(
            rows,
            hitter_counts,
            hitter_pa,
            pitcher_counts,
            pitcher_bf,
        )

    if not output:
        raise RuntimeError("Replay builder produced zero eligible games")

    payload = {
        "version": "i2-vnext-replay-inputs-v2-player-asof-i1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "market_inputs_used": False,
        "prediction_inputs": "pregame starting lineups + starting pitchers + handedness + site",
        "target_only_fields": "observed full-I2 runs/under result",
        "identity_crosswalk": "Chadwick public register key_retro -> key_mlbam",
        "i1_state_model": {
            "method": (
                "existing batting-order simulator with leakage-safe current-season "
                "player event rates shrunk to prior-season league PA event vector"
            ),
            "source_season": prior_year,
            "event_rates": i1_rates,
            "player_specific_i1_talent_used": True,
            "selection_evidence": "2024 I1 start-slot A/B selected player_asof",
        },
        "i1_player_asof_model": {
            "hitter_prior_strength_pa": 100,
            "pitcher_prior_strength_bf": 180,
            "prior_mean_source_season": prior_year,
            "current_season_stats": "strictly before each game",
            "same_day_ordering": "date + Retrosheet game number/gid",
            "fallback_to_league_if_missing": False,
        },
        "games_total": len(games),
        "games_eligible": len(output),
        "games_excluded": int(sum(exclusions.values())),
        "exclusions": dict(exclusions),
        "historical_starter_i2_continuation_rate_audit_only": (
            sum(continuation) / len(continuation) if continuation else None
        ),
        "retrosheet_attribution": RETROSHEET_ATTRIBUTION,
        "games": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()

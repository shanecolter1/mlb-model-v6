#!/usr/bin/env python3
"""Build leakage-safe 2026-YTD vNext replay inputs from MLB Stats API archives.

Historical final feeds are used only to reconstruct facts that were pregame
inputs (official starting batting orders, actual starting pitchers, handedness,
venue) and later outcomes used as targets. Player event-rate snapshots are
constructed chronologically and updated only after each game's pregame snapshot.
No sportsbook/market inputs are used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


EVENT_KEYS = (
    "single", "double", "triple", "home_run", "walk", "hit_by_pitch",
    "strikeout", "ball_in_play_out",
)

OUT_EVENTS = {
    "field_out", "force_out", "grounded_into_double_play", "double_play",
    "triple_play", "fielders_choice", "fielders_choice_out", "sac_fly",
    "sac_bunt", "strikeout_double_play",
}
WALK_EVENTS = {"walk", "intent_walk", "intentional_walk"}
SINGLE_PROXY_EVENTS = {"field_error"}
WALK_PROXY_EVENTS = {"catcher_interf", "catcher_interference"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--prior-league-rates", type=Path, required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def norm_event(raw):
    e = str(raw or "").strip().lower()
    if e in {"single", "double", "triple", "home_run", "hit_by_pitch", "strikeout"}:
        return e
    if e in WALK_EVENTS:
        return "walk"
    if e in OUT_EVENTS:
        return "strikeout" if e == "strikeout_double_play" else "ball_in_play_out"
    if e in SINGLE_PROXY_EVENTS:
        return "single"
    if e in WALK_PROXY_EVENTS:
        return "walk"
    return None


def shrunk_event_rates(counts, denom, league, strength):
    d = max(0, int(denom))
    raw = {
        k: (counts[k] + league[k] * strength) / (d + strength)
        for k in EVENT_KEYS
    }
    total = sum(raw.values())
    if total <= 0:
        raise RuntimeError("Invalid shrunk event-rate vector")
    return {k: raw[k] / total for k in EVENT_KEYS}


def load_prior_rates(path):
    payload = json.loads(path.read_text())
    rates = payload.get("event_rates") if isinstance(payload, dict) else None
    if not rates:
        rates = payload
    out = {k: float(rates[k]) for k in EVENT_KEYS}
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Invalid prior league event rates")
    return {k: v / total for k, v in out.items()}


def player_meta(feed, player_id):
    players = feed.get("gameData", {}).get("players", {})
    return players.get(f"ID{int(player_id)}", {})


def side_code(meta):
    return str(meta.get("batSide", {}).get("code") or "").strip().upper()


def throw_code(meta):
    return str(meta.get("pitchHand", {}).get("code") or "").strip().upper()


def matchup_bat_side(feed, player_id):
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        matchup = play.get("matchup", {})
        if int(matchup.get("batter", {}).get("id") or -1) != int(player_id):
            continue
        code = str(matchup.get("batSide", {}).get("code") or "").strip().upper()
        if code in {"L", "R"}:
            return code
    return ""


def matchup_pitch_hand(feed, player_id):
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        matchup = play.get("matchup", {})
        if int(matchup.get("pitcher", {}).get("id") or -1) != int(player_id):
            continue
        code = str(matchup.get("pitchHand", {}).get("code") or "").strip().upper()
        if code in {"L", "R"}:
            return code
    return ""


def starting_lineup(feed, side):
    team = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    players = team.get("players", {})
    starters = []
    for rec in players.values():
        order = str(rec.get("battingOrder") or "").strip()
        person = rec.get("person", {})
        pid = person.get("id")
        if not order or pid is None:
            continue
        try:
            order_num = int(order)
        except ValueError:
            continue
        # MLB boxscore convention: starters are x00; substitutions increment suffix.
        if order_num % 100 != 0:
            continue
        slot = order_num // 100
        if 1 <= slot <= 9:
            starters.append((slot, int(pid)))
    starters.sort()
    if len(starters) == 9 and [x[0] for x in starters] == list(range(1, 10)):
        return [pid for _, pid in starters]

    # Conservative fallback: boxscore battingOrder is ordered and normally begins
    # with the nine starters. Use only if exactly nine distinct starter IDs can be
    # recovered from the first nine entries.
    order = [int(x) for x in team.get("battingOrder", []) if x is not None]
    first = []
    for pid in order:
        if pid not in first:
            first.append(pid)
        if len(first) == 9:
            break
    return first if len(first) == 9 else []


def starting_pitcher(feed, fielding_side):
    team = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(fielding_side, {})
    pitchers = [int(x) for x in team.get("pitchers", []) if x is not None]
    if pitchers:
        return pitchers[0]

    target_half = "top" if fielding_side == "home" else "bottom"
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        about = play.get("about", {})
        if int(about.get("inning") or 0) != 1:
            continue
        if str(about.get("halfInning") or "").lower() != target_half:
            continue
        pid = play.get("matchup", {}).get("pitcher", {}).get("id")
        if pid is not None:
            return int(pid)
    return None


def observed_i2(feed):
    innings = feed.get("liveData", {}).get("linescore", {}).get("innings", [])
    second = next((x for x in innings if int(x.get("num") or 0) == 2), None)
    if second is None:
        return None
    top = int(second.get("away", {}).get("runs") or 0)
    bottom = int(second.get("home", {}).get("runs") or 0)
    return {
        "top2_runs": top,
        "bottom2_runs": bottom,
        "full_i2_runs": top + bottom,
        "under05": int(top + bottom == 0),
    }


def terminal_pas(feed):
    out = []
    for play in feed.get("liveData", {}).get("plays", {}).get("allPlays", []):
        event = norm_event(play.get("result", {}).get("eventType"))
        if not event:
            continue
        matchup = play.get("matchup", {})
        batter = matchup.get("batter", {}).get("id")
        pitcher = matchup.get("pitcher", {}).get("id")
        if batter is None or pitcher is None:
            continue
        out.append((int(batter), int(pitcher), event))
    return out


def build_player(pid, feed, hitter_counts, hitter_pa, league):
    meta = player_meta(feed, pid)
    bats = side_code(meta)
    if bats not in {"L", "R", "B"}:
        # Final-feed fallback recovers the effective batting side actually used
        # against the game's pitching. This is a matchup attribute, not an outcome.
        bats = matchup_bat_side(feed, pid)
    if bats not in {"L", "R", "B"}:
        return None
    return {
        "mlbam": int(pid),
        "bats": bats,
        "i1_event_rates_asof": shrunk_event_rates(
            hitter_counts[int(pid)], hitter_pa[int(pid)], league, 100.0
        ),
        "i1_season_pa_before_game": int(hitter_pa[int(pid)]),
    }


def build_pitcher(pid, feed, pitcher_counts, pitcher_bf, league):
    if pid is None:
        return None
    meta = player_meta(feed, pid)
    throws = throw_code(meta)
    if throws not in {"L", "R"}:
        throws = matchup_pitch_hand(feed, pid)
    if throws not in {"L", "R"}:
        return None
    return {
        "mlbam": int(pid),
        "throws": throws,
        "i1_event_rates_asof": shrunk_event_rates(
            pitcher_counts[int(pid)], pitcher_bf[int(pid)], league, 180.0
        ),
        "i1_season_bf_before_game": int(pitcher_bf[int(pid)]),
    }


def game_sort_key(rec):
    return (
        str(rec.get("officialDate") or rec.get("gameDate") or ""),
        int(rec.get("gameNumber") or 1),
        int(rec.get("gamePk") or 0),
    )


def main():
    args = parse_args()
    manifest = json.loads((args.raw_dir / "fetch_manifest.json").read_text())
    league = load_prior_rates(args.prior_league_rates)

    records = []
    for rec in manifest.get("games", []):
        feed_path = Path(rec["feed_path"])
        if not feed_path.is_absolute():
            feed_path = Path.cwd() / feed_path
        if not feed_path.exists():
            continue
        feed = json.loads(feed_path.read_text())
        gd = feed.get("gameData", {})
        if int(gd.get("game", {}).get("season") or args.season) != args.season:
            continue
        status = str(gd.get("status", {}).get("abstractGameState") or gd.get("status", {}).get("detailedState") or "")
        if "final" not in status.lower():
            continue
        records.append({
            "gamePk": int(gd.get("game", {}).get("pk") or rec["game_id"]),
            "officialDate": gd.get("datetime", {}).get("officialDate") or rec.get("game_date"),
            "gameNumber": gd.get("game", {}).get("gameNumber") or 1,
            "feed": feed,
        })
    records.sort(key=game_sort_key)

    hitter_counts = defaultdict(Counter)
    hitter_pa = Counter()
    pitcher_counts = defaultdict(Counter)
    pitcher_bf = Counter()
    output = []
    exclusions = Counter()

    for rec in records:
        feed = rec["feed"]
        away_ids = starting_lineup(feed, "away")
        home_ids = starting_lineup(feed, "home")
        home_sp = starting_pitcher(feed, "home")
        away_sp = starting_pitcher(feed, "away")
        obs = observed_i2(feed)

        reason = None
        if len(away_ids) != 9 or len(home_ids) != 9:
            reason = "MISSING_STARTING_LINEUP"
        elif home_sp is None or away_sp is None:
            reason = "MISSING_STARTER"
        elif obs is None:
            reason = "MISSING_I2"

        game = None
        if reason is None:
            away_lineup = [
                build_player(pid, feed, hitter_counts, hitter_pa, league)
                for pid in away_ids
            ]
            home_lineup = [
                build_player(pid, feed, hitter_counts, hitter_pa, league)
                for pid in home_ids
            ]
            home_starter = build_pitcher(
                home_sp, feed, pitcher_counts, pitcher_bf, league
            )
            away_starter = build_pitcher(
                away_sp, feed, pitcher_counts, pitcher_bf, league
            )
            if any(x is None for x in away_lineup + home_lineup):
                reason = "MISSING_BAT_SIDE"
            elif home_starter is None or away_starter is None:
                reason = "MISSING_PITCH_HAND"
            else:
                gd = feed["gameData"]
                game = {
                    "gid": str(rec["gamePk"]),
                    "date": str(rec["officialDate"]),
                    "venue_id": gd.get("venue", {}).get("id"),
                    "venue_name": gd.get("venue", {}).get("name"),
                    "away_team_id": gd.get("teams", {}).get("away", {}).get("id"),
                    "away_team_name": gd.get("teams", {}).get("away", {}).get("name"),
                    "home_team_id": gd.get("teams", {}).get("home", {}).get("id"),
                    "home_team_name": gd.get("teams", {}).get("home", {}).get("name"),
                    "away_lineup": away_lineup,
                    "home_lineup": home_lineup,
                    "away_starter": away_starter,
                    "home_starter": home_starter,
                    "observed": obs,
                    "audit": {
                        "source": "MLB Stats API archived final feed",
                        "historical_final_feed_used_only_to_reconstruct_pregame_starting_personnel_and_target": True,
                        "observed_i2_pitcher_identity_used_as_predictor": False,
                        "observed_i2_start_slot_used_as_predictor": False,
                    },
                }

        if game is None:
            exclusions[reason or "UNKNOWN"] += 1
        else:
            output.append(game)

        # Update only after the pregame snapshot has been created.
        for batter, pitcher, event in terminal_pas(feed):
            hitter_counts[batter][event] += 1
            hitter_pa[batter] += 1
            pitcher_counts[pitcher][event] += 1
            pitcher_bf[pitcher] += 1

    if not output:
        raise RuntimeError("2026 replay builder produced zero games")

    payload = {
        "version": "i2-vnext-replay-inputs-mlbstats-v1-player-asof-i1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "market_inputs_used": False,
        "prediction_inputs": "official starting batting orders + starting pitchers + handedness + venue",
        "target_only_fields": "observed I2 runs/under result",
        "i1_state_model": {
            "method": (
                "existing batting-order simulator with leakage-safe current-season "
                "player event rates shrunk to prior-season league PA event vector"
            ),
            "source_season": args.season - 1,
            "event_rates": league,
            "player_specific_i1_talent_used": True,
        },
        "i1_player_asof_model": {
            "hitter_prior_strength_pa": 100,
            "pitcher_prior_strength_bf": 180,
            "prior_mean_source_season": args.season - 1,
            "current_season_stats": "strictly before each game",
            "same_day_ordering": "officialDate + gameNumber + gamePk",
            "fallback_to_league_if_missing": False,
        },
        "games_total": len(records),
        "games_eligible": len(output),
        "games_excluded": int(sum(exclusions.values())),
        "exclusions": dict(exclusions),
        "games": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps({k: v for k, v in payload.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build season-specific primary home-venue specifications from replay inputs.

This is a classification artifact, not a park-factor model. A team's primary
home venue is its modal home site among replay-eligible regular-season games.
Physical park-factor coverage is intentionally kept separate.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def team_key(game):
    for key in ("home_team_retro","home_team_id","home_team_name"):
        v=game.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    raise ValueError(f"Missing home-team identity for {game.get('gid')}")


def site_key(game):
    for key in ("site","venue_id","venue_name"):
        v=game.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    raise ValueError(f"Missing home venue/site for {game.get('gid')}")


def main():
    a=parse_args()
    payload=json.loads(a.inputs.read_text())
    if payload.get("market_inputs_used") is not False:
        raise ValueError("Replay inputs are not market-isolated")
    season=int(payload.get("season") or 0)
    games=payload.get("games") or []
    if season <= 0 or not games:
        raise ValueError("Invalid replay input")

    counts=defaultdict(Counter)
    for g in games:
        counts[team_key(g)][site_key(g)] += 1

    teams={}
    for team,sites in sorted(counts.items()):
        ordered=sorted(sites.items(),key=lambda kv:(-kv[1],kv[0]))
        primary,n_primary=ordered[0]
        total=sum(sites.values())
        teams[team]={
            "primary_site":primary,
            "primary_site_games":int(n_primary),
            "home_games_in_replay":int(total),
            "primary_site_share":n_primary/total,
            "site_counts":{site:int(n) for site,n in ordered},
            "alternate_site_games":int(total-n_primary),
        }

    game_rows={}
    primary_games=0
    for g in games:
        t=team_key(g); s=site_key(g)
        spec=teams[t]
        is_primary=s==spec["primary_site"]
        primary_games += int(is_primary)
        game_rows[str(g["gid"])]={
            "home_team_key":t,
            "site":s,
            "status":"PRIMARY_HOME_VENUE" if is_primary else "ALTERNATE_OR_NEUTRAL_HOME_SITE",
            "primary_site":spec["primary_site"],
            "primary_site_share":spec["primary_site_share"],
        }

    result={
        "version":"i2-season-primary-home-venue-spec-v1",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "season":season,
        "market_inputs_used":False,
        "definition":"Primary home venue is the modal home site for each team among replay-eligible regular-season games. This classification is independent of Savant park-factor availability.",
        "teams":teams,
        "games":game_rows,
        "summary":{
            "teams":len(teams),
            "games":len(games),
            "primary_home_venue_games":primary_games,
            "alternate_or_neutral_games":len(games)-primary_games,
            "primary_home_venue_share":primary_games/len(games),
        },
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":
    main()

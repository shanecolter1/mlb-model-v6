#!/usr/bin/env python3
"""Merge deterministic I2 vNext replay shards into one governed replay artifact."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=Path,required=True)
    p.add_argument("--expected-shards",type=int,required=True)
    p.add_argument("--expected-games",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    return p.parse_args()


def logloss(y,p):
    q=max(1e-9,min(1-1e-9,float(p)))
    return -(y*math.log(q)+(1-y)*math.log(1-q))


def main():
    args=parse_args()
    files=sorted(args.input_dir.glob("replay_*_shard_*.json"))
    if len(files)!=args.expected_shards:
        raise RuntimeError(f"Expected {args.expected_shards} shard files, found {len(files)}")

    shards=[json.loads(p.read_text()) for p in files]
    indices=sorted(int(s["shard_index"]) for s in shards)
    if indices!=list(range(args.expected_shards)):
        raise RuntimeError(f"Shard indices mismatch: {indices}")

    first=shards[0]
    invariant_keys=[
        "season","base_model_version","trials_per_game","market_inputs_used",
        "observed_i2_state_used_as_predictor","point_in_time_player_refits",
        "i1_state_mode","i1_environment","transition_model","i2_model",
    ]
    for s in shards[1:]:
        for key in invariant_keys:
            if s.get(key)!=first.get(key):
                raise RuntimeError(f"Shard invariant mismatch for {key}")

    predictions=[]
    park_matched_games=0.0
    for s in shards:
        if int(s.get("shard_count",0))!=args.expected_shards:
            raise RuntimeError("Shard-count metadata mismatch")
        if int(s.get("replay_games_total",0))!=args.expected_games:
            raise RuntimeError("Replay total metadata mismatch")
        predictions.extend(s["predictions"])
        park_matched_games += float(s.get("park_match_rate") or 0) * int(s["n"])

    if len(predictions)!=args.expected_games:
        raise RuntimeError(f"Expected {args.expected_games} merged games, found {len(predictions)}")
    gids=[str(p["gid"]) for p in predictions]
    if len(set(gids))!=len(gids):
        raise RuntimeError("Duplicate game IDs across replay shards")

    predictions.sort(key=lambda p:(str(p.get("date","")),str(p["gid"])))
    brier=sum((float(p["raw_under05"])-int(p["observed_under05"]))**2 for p in predictions)/len(predictions)
    ll=sum(logloss(int(p["observed_under05"]),float(p["raw_under05"])) for p in predictions)/len(predictions)

    out=dict(first)
    out["version"]="i2-vnext-full-replay-v4-player-asof-i1-sharded"
    out["n"]=len(predictions)
    out["raw_brier"]=brier
    out["raw_logloss"]=ll
    out["park_match_rate"]=park_matched_games/len(predictions)
    out["replay_games_total"]=args.expected_games
    out["shard_count"]=args.expected_shards
    out["shard_index"]=None
    out["shard_games"]=None
    out["shard_merge"]={
        "deterministic":True,
        "shards":args.expected_shards,
        "game_assignment":"original replay index modulo shard_count",
        "per_game_seed_unchanged":True,
    }
    out["predictions"]=predictions

    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,separators=(",",":")))
    print(json.dumps({
        "n":out["n"],
        "raw_brier":out["raw_brier"],
        "raw_logloss":out["raw_logloss"],
        "park_match_rate":out["park_match_rate"],
        "trials_per_game":out["trials_per_game"],
        "shards":args.expected_shards,
        "i1_state_mode":out["i1_state_mode"],
    },indent=2))


if __name__=="__main__":
    main()

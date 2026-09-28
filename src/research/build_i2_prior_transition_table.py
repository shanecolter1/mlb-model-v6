#!/usr/bin/env python3
"""Build a target-season PA transition table from prior seasons only.

Research replay utility. It reuses the already-selected production shrinkage
constants but excludes target-season and later empirical transition counts.
This prevents direct target-outcome leakage in historical replays.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

EVENTS = ("out", "bb", "single", "double", "triple", "hr")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--target-season", type=int, required=True)
    p.add_argument("--first-season", type=int, default=2021)
    p.add_argument("--k-exact", type=float, default=5.0)
    p.add_argument("--k-parent", type=float, default=1280.0)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_counts(path: Path):
    payload = json.loads(path.read_text())
    out = defaultdict(Counter)
    for key, rows in payload["states"].items():
        ev, outs, mask = key.split("|")
        mapped = "out" if ev in {"strikeout", "ball_in_play_out"} else ev
        if mapped not in EVENTS:
            continue
        state = (mapped, int(outs), int(mask))
        for row in rows:
            outcome = (
                int(row["outs_added"]),
                int(row["post_mask"]),
                int(row["runs"]),
            )
            out[state][outcome] += int(row["n"])
    return out


def merge(parts):
    out = defaultdict(Counter)
    for part in parts:
        for key, counts in part.items():
            out[key].update(counts)
    return out


def parents(counts):
    event_outs = defaultdict(Counter)
    event = defaultdict(Counter)
    for (ev, outs, _mask), child in counts.items():
        event_outs[(ev, outs)].update(child)
        event[ev].update(child)
    return event_outs, event


def norm(counter):
    n = sum(counter.values())
    return {k: v / n for k, v in counter.items()} if n else {}


def shrunk_dist(key, child, event_outs, event, k_exact, k_parent):
    ev, outs, _ = key
    grand = norm(event.get(ev, Counter()))
    parent_counts = event_outs.get((ev, outs), Counter())
    parent_n = sum(parent_counts.values())
    outcomes = set(grand) | set(parent_counts) | set(child)
    if not outcomes:
        return {}

    parent = {}
    for outcome in outcomes:
        gp = grand.get(outcome, 0.0)
        parent[outcome] = (
            (parent_counts.get(outcome, 0) + k_parent * gp) / (parent_n + k_parent)
            if parent_n + k_parent
            else gp
        )

    child_n = sum(child.values())
    dist = {
        outcome: (
            (child.get(outcome, 0) + k_exact * parent.get(outcome, 0.0))
            / (child_n + k_exact)
            if child_n + k_exact
            else parent.get(outcome, 0.0)
        )
        for outcome in outcomes
    }
    total = sum(dist.values())
    return {outcome: p / total for outcome, p in dist.items()} if total else dist


def main():
    args = parse_args()
    years = list(range(args.first_season, args.target_season))
    if not years:
        raise ValueError("No prior transition season is available for target")

    parts = []
    for year in years:
        path = args.source_dir / f"transitions_{year}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(load_counts(path))

    counts = merge(parts)
    event_outs, event = parents(counts)
    states = {}
    samples = {}

    for ev in EVENTS:
        for outs in (0, 1, 2):
            for mask in range(8):
                key = (ev, outs, mask)
                child = counts.get(key, Counter())
                dist = shrunk_dist(
                    key,
                    child,
                    event_outs,
                    event,
                    args.k_exact,
                    args.k_parent,
                )
                if not dist:
                    raise ValueError(f"No prior transition support for {key}")
                states[f"{ev}|{outs}|{mask}"] = [
                    {
                        "outs_added": outcome[0],
                        "post_mask": outcome[1],
                        "runs": outcome[2],
                        "p": p,
                    }
                    for outcome, p in sorted(dist.items())
                ]
                samples[f"{ev}|{outs}|{mask}"] = int(sum(child.values()))

    result = {
        "version": f"historical-prior-only-pa-transitions-before-{args.target_season}-v1",
        "market_inputs_used": False,
        "training_years": years,
        "target_season": args.target_season,
        "target_or_later_counts_used": False,
        "shrinkage": {
            "k_exact_to_event_outs": args.k_exact,
            "k_event_outs_to_event": args.k_parent,
            "source": (
                "Frozen governed shrinkage constants; not reselected inside "
                "historical replay"
            ),
        },
        "state_definition": "model_event|outs_before|base_mask_before",
        "states": states,
        "raw_state_sample_sizes": samples,
        "generic_out": "prior-season observed-count pool of strikeout + ball_in_play_out",
        "hbp_status": "excluded until separately modeled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")))
    print(json.dumps({
        "target_season": args.target_season,
        "training_years": years,
        "states": len(states),
        "market_inputs_used": False,
    }, indent=2))


if __name__ == "__main__":
    main()

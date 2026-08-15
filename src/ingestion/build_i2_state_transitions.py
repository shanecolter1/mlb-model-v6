#!/usr/bin/env python3
"""Build leakage-safe I1->I2 state-transition datasets from Retrosheet CSV ZIPs.

Expected input archives: YEARcsvs.zip containing YEARplays.csv.
Outputs compact per-season state-calibration datasets plus aggregate summaries.

Retrosheet attribution is required for redistributed derived data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

FULL_FIELDS = [
    "season", "gid", "date", "half", "batting_team", "pitching_team",
    "i1_pa", "i1_runs", "i1_pitches", "i2_start_slot", "i2_first_batter",
    "i1_pitcher", "i2_pitcher", "same_pitcher_i2", "i2_pa", "i2_runs",
    "i2_pitches", "i2_scored", "i2_2plus", "i2_3plus", "i2_4plus",
    "i2_exact_bucket",
]

COMPACT_FIELDS = [
    "season", "gid", "date", "half", "i1_pa", "i1_runs", "i1_pitches",
    "i2_start_slot", "same_pitcher_i2", "i2_pa", "i2_runs", "i2_pitches",
]

RETROSHEET_ATTRIBUTION = (
    "The information used here was obtained free of charge from and is copyrighted "
    "by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711."
)


def as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def pitch_bin(pitches: int) -> str:
    if pitches <= 12:
        return "0-12"
    if pitches <= 17:
        return "13-17"
    if pitches <= 22:
        return "18-22"
    if pitches <= 27:
        return "23-27"
    return "28+"


def iter_rows(zip_path: Path):
    year = int(zip_path.name[:4])
    with zipfile.ZipFile(zip_path) as zf:
        plays_name = f"{year}plays.csv"
        with zf.open(plays_name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            by_side = defaultdict(lambda: {1: [], 2: []})
            for row in reader:
                if row.get("gametype") != "regular":
                    continue
                inning = as_int(row.get("inning"))
                if inning not in (1, 2):
                    continue
                top_bot = as_int(row.get("top_bot"))
                by_side[(row["gid"], top_bot)][inning].append(row)

    for (gid, top_bot), innings in by_side.items():
        i1, i2 = innings[1], innings[2]
        if not i1 or not i2:
            continue
        first1, first2 = i1[0], i2[0]
        lineup = [first1.get(f"l{i}") for i in range(1, 10)]
        slot_by_batter = {b: i + 1 for i, b in enumerate(lineup) if b}
        i2_first = first2.get("batter")
        i2_slot = slot_by_batter.get(i2_first) or as_int(first2.get("lp"))
        i1_pitcher = first1.get("pitcher")
        i2_pitcher = first2.get("pitcher")
        i2_runs = sum(as_int(r.get("runs")) for r in i2)

        yield {
            "season": year,
            "gid": gid,
            "date": first1.get("date"),
            "half": "top" if top_bot == 0 else "bottom",
            "batting_team": first1.get("batteam"),
            "pitching_team": first1.get("pitteam"),
            "i1_pa": sum(as_int(r.get("pa")) for r in i1),
            "i1_runs": sum(as_int(r.get("runs")) for r in i1),
            "i1_pitches": sum(as_int(r.get("nump")) for r in i1),
            "i2_start_slot": i2_slot,
            "i2_first_batter": i2_first,
            "i1_pitcher": i1_pitcher,
            "i2_pitcher": i2_pitcher,
            "same_pitcher_i2": int(bool(i1_pitcher and i2_pitcher and i1_pitcher == i2_pitcher)),
            "i2_pa": sum(as_int(r.get("pa")) for r in i2),
            "i2_runs": i2_runs,
            "i2_pitches": sum(as_int(r.get("nump")) for r in i2),
            "i2_scored": int(i2_runs >= 1),
            "i2_2plus": int(i2_runs >= 2),
            "i2_3plus": int(i2_runs >= 3),
            "i2_4plus": int(i2_runs >= 4),
            "i2_exact_bucket": str(i2_runs) if i2_runs < 4 else "4+",
        }


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_start_slot_summary(path: Path, rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["i2_start_slot"]].append(row)
    fields = ["i2_start_slot", "n", "p_score", "mean_runs", "mean_i1_pitches", "p_same_pitcher"]
    output = []
    for slot in sorted(grouped):
        group = grouped[slot]
        output.append({
            "i2_start_slot": slot,
            "n": len(group),
            "p_score": sum(r["i2_scored"] for r in group) / len(group),
            "mean_runs": statistics.fmean(r["i2_runs"] for r in group),
            "mean_i1_pitches": statistics.fmean(r["i1_pitches"] for r in group),
            "p_same_pitcher": sum(r["same_pitcher_i2"] for r in group) / len(group),
        })
    write_csv(path, output, fields)


def write_state_benchmark(path: Path, rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["i2_start_slot"], pitch_bin(row["i1_pitches"]), row["same_pitcher_i2"])
        grouped[key].append(row)
    fields = [
        "i2_start_slot", "i1_pitch_bin", "same_pitcher_i2", "n", "p0", "p1", "p2",
        "p3", "p4plus", "p1plus", "mean_runs",
    ]
    output = []
    for key in sorted(grouped, key=lambda x: (x[0], ["0-12", "13-17", "18-22", "23-27", "28+"].index(x[1]), x[2])):
        slot, pbin, same = key
        group = grouped[key]
        n = len(group)
        output.append({
            "i2_start_slot": slot,
            "i1_pitch_bin": pbin,
            "same_pitcher_i2": same,
            "n": n,
            "p0": sum(r["i2_runs"] == 0 for r in group) / n,
            "p1": sum(r["i2_runs"] == 1 for r in group) / n,
            "p2": sum(r["i2_runs"] == 2 for r in group) / n,
            "p3": sum(r["i2_runs"] == 3 for r in group) / n,
            "p4plus": sum(r["i2_runs"] >= 4 for r in group) / n,
            "p1plus": sum(r["i2_runs"] >= 1 for r in group) / n,
            "mean_runs": statistics.fmean(r["i2_runs"] for r in group),
        })
    write_csv(path, output, fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    zips = sorted(input_dir.glob("20??csvs*.zip"))
    if not zips:
        raise SystemExit("No YEARcsvs*.zip files found")

    rows = []
    for zip_path in zips:
        rows.extend(iter_rows(zip_path))

    if len(rows) != 24296:
        raise SystemExit(f"Expected 24,296 team/game observations for 2021-2025; got {len(rows):,}")
    if len({r['gid'] for r in rows}) != 12148:
        raise SystemExit("Expected 12,148 unique regular-season games")
    if any(not (1 <= int(r["i2_start_slot"]) <= 9) for r in rows):
        raise SystemExit("Invalid I2 starting slot detected")

    output_dir.mkdir(parents=True, exist_ok=True)
    for year in sorted({r["season"] for r in rows}):
        year_rows = [r for r in rows if r["season"] == year]
        write_csv(output_dir / f"i2_state_compact_{year}.csv", year_rows, COMPACT_FIELDS)

    write_start_slot_summary(output_dir / "i2_start_slot_summary.csv", rows)
    write_state_benchmark(output_dir / "i2_state_benchmark.csv", rows)

    manifest = {
        "dataset": "I2 state-transition compact training data",
        "version": "2026-08-15-v1",
        "seasons": [2021, 2022, 2023, 2024, 2025],
        "regular_season_games": 12148,
        "team_game_observations": len(rows),
        "i2_start_slot_missing": 0,
        "same_pitcher_i2_rate": sum(r["same_pitcher_i2"] for r in rows) / len(rows),
        "half_i2_score_rate": sum(r["i2_scored"] for r in rows) / len(rows),
        "half_i2_mean_runs": statistics.fmean(r["i2_runs"] for r in rows),
        "start_slot_counts": dict(sorted(Counter(r["i2_start_slot"] for r in rows).items())),
        "source": "Retrosheet 2021-2025 season CSV packages (plays.csv)",
        "derived_fields_note": "Observed I1 state is for training/calibration only; pregame model must simulate I1.",
        "retrosheet_attribution": RETROSHEET_ATTRIBUTION,
    }
    with (output_dir / "i2_state_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

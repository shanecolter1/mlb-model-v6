#!/usr/bin/env python3
"""Archive pregame vNext shadow forecasts and score later official MLB outcomes.

The archive and selection policy use only pregame information. Outcome feeds
are read only by the scoring command, after the forecast is fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Forecast timestamps must specify a timezone")
    return parsed.astimezone(timezone.utc)


def validated_snapshot(data: dict) -> datetime:
    if data.get("marketDataUsed") is not False:
        raise ValueError("Snapshot is not market-isolated")
    if data.get("promotionStatus") != "SHADOW_ONLY_PROSPECTIVE_VALIDATION_REQUIRED":
        raise ValueError("Snapshot is not from the vNext shadow runner")
    generated = timestamp(data["generatedAt"])
    if timestamp(data["cutoff"]) > generated:
        raise ValueError("Forecast cutoff occurs after generation")
    return generated


def validation_cohort(data: dict) -> dict:
    half = data.get("halfCalibration") or {}
    final = data.get("finalCalibration") or {}
    return {
        "prospective_validation_start": data.get("prospectiveValidationStart"),
        "half_calibration_version": half.get("version"),
        "half_calibration_type": half.get("type"),
        "half_contrast_h": half.get("h"),
        "final_calibration_type": final.get("type") or final.get("method") or "identity",
    }


def cohort_key(cohort: dict) -> str:
    return json.dumps(cohort, sort_keys=True, separators=(",", ":"))


def archive(snapshot: Path, directory: Path) -> dict:
    contents = snapshot.read_bytes()
    data = json.loads(contents)
    generated = validated_snapshot(data)
    digest = hashlib.sha256(contents).hexdigest()
    date = str(data["date"])
    # A Central-time evening run can target the next UTC date, so this is
    # intentionally only a well-formed date check.
    datetime.fromisoformat(date)
    if not isinstance(data.get("games"), list):
        raise ValueError("Missing game rows")
    half = data.get("halfCalibration") or None
    for game in data["games"]:
        if game.get("modelStatus") == "FROZEN_VNEXT_SHADOW_PROJECTION":
            if not generated < timestamp(game["gameDate"]):
                raise ValueError(f"Forecast generated after first pitch: {game['gamePk']}")
            if game.get("bettingEligibility", {}).get("eligible") is not False:
                raise ValueError("Shadow forecast unexpectedly betting eligible")
            if half:
                matched = bool((game.get("dataAudit") or {}).get("venueProfileMatched"))
                applied = bool(game.get("halfContrastApplied"))
                if half.get("matchedHomeVenueOnly") is True and applied != matched:
                    raise ValueError(f"Half-contrast venue scope mismatch: {game['gamePk']}")
                for field in (
                    "rawTop2ScoreProbability", "rawBottom2ScoreProbability",
                    "top2ScoreProbability", "bottom2ScoreProbability",
                    "halfAdjustedUnder05",
                ):
                    if field not in game:
                        raise ValueError(f"Missing half-contrast audit field {field}: {game['gamePk']}")
    filename = f"{generated.strftime('%Y%m%dT%H%M%S%fZ')}_{digest}.json"
    target = directory / date / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(contents)
    except FileExistsError:
        if target.read_bytes() != contents:
            raise ValueError("Archive filename collision with changed content")
    return {"path": str(target), "sha256": digest, "game_rows": len(data["games"]), "generated_at": data["generatedAt"]}


def candidates(directory: Path) -> tuple[dict[int, list[dict]], int]:
    choices: dict[int, list[dict]] = defaultdict(list)
    snapshots = 0
    for file in sorted(directory.rglob("*.json")):
        contents = file.read_bytes()
        data = json.loads(contents)
        generated = validated_snapshot(data)
        digest = hashlib.sha256(contents).hexdigest()
        snapshots += 1
        cohort = validation_cohort(data)
        cohort_id = cohort_key(cohort)
        for game in data.get("games", []):
            if game.get("modelStatus") != "FROZEN_VNEXT_SHADOW_PROJECTION":
                continue
            start = timestamp(game["gameDate"])
            if not generated < start:
                raise ValueError(f"Post-first-pitch projection in archive: {file}")
            if game.get("bettingEligibility", {}).get("eligible") is not False:
                raise ValueError(f"Betting eligible shadow projection in archive: {file}")
            check = game.get("inputAudit", {}).get("freezeCheck", {})
            if not check.get("checkedAt") or check.get("gate", {}).get("requiresCleanRerun") is not False:
                continue
            if timestamp(check["checkedAt"]) >= start:
                continue
            p = float(game["under05"])
            if not math.isfinite(p) or not (0 < p < 1):
                raise ValueError(f"Invalid Under probability in {file}")
            if abs(p + float(game["over05"]) - 1) > 1e-9:
                raise ValueError(f"Under/Over mismatch in {file}")
            def half_under(field: str) -> float:
                score_pct = float(game[field])
                if not math.isfinite(score_pct) or not 0 <= score_pct <= 100:
                    raise ValueError(f"Invalid {field} in {file}")
                return min(1 - 1e-9, max(1e-9, 1 - score_pct / 100))

            def exact_half_under(probability_field: str, pct_field: str, fallback: float) -> float:
                if game.get(probability_field) is not None:
                    score_p = float(game[probability_field])
                    if not math.isfinite(score_p) or not 0 < score_p < 1:
                        raise ValueError(f"Invalid {probability_field} in {file}")
                    return 1 - score_p
                if game.get(pct_field) is not None:
                    return half_under(pct_field)
                return fallback

            if data.get("prospectiveValidationStart") and generated.date() < timestamp(data["prospectiveValidationStart"] + "T00:00:00Z").date():
                continue
            adjusted_top_under = exact_half_under("top2ScoreProbability", "top2ScorePct", half_under("top2ScorePct"))
            adjusted_bottom_under = exact_half_under("bottom2ScoreProbability", "bottom2ScorePct", half_under("bottom2ScorePct"))
            raw_top_under = exact_half_under("rawTop2ScoreProbability", "rawTop2ScorePct", adjusted_top_under)
            raw_bottom_under = exact_half_under("rawBottom2ScoreProbability", "rawBottom2ScorePct", adjusted_bottom_under)
            raw_under = float(game.get("rawUnder05", p))
            half_adjusted_under = float(game.get("halfAdjustedUnder05", p))
            for label, value in (("rawUnder05", raw_under), ("halfAdjustedUnder05", half_adjusted_under)):
                if not math.isfinite(value) or not 0 < value < 1:
                    raise ValueError(f"Invalid {label} in {file}")
            choices[int(game["gamePk"])].append({
                "game_pk": int(game["gamePk"]),
                "game_date": game["gameDate"],
                "date": start.date().isoformat(),
                "generated_at": data["generatedAt"],
                "generated_timestamp": generated,
                "prediction_class": game.get("predictionClass"),
                "venue_status": game.get("venueStatus"),
                "p_under": p,
                "p_raw_under": raw_under,
                "p_half_adjusted_under": half_adjusted_under,
                "p_top_under": adjusted_top_under,
                "p_bottom_under": adjusted_bottom_under,
                "p_raw_top_under": raw_top_under,
                "p_raw_bottom_under": raw_bottom_under,
                "half_contrast_applied": bool(game.get("halfContrastApplied")),
                "half_contrast_h": game.get("halfContrastH"),
                "validation_cohort": cohort,
                "validation_cohort_key": cohort_id,
                "snapshot_sha256": digest,
            })
    return choices, snapshots


def prior_season_under(csv_path: Path, target_season: int) -> tuple[float, int]:
    games = defaultdict(dict)
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if int(row["season"]) != target_season - 1:
                raise ValueError("Baseline file is not the preceding season")
            key = row["gid"]
            if row["half"] in games[key]:
                raise ValueError("Duplicate prior-season half inning")
            games[key][row["half"]] = int(row["i2_runs"])
    if any(set(pair) != {"top", "bottom"} for pair in games.values()):
        raise ValueError("Incomplete prior-season baseline games")
    if not games:
        raise ValueError("No prior-season baseline games")
    return sum(int(not pair["top"] and not pair["bottom"]) for pair in games.values()) / len(games), len(games)


def official_outcome(feed: dict, game_pk: int) -> tuple[int, int] | None:
    feed_pk = feed.get("gamePk") or feed.get("gameData", {}).get("game", {}).get("pk")
    if int(feed_pk or 0) != game_pk:
        raise ValueError(f"Outcome feed game ID mismatch: {game_pk}")
    state = feed.get("gameData", {}).get("status", {}).get("abstractGameState")
    if state != "Final":
        return None
    innings = [row for row in feed.get("liveData", {}).get("linescore", {}).get("innings", []) if int(row.get("num", 0)) == 2]
    if len(innings) != 1:
        raise ValueError(f"Missing or duplicated final second inning: {game_pk}")
    top = innings[0].get("away", {}).get("runs")
    bottom = innings[0].get("home", {}).get("runs")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (top, bottom)):
        raise ValueError(f"Invalid final second-inning runs: {game_pk}")
    return top, bottom


def fetch_final_feeds(archive_dir: Path, outcome_dir: Path, open_url=urlopen) -> dict:
    groups, _ = candidates(archive_dir)
    outcome_dir.mkdir(parents=True, exist_ok=True)
    fetched, pending, existing = [], [], []
    for game_pk in sorted(groups):
        target = outcome_dir / f"{game_pk}.json"
        if target.exists():
            if official_outcome(json.loads(target.read_text()), game_pk) is None:
                raise ValueError(f"Saved MLB feed is not final: {target}")
            existing.append(game_pk)
            continue
        request = Request(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live",
            headers={"Accept": "application/json", "User-Agent": "MLB-I2-vNext-Research/1.0"},
        )
        with open_url(request, timeout=30) as response:
            payload = response.read()
        feed = json.loads(payload)
        if official_outcome(feed, game_pk) is None:
            pending.append(game_pk)
            continue
        with target.open("xb") as stream:
            stream.write(payload)
        fetched.append(game_pk)
    return {"source": "MLB Stats API final game feed", "fetched_final": fetched, "pending_not_final": pending, "already_saved": existing}


def metrics(rows: list[dict], key: str, target: str) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    pairs = sorted((r[key], r[target]) for r in rows)
    positives = sum(y for _, y in pairs)
    negatives = n - positives
    auc = None
    if positives and negatives:
        lower_negatives = 0
        favorable = 0.0
        i = 0
        while i < n:
            j = i + 1
            while j < n and pairs[j][0] == pairs[i][0]:
                j += 1
            group_positives = sum(y for _, y in pairs[i:j])
            group_negatives = j - i - group_positives
            favorable += group_positives * (lower_negatives + group_negatives / 2)
            lower_negatives += group_negatives
            i = j
        auc = favorable / (positives * negatives)
    calibration = []
    for bucket in range(10):
        selected = [r for r in rows if min(int(r[key] * 10), 9) == bucket]
        if selected:
            calibration.append({
                "range": [bucket / 10, (bucket + 1) / 10],
                "n": len(selected),
                "mean_probability": sum(r[key] for r in selected) / len(selected),
                "realized_rate": sum(r[target] for r in selected) / len(selected),
            })
    return {
        "n": n,
        "mean_probability": sum(r[key] for r in rows) / n,
        "realized_rate": sum(r[target] for r in rows) / n,
        "brier": sum((r[key] - r[target]) ** 2 for r in rows) / n,
        "logloss": sum(-r[target] * math.log(r[key]) - (1 - r[target]) * math.log1p(-r[key]) for r in rows) / n,
        "auc": auc,
        "calibration_bins": calibration,
    }


def paired_raw_adjusted_delta(
    rows: list[dict],
    adjusted_keys: tuple[str, ...],
    raw_keys: tuple[str, ...],
    targets: tuple[str, ...],
    draws: int,
) -> dict:
    if not rows:
        return {"n": 0, "brier": None, "logloss": None, "ci95_date_cluster": None}
    if not (len(adjusted_keys) == len(raw_keys) == len(targets)):
        raise ValueError("Paired metric key mismatch")
    brier_delta = 0.0
    logloss_delta = 0.0
    observations = 0
    days = defaultdict(lambda: [0, 0.0, 0.0])
    for row in rows:
        for adjusted_key, raw_key, target in zip(adjusted_keys, raw_keys, targets):
            a = float(row[adjusted_key])
            r = float(row[raw_key])
            y = int(row[target])
            db = (a-y)**2 - (r-y)**2
            dl = (
                -y*math.log(a) - (1-y)*math.log1p(-a)
                +y*math.log(r) + (1-y)*math.log1p(-r)
            )
            brier_delta += db
            logloss_delta += dl
            observations += 1
            cluster = days[row["date"]]
            cluster[0] += 1
            cluster[1] += db
            cluster[2] += dl
    interval = None
    if len(days) >= 20:
        values = list(days.values())
        rng = random.Random(20260928)
        boot = []
        for _ in range(draws):
            picks = rng.choices(values, k=len(values))
            count = sum(x[0] for x in picks)
            boot.append((
                sum(x[1] for x in picks)/count,
                sum(x[2] for x in picks)/count,
            ))
        briers = sorted(x[0] for x in boot)
        logs = sorted(x[1] for x in boot)
        lo = int(0.025*(draws-1))
        hi = int(0.975*(draws-1))
        interval = {"brier": [briers[lo], briers[hi]], "logloss": [logs[lo], logs[hi]]}
    return {
        "n": observations,
        "brier": brier_delta/observations,
        "logloss": logloss_delta/observations,
        "ci95_date_cluster": interval,
    }


def score(archive_dir: Path, outcome_dir: Path, baseline_file: Path, draws: int = 5000) -> dict:
    groups, snapshot_count = candidates(archive_dir)
    selected = []
    for game_pk, rows in sorted(groups.items()):
        # Fixed before looking at outcomes: the latest valid pregame snapshot.
        # A newer provisional correction supersedes an older confirmation.
        # Invalidated forecasts were excluded above for every result.
        selected.append(max(rows, key=lambda r: (r["generated_timestamp"], r["snapshot_sha256"])))
    if not selected:
        raise ValueError("No eligible pregame vNext shadow forecasts")
    all_selected = selected
    starts = [r["validation_cohort"].get("prospective_validation_start") or "0000-00-00" for r in all_selected]
    current_start = max(starts)
    selected = [
        r for r in all_selected
        if (r["validation_cohort"].get("prospective_validation_start") or "0000-00-00") == current_start
    ]
    current_cohorts = {r["validation_cohort_key"] for r in selected}
    if len(current_cohorts) != 1:
        raise ValueError("Multiple current prospective validation cohorts share the latest start date")
    current_cohort = selected[0]["validation_cohort"]
    excluded_prior_cohort = len(all_selected) - len(selected)
    seasons = {int(r["date"][:4]) for r in selected}
    if len(seasons) != 1:
        raise ValueError("Score one prospective season per report")
    baseline, baseline_games = prior_season_under(baseline_file, seasons.pop())
    if not 0 < baseline < 1:
        raise ValueError("Prior-season baseline probability must lie strictly between 0 and 1")
    scored = []
    pending = []
    for row in selected:
        path = outcome_dir / f"{row['game_pk']}.json"
        if not path.exists():
            pending.append({"game_pk": row["game_pk"], "reason": "FINAL_FEED_MISSING"})
            continue
        observed = official_outcome(json.loads(path.read_text()), row["game_pk"])
        if observed is None:
            pending.append({"game_pk": row["game_pk"], "reason": "GAME_NOT_FINAL"})
            continue
        top, bottom = observed
        scored.append({
            **{k: v for k, v in row.items() if k != "generated_timestamp"},
            "observed_top_runs": top,
            "observed_bottom_runs": bottom,
            "observed_under": int(top + bottom == 0),
            "observed_top_under": int(top == 0),
            "observed_bottom_under": int(bottom == 0),
            "baseline_under": baseline,
        })
    all_metrics = metrics(scored, "p_under", "observed_under")
    control = metrics(scored, "baseline_under", "observed_under")
    interval = None
    days = defaultdict(list)
    for row in scored:
        days[row["date"]].append(row)
    if len(days) >= 20:
        values = []
        for rows in days.values():
            brier_delta = sum((r["p_under"]-r["observed_under"])**2-(baseline-r["observed_under"])**2 for r in rows)
            logloss_delta = sum(
                -r["observed_under"]*math.log(r["p_under"])-(1-r["observed_under"])*math.log1p(-r["p_under"])
                +r["observed_under"]*math.log(baseline)+(1-r["observed_under"])*math.log1p(-baseline)
                for r in rows
            )
            values.append((len(rows), brier_delta, logloss_delta))
        rng = random.Random(20260928)
        deltas = []
        for _ in range(draws):
            picks = rng.choices(values, k=len(values))
            count = sum(n for n, _, _ in picks)
            deltas.append((sum(b for _, b, _ in picks)/count, sum(l for _, _, l in picks)/count))
        deltas.sort(key=lambda pair: pair[0])
        brier_ci = [deltas[int(0.025*(draws-1))][0], deltas[int(0.975*(draws-1))][0]]
        log_ci = sorted(pair[1] for pair in deltas)
        interval = {"brier": brier_ci, "logloss": [log_ci[int(0.025*(draws-1))], log_ci[int(0.975*(draws-1))]]}
    half_delta = paired_raw_adjusted_delta(
        scored,
        ("p_top_under", "p_bottom_under"),
        ("p_raw_top_under", "p_raw_bottom_under"),
        ("observed_top_under", "observed_bottom_under"),
        draws,
    )
    full_half_delta = paired_raw_adjusted_delta(
        scored,
        ("p_half_adjusted_under",),
        ("p_raw_under",),
        ("observed_under",),
        draws,
    )
    return {
        "version": "i2-vnext-prospective-score-v2",
        "market_inputs_used": False,
        "promotion_status": "SHADOW_ONLY",
        "selection_policy": "Latest valid pregame forecast per game, then restrict scoring to the single latest declared prospective-validation cohort; exclude source-invalidated snapshots before outcomes.",
        "archive_snapshots": snapshot_count,
        "all_selected_games_before_cohort_filter": len(all_selected),
        "excluded_prior_cohort_games": excluded_prior_cohort,
        "validation_cohort": current_cohort,
        "selected_games": len(selected),
        "scored_games": len(scored),
        "pending": pending,
        "baseline": {"method": "prior regular-season full-I2 Under constant", "probability": baseline, "prior_games": baseline_games},
        "vnext": all_metrics,
        "prior_constant": control,
        "vnext_minus_prior": {
            "brier": all_metrics.get("brier", 0) - control.get("brier", 0) if scored else None,
            "logloss": all_metrics.get("logloss", 0) - control.get("logloss", 0) if scored else None,
            "ci95_date_cluster": interval,
        },
        "confirmed_only": metrics([r for r in scored if r["prediction_class"] == "CONFIRMED_INPUTS"], "p_under", "observed_under"),
        "half_innings": {
            "top": metrics(scored, "p_top_under", "observed_top_under"),
            "bottom": metrics(scored, "p_bottom_under", "observed_bottom_under"),
        },
        "half_innings_raw": {
            "top": metrics(scored, "p_raw_top_under", "observed_top_under"),
            "bottom": metrics(scored, "p_raw_bottom_under", "observed_bottom_under"),
        },
        "half_contrast_adjusted_minus_raw": half_delta,
        "full_i2_half_adjusted": metrics(scored, "p_half_adjusted_under", "observed_under"),
        "full_i2_raw_before_half_adjustment": metrics(scored, "p_raw_under", "observed_under"),
        "full_i2_half_adjusted_minus_raw": full_half_delta,
        "scored_rows": scored,
        "limitations": [
            "No inference interval is reported with fewer than 20 scored calendar dates.",
            "The baseline is a prior-season constant, not the deployed production model at the same cutoff.",
            "No prices, EV, or staking metrics are calculated; this is baseball-only forecast validation.",
            "Prior validation cohorts are reported as excluded rather than pooled with the current half-calibration cohort.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    commands = p.add_subparsers(dest="command", required=True)
    a = commands.add_parser("archive")
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--archive-dir", type=Path, required=True)
    f = commands.add_parser("fetch-final")
    f.add_argument("--archive-dir", type=Path, required=True)
    f.add_argument("--outcome-dir", type=Path, required=True)
    s = commands.add_parser("score")
    s.add_argument("--archive-dir", type=Path, required=True)
    s.add_argument("--outcome-dir", type=Path, required=True)
    s.add_argument("--prior-csv", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.command == "archive":
        result = archive(args.snapshot, args.archive_dir)
    elif args.command == "fetch-final":
        result = fetch_final_feeds(args.archive_dir, args.outcome_dir)
    else:
        result = score(args.archive_dir, args.outcome_dir, args.prior_csv)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

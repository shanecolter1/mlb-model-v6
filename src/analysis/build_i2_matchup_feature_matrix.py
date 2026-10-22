#!/usr/bin/env python3
"""Build a flat historical I2 matchup feature matrix from research snapshots.

This is a research/discovery builder. Existing historical snapshots may be marked
retrospective_research because final-feed lineup/starter timing was not verified as
pregame. Those rows remain explicitly flagged and MUST NOT be promoted as pristine
pregame validation evidence.

The output can feed i2_matchup_variable_research.py for exploratory screening. A
production-eligible fit requires archived/as-of pregame snapshots or later forward data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = ["single", "double", "triple", "home_run", "walk", "hit_by_pitch", "strikeout", "ball_in_play_out"]


def pick_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def detect_master_cols(df):
    gid = pick_col(df, ["game_id", "game_pk", "gamepk", "mlb_game_id"])
    date = pick_col(df, ["game_date", "date"])
    total = pick_col(df, ["opening_total", "dk_opening_total", "draftkings_opening_total", "pregame_opening_total", "game_total_open", "total_open", "open_total"])
    i2 = pick_col(df, ["i2_runs", "inning_2_runs", "inning2_runs", "runs_inning_2"])
    away_i2 = pick_col(df, ["away_i2", "away_inning_2", "away_inning2_runs", "away_runs_i2"])
    home_i2 = pick_col(df, ["home_i2", "home_inning_2", "home_inning2_runs", "home_runs_i2"])
    if not total or (not i2 and not (away_i2 and home_i2)):
        raise ValueError("Joined master must contain opening total and I2 run outcome columns.")
    if not gid and not date:
        raise ValueError("Joined master must contain game_id/gamePk or game_date for snapshot reconciliation.")
    return gid, date, total, i2, away_i2, home_i2


def load_start_slot_weights(path: Path):
    df = pd.read_csv(path)
    slot_col = pick_col(df, ["i2_start_slot", "slot"])
    n_col = pick_col(df, ["n", "count"])
    if not slot_col or not n_col:
        raise ValueError("Start-slot summary must contain i2_start_slot and n.")
    d = {int(r[slot_col]): float(r[n_col]) for _, r in df.iterrows()}
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}


def exposure_weights(start_probs, guaranteed_pa=3):
    """Approximate I2 hitter exposure from historical I2 start-slot distribution.

    Each start slot contributes to the first guaranteed_pa batting positions. This is
    deliberately simple and is a discovery feature only; the production model should
    replace it with matchup-specific I1 simulation exposure.
    """
    weights = {i: 0.0 for i in range(1, 10)}
    for start, p in start_probs.items():
        for j in range(guaranteed_pa):
            slot = ((start - 1 + j) % 9) + 1
            weights[slot] += p
    s = sum(weights.values())
    return {k: v / s for k, v in weights.items()}


def rate(lineup_row, key):
    try:
        return float(lineup_row.get("event_rates", {}).get(key, np.nan))
    except Exception:
        return np.nan


def weighted_lineup_rate(lineup, exposure, key):
    vals, weights = [], []
    for idx, batter in enumerate(lineup, start=1):
        v = rate(batter, key)
        if np.isfinite(v):
            vals.append(v)
            weights.append(float(exposure.get(idx, 0.0)))
    if not vals or sum(weights) <= 0:
        return np.nan
    return float(np.average(vals, weights=weights))


def starter_rate(team_input, key):
    try:
        return float(team_input.get("starter_allowed_rates", {}).get(key, np.nan))
    except Exception:
        return np.nan


def side_features(offense_input, defense_input, exposure):
    lineup = offense_input.get("lineup", []) or []
    hit_1b = weighted_lineup_rate(lineup, exposure, "single")
    hit_2b = weighted_lineup_rate(lineup, exposure, "double")
    hit_3b = weighted_lineup_rate(lineup, exposure, "triple")
    h_hr = weighted_lineup_rate(lineup, exposure, "home_run")
    h_bb = weighted_lineup_rate(lineup, exposure, "walk")
    h_k = weighted_lineup_rate(lineup, exposure, "strikeout")

    p_1b = starter_rate(defense_input, "single")
    p_2b = starter_rate(defense_input, "double")
    p_3b = starter_rate(defense_input, "triple")
    p_hr = starter_rate(defense_input, "home_run")
    p_bb = starter_rate(defense_input, "walk")
    p_k = starter_rate(defense_input, "strikeout")

    return {
        "starter_k_rate": p_k,
        "starter_bb_rate": p_bb,
        "starter_hr_rate": p_hr,
        "starter_nonhr_hit_rate": np.nansum([p_1b, p_2b, p_3b]),
        "i2_expected_batter_k_rate": h_k,
        "i2_expected_batter_bb_rate": h_bb,
        "i2_expected_batter_hr_rate": h_hr,
        "i2_expected_batter_woba": np.nansum([0.89 * hit_1b, 1.27 * hit_2b, 1.62 * hit_3b, 2.10 * h_hr, 0.69 * h_bb]),
        "contact_interaction": (1 - h_k) * (1 - p_k) if np.isfinite(h_k) and np.isfinite(p_k) else np.nan,
        "power_interaction": h_hr * p_hr if np.isfinite(h_hr) and np.isfinite(p_hr) else np.nan,
        "baserunner_interaction": h_bb * p_bb if np.isfinite(h_bb) and np.isfinite(p_bb) else np.nan,
    }


def combine_halves(a, b):
    keys = sorted(set(a) | set(b))
    out = {}
    for k in keys:
        vals = [x for x in [a.get(k), b.get(k)] if x is not None and np.isfinite(x)]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out


def snapshot_paths(root: Path):
    return sorted(root.glob("**/*_research.json")) + sorted(root.glob("**/*_pregame.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined-master", required=True)
    ap.add_argument("--snapshots-dir", required=True)
    ap.add_argument("--start-slot-summary", default="data/derived/i2/i2_start_slot_summary.csv")
    ap.add_argument("--output", default="data/derived/i2/i2_matchup_feature_matrix.csv.gz")
    ap.add_argument("--manifest", default="data/derived/i2/i2_matchup_feature_matrix.manifest.json")
    args = ap.parse_args()

    master = pd.read_csv(args.joined_master)
    gid_col, date_col, total_col, i2_col, away_i2_col, home_i2_col = detect_master_cols(master)
    if i2_col:
        master["__i2_runs"] = pd.to_numeric(master[i2_col], errors="coerce")
    else:
        master["__i2_runs"] = pd.to_numeric(master[away_i2_col], errors="coerce") + pd.to_numeric(master[home_i2_col], errors="coerce")
    master["__opening_total"] = pd.to_numeric(master[total_col], errors="coerce")
    if gid_col:
        master["__game_id"] = master[gid_col].astype(str)
    if date_col:
        master["__game_date"] = pd.to_datetime(master[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    start_probs = load_start_slot_weights(Path(args.start_slot_summary))
    exposure = exposure_weights(start_probs)
    rows = []
    classes = {}

    for path in snapshot_paths(Path(args.snapshots_dir)):
        snap = json.loads(path.read_text(encoding="utf-8"))
        game_id = str(snap.get("game_id"))
        game_date = str(snap.get("game_date"))
        if gid_col:
            m = master[master["__game_id"] == game_id]
        else:
            m = master[master["__game_date"] == game_date]
        if len(m) != 1:
            continue
        m = m.iloc[0]
        ti = snap.get("team_inputs", {}) or {}
        home_input = ti.get("home", {}) or {}
        away_input = ti.get("away", {}) or {}
        if not home_input or not away_input:
            continue

        # Top 2: away offense vs home starter. Bottom 2: home offense vs away starter.
        top = side_features(away_input, home_input, exposure)
        bottom = side_features(home_input, away_input, exposure)
        features = combine_halves(top, bottom)

        env = snap.get("environmental_context") or {}
        if isinstance(env, dict):
            for src, dst in [("run_factor", "venue_run_factor"), ("R", "venue_run_factor"), ("hr_factor", "venue_hr_factor"), ("HR", "venue_hr_factor")]:
                if src in env and dst not in features:
                    try:
                        features[dst] = float(env[src]) / (100.0 if float(env[src]) > 5 else 1.0)
                    except Exception:
                        pass

        snap_class = snap.get("snapshot_class", "unknown")
        classes[snap_class] = classes.get(snap_class, 0) + 1
        row = {
            "game_id": game_id,
            "game_date": game_date,
            "season": int(game_date[:4]),
            "opening_total": float(m["__opening_total"]),
            "i2_runs": float(m["__i2_runs"]),
            "snapshot_class": snap_class,
            "production_eligible_snapshot": bool((snap.get("audit") or {}).get("production_eligible", False)),
            "feature_lineage": "retrospective_discovery" if not bool((snap.get("audit") or {}).get("production_eligible", False)) else "pregame_verified",
            **features,
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, compression="gzip" if out_path.suffix == ".gz" else None)

    manifest = {
        "joined_master": args.joined_master,
        "snapshots_dir": args.snapshots_dir,
        "rows": int(len(out)),
        "snapshot_classes": classes,
        "production_eligible_rows": int(out["production_eligible_snapshot"].sum()) if len(out) else 0,
        "retrospective_discovery_rows": int((~out["production_eligible_snapshot"]).sum()) if len(out) else 0,
        "i2_start_slot_exposure_source": args.start_slot_summary,
        "i2_start_slot_exposure_method": "pooled historical start-slot distribution applied to first three I2 lineup positions; discovery only",
        "governance_status": "WARNING" if len(out) and not out["production_eligible_snapshot"].all() else "PASS",
        "governance_warning": "Rows built from retrospective final-feed lineup/actual starter snapshots may be used for exploratory variable screening only, not pristine production-weight validation.",
        "output": str(out_path),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

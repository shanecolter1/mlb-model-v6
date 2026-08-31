#!/usr/bin/env python3
"""Build a reusable leakage-safe historical baseball as-of feature store.

Inputs are the normalized tables produced by normalize_mlb_feeds.py.
All historical predictor values use observations strictly before the target game date.
Outcomes are persisted separately from predictors.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EVENTS = [
    "single", "double", "triple", "home_run", "walk",
    "hit_by_pitch", "strikeout", "ball_in_play_out",
]
WINDOWS = (30, 90, 365)
EPS = 1e-12


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/derived/baseball_asof"))
    p.add_argument("--min-season", type=int, default=2021)
    p.add_argument("--max-season", type=int, default=2025)
    p.add_argument("--shrink-strength", type=float, default=50.0)
    return p.parse_args()


def read_table(root: Path, name: str) -> pd.DataFrame:
    pq = root / f"{name}.parquet"
    csv = root / f"{name}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {name}.parquet/.csv in {root}")


def prep_pa(pa: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    out = out[out["game_date"].notna()].copy()
    for e in EVENTS:
        out[f"ev_{e}"] = (out["event"].astype(str) == e).astype(np.int16)
    out["ev_hit"] = out[["ev_single", "ev_double", "ev_triple", "ev_home_run"]].sum(axis=1)
    out["ev_xbh"] = out[["ev_double", "ev_triple", "ev_home_run"]].sum(axis=1)
    out["ev_onbase"] = out[["ev_single", "ev_double", "ev_triple", "ev_home_run", "ev_walk", "ev_hit_by_pitch"]].sum(axis=1)
    out["ev_contact"] = 1 - out["ev_strikeout"]
    return out


def entity_daily(pa: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if entity_type == "batter":
        entity_col = "batter_id"
    elif entity_type == "pitcher":
        entity_col = "pitcher_id"
    else:
        raise ValueError(entity_type)
    metric_cols = [f"ev_{e}" for e in EVENTS] + ["ev_hit", "ev_xbh", "ev_onbase", "ev_contact"]
    g = (
        pa.groupby(["game_date", entity_col], dropna=False)[metric_cols]
        .sum().reset_index()
        .rename(columns={entity_col: "entity_id"})
    )
    g["opportunities"] = g[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    g["entity_type"] = entity_type
    return g


def league_prior_by_date(pa: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [f"ev_{e}" for e in EVENTS] + ["ev_hit", "ev_xbh", "ev_onbase", "ev_contact"]
    daily = pa.groupby("game_date")[metric_cols].sum().sort_index()
    daily["opportunities"] = daily[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    prior_counts = daily.shift(1).fillna(0).cumsum()
    rates = prior_counts[metric_cols].div(prior_counts["opportunities"].replace(0, np.nan), axis=0)
    rates = rates.fillna(0.0)
    rates.columns = [f"league_{c}_rate" for c in metric_cols]
    return rates.reset_index()


def build_entity_asof(pa: pd.DataFrame, dates: list[pd.Timestamp], shrink_strength: float) -> pd.DataFrame:
    metric_cols = [f"ev_{e}" for e in EVENTS] + ["ev_hit", "ev_xbh", "ev_onbase", "ev_contact"]
    league = league_prior_by_date(pa).set_index("game_date")
    parts = []
    for entity_type in ("batter", "pitcher"):
        daily = entity_daily(pa, entity_type).sort_values(["entity_id", "game_date"])
        entities = daily["entity_id"].dropna().unique()
        if len(entities) == 0:
            continue
        for cutoff in dates:
            hist = daily[daily["game_date"] < cutoff]
            if hist.empty:
                continue
            season_start = pd.Timestamp(year=cutoff.year, month=1, day=1)
            season_hist = hist[hist["game_date"] >= season_start]
            prior = league.loc[cutoff] if cutoff in league.index else None
            for label, frame in [("season", season_hist)] + [
                (f"{w}d", hist[hist["game_date"] >= cutoff - pd.Timedelta(days=w)]) for w in WINDOWS
            ]:
                if frame.empty:
                    continue
                agg = frame.groupby("entity_id")[metric_cols + ["opportunities"]].sum()
                opp = agg["opportunities"].astype(float)
                rows = pd.DataFrame({
                    "as_of_date": cutoff,
                    "entity_type": entity_type,
                    "entity_id": agg.index,
                    "horizon": label,
                    "opportunities": opp.values,
                })
                for c in metric_cols:
                    raw = agg[c].astype(float) / opp.replace(0, np.nan)
                    if prior is not None:
                        prior_rate = float(prior.get(f"league_{c}_rate", 0.0))
                    else:
                        prior_rate = 0.0
                    shrunk = (agg[c].astype(float) + shrink_strength * prior_rate) / (opp + shrink_strength)
                    rows[f"{c}_rate_raw"] = raw.values
                    rows[f"{c}_rate_shrunk"] = shrunk.values
                rows["reliability"] = (opp / (opp + shrink_strength)).values
                parts.append(rows)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def latest_feature_lookup(entity_asof: pd.DataFrame, entity_type: str, horizon: str) -> pd.DataFrame:
    cols = [c for c in entity_asof.columns if c not in {"entity_type", "horizon"}]
    return entity_asof[(entity_asof["entity_type"] == entity_type) & (entity_asof["horizon"] == horizon)][cols].copy()


def weighted_lineup_summary(lineup_rows: pd.DataFrame, batter_features: pd.DataFrame, date, side: str) -> dict:
    lineup = lineup_rows[(lineup_rows["game_date"] == date) & (lineup_rows["team_side"] == side)].sort_values("batting_order_slot")
    out = {"lineup_count": int(len(lineup)), "lineup_provenance": "retrospective_unverified"}
    if lineup.empty:
        return out
    feat = batter_features[batter_features["as_of_date"] == date]
    merged = lineup.merge(feat, left_on="player_id", right_on="entity_id", how="left")
    weights = np.array([0.125, 0.122, 0.119, 0.116, 0.112, 0.108, 0.104, 0.099, 0.095], dtype=float)[:len(merged)]
    weights = weights / weights.sum() if weights.sum() else np.ones(len(merged)) / max(len(merged), 1)
    key_metrics = [
        "ev_strikeout_rate_shrunk", "ev_walk_rate_shrunk", "ev_home_run_rate_shrunk",
        "ev_hit_rate_shrunk", "ev_xbh_rate_shrunk", "ev_onbase_rate_shrunk", "reliability",
    ]
    for m in key_metrics:
        vals = pd.to_numeric(merged.get(m), errors="coerce") if m in merged.columns else pd.Series(np.nan, index=merged.index)
        mask = vals.notna().to_numpy()
        if mask.any():
            ww = weights[mask]
            ww = ww / ww.sum()
            out[f"lineup_{m}"] = float(np.sum(vals.to_numpy()[mask] * ww))
        else:
            out[f"lineup_{m}"] = np.nan
    return out


def starter_summary(starters: pd.DataFrame, pitcher_features: pd.DataFrame, date, side: str) -> dict:
    s = starters[(starters["game_date"] == date) & (starters["team_side"] == side)]
    out = {"starter_provenance": "retrospective_unverified"}
    if s.empty:
        return out
    row = s.iloc[0]
    pid = row["pitcher_id"]
    out["starter_id"] = pid
    out["starter_name"] = row.get("name")
    feat = pitcher_features[(pitcher_features["as_of_date"] == date) & (pitcher_features["entity_id"].astype(str) == str(pid))]
    if feat.empty:
        return out
    f = feat.iloc[0]
    for m in [
        "opportunities", "reliability", "ev_strikeout_rate_shrunk", "ev_walk_rate_shrunk",
        "ev_home_run_rate_shrunk", "ev_hit_rate_shrunk", "ev_xbh_rate_shrunk", "ev_onbase_rate_shrunk",
    ]:
        out[f"starter_{m}"] = f.get(m)
    return out


def build_game_features(games, lineups, starters, entity_asof):
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    lineups = lineups.copy(); lineups["game_date"] = pd.to_datetime(lineups["game_date"], errors="coerce").dt.normalize()
    starters = starters.copy(); starters["game_date"] = pd.to_datetime(starters["game_date"], errors="coerce").dt.normalize()
    batter = latest_feature_lookup(entity_asof, "batter", "365d")
    pitcher = latest_feature_lookup(entity_asof, "pitcher", "365d")
    rows = []
    for _, g in games.sort_values(["game_date", "game_id"]).iterrows():
        d = g["game_date"]
        row = {
            "game_id": g["game_id"], "game_date": d,
            "home_team_id": g.get("home_team_id"), "away_team_id": g.get("away_team_id"),
            "home_team": g.get("home_team"), "away_team": g.get("away_team"),
            "venue_id": g.get("venue_id"), "venue_name": g.get("venue_name"),
            "feature_cutoff_rule": "strictly_before_game_date",
        }
        for side in ("home", "away"):
            ls = weighted_lineup_summary(lineups[lineups["game_id"] == g["game_id"]], batter, d, side)
            ss = starter_summary(starters[starters["game_id"] == g["game_id"]], pitcher, d, side)
            row.update({f"{side}_{k}": v for k, v in {**ls, **ss}.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_outcomes(games: pd.DataFrame, results: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    games2 = games[["game_id", "game_date"]].copy()
    out = games2.merge(results, on=["game_id", "game_date"], how="left")
    pa2 = pa.copy()
    pa2["rbi"] = pd.to_numeric(pa2.get("rbi"), errors="coerce").fillna(0)
    # RBI is not a perfect inning-run reconstruction in every edge case; keep explicit provenance.
    inn = pa2.groupby(["game_id", "inning", "half_inning"])["rbi"].sum().unstack(["inning", "half_inning"], fill_value=0)
    inn.columns = [f"i{int(i)}_{h}_rbi_proxy_runs" for i, h in inn.columns]
    inn = inn.reset_index()
    out = out.merge(inn, on="game_id", how="left")
    out["inning_outcome_provenance"] = "rbi_proxy_from_normalized_pa; use linescore-derived runs when available"
    return out


def feature_dictionary(game_features: pd.DataFrame, entity_asof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in entity_asof.columns:
        rows.append({"table": "entity_asof", "feature": c, "leakage_class": "as_of_safe" if c not in {"entity_type", "entity_id", "horizon"} else "key", "notes": "strictly prior-date aggregation"})
    for c in game_features.columns:
        if "provenance" in c:
            cls = "audit"
        elif c in {"game_id", "game_date", "home_team_id", "away_team_id", "venue_id"}:
            cls = "key"
        elif "lineup_" in c or "starter_" in c:
            cls = "retrospective_unverified_source"
        else:
            cls = "context"
        rows.append({"table": "game_asof_features", "feature": c, "leakage_class": cls, "notes": "see source provenance fields"})
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    games = read_table(args.processed_dir, "games")
    pa = prep_pa(read_table(args.processed_dir, "plate_appearances"))
    lineups = read_table(args.processed_dir, "lineups")
    starters = read_table(args.processed_dir, "starters")
    results = read_table(args.processed_dir, "results")

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    season = games["game_date"].dt.year
    games = games[(season >= args.min_season) & (season <= args.max_season)].copy()
    game_ids = set(games["game_id"].dropna())
    pa = pa[pa["game_id"].isin(game_ids)].copy()
    lineups = lineups[lineups["game_id"].isin(game_ids)].copy()
    starters = starters[starters["game_id"].isin(game_ids)].copy()
    results["game_date"] = pd.to_datetime(results["game_date"], errors="coerce").dt.normalize()
    results = results[results["game_id"].isin(game_ids)].copy()

    dates = sorted(games["game_date"].dropna().unique())
    dates = [pd.Timestamp(d) for d in dates]
    entity = build_entity_asof(pa, dates, args.shrink_strength)
    game_features = build_game_features(games, lineups, starters, entity)
    outcomes = build_outcomes(games, results, pa)
    dictionary = feature_dictionary(game_features, entity)

    entity.to_parquet(args.output_dir / "entity_asof.parquet", index=False)
    game_features.to_parquet(args.output_dir / "game_asof_features.parquet", index=False)
    outcomes.to_parquet(args.output_dir / "game_outcomes.parquet", index=False)
    dictionary.to_csv(args.output_dir / "feature_dictionary.csv", index=False)

    manifest = {
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "version": "1.0-research",
        "source_processed_dir": str(args.processed_dir),
        "season_window": [args.min_season, args.max_season],
        "strict_cutoff": "event_date < target_game_date",
        "same_day_policy": "same-day prior games excluded by default",
        "shrink_strength": args.shrink_strength,
        "windows_days": list(WINDOWS),
        "rows": {
            "games": int(len(games)),
            "plate_appearances": int(len(pa)),
            "entity_asof": int(len(entity)),
            "game_asof_features": int(len(game_features)),
            "game_outcomes": int(len(outcomes)),
        },
        "source_provenance_warning": "normalized final-feed lineups and actual first pitchers are retrospective_unverified until archived pregame availability is established",
        "outcome_warning": "inning-level outcome columns generated here are RBI proxies; use linescore-derived inning runs for final target construction when available",
        "market_data_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build reusable leakage-safe batter/pitcher as-of event rates efficiently.

This component is intentionally model-agnostic. It creates one snapshot per active
entity/date using only observations from strictly earlier dates. Same-day prior
games are excluded because daily events are aggregated before lagging.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

EVENTS = ["single","double","triple","home_run","walk","hit_by_pitch","strikeout","ball_in_play_out"]
DERIVED = ["hit","xbh","onbase","contact"]
WINDOWS = (30, 90, 365)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plate-appearances", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shrink-strength", type=float, default=50.0)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def prep(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce").dt.normalize()
    x = x[x["game_date"].notna()].copy()
    for e in EVENTS:
        x[f"ev_{e}"] = (x["event"].astype(str) == e).astype("int16")
    x["ev_hit"] = x[["ev_single","ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_xbh"] = x[["ev_double","ev_triple","ev_home_run"]].sum(axis=1)
    x["ev_onbase"] = x[["ev_single","ev_double","ev_triple","ev_home_run","ev_walk","ev_hit_by_pitch"]].sum(axis=1)
    x["ev_contact"] = 1 - x["ev_strikeout"]
    metrics = [f"ev_{e}" for e in EVENTS] + [f"ev_{e}" for e in DERIVED]
    return x, metrics


def league_prior(pa: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    d = pa.groupby("game_date")[metrics].sum().sort_index()
    d["opportunities"] = d[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    prior = d.shift(1).fillna(0).cumsum()
    out = prior[metrics].div(prior["opportunities"].replace(0, np.nan), axis=0).fillna(0)
    out.columns = [f"league_{c}" for c in metrics]
    return out


def daily_entity(pa: pd.DataFrame, entity_col: str, metrics: list[str]) -> pd.DataFrame:
    d = pa.dropna(subset=[entity_col]).groupby([entity_col,"game_date"])[metrics].sum().reset_index()
    d = d.rename(columns={entity_col:"entity_id"}).sort_values(["entity_id","game_date"])
    d["opportunities"] = d[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    return d


def season_prior(daily: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    x = daily.copy()
    x["season"] = x["game_date"].dt.year
    cols = metrics + ["opportunities"]
    additions = {}
    for c in cols:
        additions[f"season_{c}"] = x.groupby(["entity_id","season"], sort=False)[c].transform(
            lambda s: s.shift(1).fillna(0).cumsum()
        )
    z = pd.concat([x[["entity_id","game_date"]].reset_index(drop=True), pd.DataFrame(additions)], axis=1)
    return z


def rolling_prior(daily: pd.DataFrame, metrics: list[str], days: int) -> pd.DataFrame:
    cols = metrics + ["opportunities"]
    parts = []
    for entity_id, g in daily.groupby("entity_id", sort=False):
        g = g.sort_values("game_date")
        r = g.set_index("game_date")[cols].rolling(f"{days}D", closed="left").sum().fillna(0)
        r = r.add_prefix(f"{days}d_").reset_index()
        r.insert(0, "entity_id", entity_id)
        parts.append(r)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def add_rates(frame: pd.DataFrame, prefix: str, metrics: list[str], prior: pd.DataFrame, strength: float) -> pd.DataFrame:
    x = frame.merge(prior.reset_index(), on="game_date", how="left")
    opp = pd.to_numeric(x[f"{prefix}_opportunities"], errors="coerce").fillna(0.0)
    additions = {f"{prefix}_reliability": opp / (opp + strength)}
    for c in metrics:
        count = pd.to_numeric(x[f"{prefix}_{c}"], errors="coerce").fillna(0.0)
        lp = pd.to_numeric(x[f"league_{c}"], errors="coerce").fillna(0.0)
        additions[f"{prefix}_{c}_rate_raw"] = count / opp.replace(0, np.nan)
        additions[f"{prefix}_{c}_rate_shrunk"] = (count + strength * lp) / (opp + strength)
    x = pd.concat([x.drop(columns=[f"league_{c}" for c in metrics]), pd.DataFrame(additions, index=x.index)], axis=1)
    return x.copy()


def build_entity(pa: pd.DataFrame, entity_type: str, metrics: list[str], league: pd.DataFrame, strength: float) -> pd.DataFrame:
    entity_col = "batter_id" if entity_type == "batter" else "pitcher_id"
    d = daily_entity(pa, entity_col, metrics)
    out = season_prior(d, metrics)
    for days in WINDOWS:
        out = out.merge(rolling_prior(d, metrics, days), on=["entity_id","game_date"], how="left")
    for prefix in ["season"] + [f"{d}d" for d in WINDOWS]:
        out = add_rates(out, prefix, metrics, league, strength)
    out = pd.concat([out[["entity_id"]], pd.Series(entity_type, index=out.index, name="entity_type"), out.drop(columns=["entity_id"])], axis=1)
    return out.rename(columns={"game_date":"as_of_date"})


def main():
    args = parse_args()
    pa, metrics = prep(read(args.plate_appearances))
    league = league_prior(pa, metrics)
    batter = build_entity(pa, "batter", metrics, league, args.shrink_strength)
    pitcher = build_entity(pa, "pitcher", metrics, league, args.shrink_strength)
    out = pd.concat([batter, pitcher], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print({"plate_appearances": len(pa), "entity_snapshots": len(out), "output": str(args.output)})


if __name__ == "__main__":
    main()

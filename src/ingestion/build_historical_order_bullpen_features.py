#!/usr/bin/env python3
"""Build leakage-safe batting-order path and bullpen context features.

Outputs
-------
- batting_order_path_asof.parquet
    Team/inning prior distribution of the lineup slot expected to lead off an
    inning, estimated only from games strictly before the target date.
- bullpen_team_asof.parquet
    Team bullpen quality and recent workload/availability proxies based only on
    relief appearances strictly before the target date.
- bullpen_pitcher_asof.parquet
    Reliever-level quality/workload histories for downstream roster-aware use.

Historical final-feed lineup/starter identities are used only to classify what
actually happened in completed historical games. They are not represented as
verified pregame identities. Same-day earlier games are intentionally excluded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

EVENTS = ["single", "double", "triple", "home_run", "walk", "hit_by_pitch", "strikeout", "ball_in_play_out"]
DERIVED = ["hit", "xbh", "onbase", "contact"]
QUALITY_WINDOWS = (30, 90, 365)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plate-appearances", type=Path, required=True)
    p.add_argument("--lineups", type=Path, required=True)
    p.add_argument("--starters", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shrink-strength", type=float, default=75.0)
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def prep_pa(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce").dt.normalize()
    x["inning"] = pd.to_numeric(x["inning"], errors="coerce").astype("Int64")
    for e in EVENTS:
        x[f"ev_{e}"] = (x["event"].astype(str) == e).astype("int16")
    x["ev_hit"] = x[["ev_single", "ev_double", "ev_triple", "ev_home_run"]].sum(axis=1)
    x["ev_xbh"] = x[["ev_double", "ev_triple", "ev_home_run"]].sum(axis=1)
    x["ev_onbase"] = x[["ev_single", "ev_double", "ev_triple", "ev_home_run", "ev_walk", "ev_hit_by_pitch"]].sum(axis=1)
    x["ev_contact"] = 1 - x["ev_strikeout"]
    return x, [f"ev_{e}" for e in EVENTS + DERIVED]


def build_order_path(pa: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """Historical team/inning lead-off slot probabilities with strict date lag."""
    lu = lineups.copy()
    lu["game_date"] = pd.to_datetime(lu["game_date"], errors="coerce").dt.normalize()
    lu["batting_order_slot"] = pd.to_numeric(lu["batting_order_slot"], errors="coerce")
    slot_map = lu[["game_id", "team_id", "player_id", "batting_order_slot"]].drop_duplicates()

    first = (pa.dropna(subset=["game_id", "batting_team_id", "inning", "batter_id", "play_index"])
             .sort_values(["game_id", "batting_team_id", "inning", "play_index"])
             .groupby(["game_id", "game_date", "batting_team_id", "inning"], as_index=False)
             .first())
    first = first.merge(
        slot_map,
        left_on=["game_id", "batting_team_id", "batter_id"],
        right_on=["game_id", "team_id", "player_id"],
        how="left",
    )
    first = first[first["inning"].between(1, 9) & first["batting_order_slot"].between(1, 9)].copy()
    first["team_id"] = first["batting_team_id"]

    # Daily counts let us exclude all same-day information from predictors.
    d = (first.groupby(["team_id", "inning", "game_date", "batting_order_slot"])
         .size().rename("n").reset_index())
    dates = first[["team_id", "game_date"]].drop_duplicates()
    rows = []

    # League inning prior, strictly before each date.
    ld = (first.groupby(["inning", "game_date", "batting_order_slot"])
          .size().rename("n").reset_index())
    league_prior = {}
    for inn in range(1, 10):
        z = ld[ld["inning"] == inn]
        for date in sorted(first["game_date"].dropna().unique()):
            h = z[z["game_date"] < date]
            counts = h.groupby("batting_order_slot")["n"].sum() if len(h) else pd.Series(dtype=float)
            total = float(counts.sum())
            league_prior[(inn, pd.Timestamp(date))] = {s: (float(counts.get(s, 0)) / total if total else 1/9) for s in range(1, 10)}

    for team, td in dates.groupby("team_id", sort=False):
        team_hist = d[d["team_id"] == team]
        for date in sorted(td["game_date"].dropna().unique()):
            for inn in range(1, 10):
                h = team_hist[(team_hist["inning"] == inn) & (team_hist["game_date"] < date)]
                counts = h.groupby("batting_order_slot")["n"].sum() if len(h) else pd.Series(dtype=float)
                n = float(counts.sum())
                # Modest inning-specific team shrinkage to lagged league distribution.
                alpha = 20.0
                lp = league_prior.get((inn, pd.Timestamp(date)), {s: 1/9 for s in range(1,10)})
                probs = {s: (float(counts.get(s, 0)) + alpha * lp[s]) / (n + alpha) for s in range(1, 10)}
                rec = {
                    "team_id": team,
                    "as_of_date": pd.Timestamp(date),
                    "inning": inn,
                    "prior_team_inning_games": int(n),
                    "reliability": n / (n + alpha),
                    "expected_start_slot": sum(s * probs[s] for s in range(1, 10)),
                    "source_class": "retrospective_outcomes_strictly_prior_date",
                }
                for s in range(1, 10):
                    rec[f"p_start_slot_{s}"] = probs[s]
                rows.append(rec)
    return pd.DataFrame(rows)


def classify_relief(pa: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    st = starters[["game_id", "team_id", "pitcher_id"]].dropna(subset=["pitcher_id"]).copy()
    st = st.rename(columns={"pitcher_id": "starter_id", "team_id": "pitching_team_id"})
    x = pa.merge(st, on=["game_id", "pitching_team_id"], how="left")
    x["is_relief"] = x["starter_id"].notna() & (x["pitcher_id"] != x["starter_id"])
    return x


def add_rates(out: pd.DataFrame, prefix: str, metrics: list[str], strength: float, priors: dict[str, float]) -> pd.DataFrame:
    opp = pd.to_numeric(out[f"{prefix}_opportunities"], errors="coerce").fillna(0.0)
    out[f"{prefix}_reliability"] = opp / (opp + strength)
    new = {}
    for m in metrics:
        cnt = pd.to_numeric(out[f"{prefix}_{m}"], errors="coerce").fillna(0.0)
        new[f"{prefix}_{m}_rate_shrunk"] = (cnt + strength * priors[m]) / (opp + strength)
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def build_bullpen_pitcher(pa: pd.DataFrame, metrics: list[str], strength: float) -> pd.DataFrame:
    r = pa[pa["is_relief"]].dropna(subset=["pitcher_id", "game_date"]).copy()
    daily = r.groupby(["pitcher_id", "game_date"])[metrics].sum().reset_index()
    daily["opportunities"] = daily[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    total_opp = float(daily["opportunities"].sum())
    priors = {m: (float(daily[m].sum()) / total_opp if total_opp else 0.0) for m in metrics}
    cols = metrics + ["opportunities"]
    parts = []
    for pid, g in daily.groupby("pitcher_id", sort=False):
        g = g.sort_values("game_date").copy()
        base = g[["pitcher_id", "game_date"]].copy()
        for days in QUALITY_WINDOWS:
            z = g.set_index("game_date")[cols].rolling(f"{days}D", closed="left").sum().fillna(0).reset_index()
            for c in cols:
                base[f"{days}d_{c}"] = z[c].values
        # Availability/workload proxies.
        for days in (1, 2, 3, 7, 14):
            z = g.set_index("game_date")[["opportunities"]].rolling(f"{days}D", closed="left").sum().fillna(0).reset_index()
            base[f"relief_bf_{days}d"] = z["opportunities"].values
        base["last_relief_date"] = g["game_date"].shift(1)
        base["days_since_relief"] = (g["game_date"] - base["last_relief_date"]).dt.days
        parts.append(base)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return out
    for days in QUALITY_WINDOWS:
        out = add_rates(out, f"{days}d", metrics, strength, priors)
    out = out.rename(columns={"game_date": "as_of_date"})
    out["source_class"] = "relief_history_strictly_prior_date"
    return out


def build_bullpen_team(pa: pd.DataFrame, metrics: list[str], strength: float) -> pd.DataFrame:
    r = pa[pa["is_relief"]].dropna(subset=["pitching_team_id", "game_date"]).copy()
    daily = r.groupby(["pitching_team_id", "game_date"])[metrics].sum().reset_index()
    daily["opportunities"] = daily[[f"ev_{e}" for e in EVENTS]].sum(axis=1)
    used = (r.groupby(["pitching_team_id", "game_date"])["pitcher_id"]
            .nunique().rename("relievers_used").reset_index())
    daily = daily.merge(used, on=["pitching_team_id", "game_date"], how="left")
    total_opp = float(daily["opportunities"].sum())
    priors = {m: (float(daily[m].sum()) / total_opp if total_opp else 0.0) for m in metrics}
    cols = metrics + ["opportunities"]
    parts = []
    for team, g in daily.groupby("pitching_team_id", sort=False):
        g = g.sort_values("game_date").copy()
        base = g[["pitching_team_id", "game_date"]].copy()
        for days in QUALITY_WINDOWS:
            z = g.set_index("game_date")[cols].rolling(f"{days}D", closed="left").sum().fillna(0).reset_index()
            for c in cols:
                base[f"{days}d_{c}"] = z[c].values
        for days in (1, 2, 3, 7, 14):
            z = g.set_index("game_date")[["opportunities", "relievers_used"]].rolling(f"{days}D", closed="left").sum().fillna(0).reset_index()
            base[f"bullpen_bf_{days}d"] = z["opportunities"].values
            base[f"reliever_uses_{days}d"] = z["relievers_used"].values
        parts.append(base)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return out
    for days in QUALITY_WINDOWS:
        out = add_rates(out, f"{days}d", metrics, strength, priors)
    out = out.rename(columns={"pitching_team_id": "team_id", "game_date": "as_of_date"})
    out["source_class"] = "team_relief_history_strictly_prior_date"
    return out


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    pa, metrics = prep_pa(read(a.plate_appearances))
    lineups = read(a.lineups)
    starters = read(a.starters)

    order = build_order_path(pa, lineups)
    classified = classify_relief(pa, starters)
    bp_pitcher = build_bullpen_pitcher(classified, metrics, a.shrink_strength)
    bp_team = build_bullpen_team(classified, metrics, a.shrink_strength * 2)

    order.to_parquet(a.output_dir / "batting_order_path_asof.parquet", index=False)
    bp_pitcher.to_parquet(a.output_dir / "bullpen_pitcher_asof.parquet", index=False)
    bp_team.to_parquet(a.output_dir / "bullpen_team_asof.parquet", index=False)

    manifest = {
        "strict_cutoff": "source game_date < target game_date",
        "same_day_prior_games_included": False,
        "market_data_used": False,
        "pregame_identity_claimed": False,
        "outputs": {
            "batting_order_path_rows": int(len(order)),
            "bullpen_pitcher_rows": int(len(bp_pitcher)),
            "bullpen_team_rows": int(len(bp_team)),
        },
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()

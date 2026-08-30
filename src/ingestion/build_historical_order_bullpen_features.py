#!/usr/bin/env python3
"""Build leakage-safe all-inning batting-order and bullpen context features.

Batting-order treatment is fully empirical. No smoothing is applied to the
production batting-order probabilities. In addition to strict prior-date team
features, the build emits all-inning descriptive and walk-forward validation
artifacts so the observed I1-I9 lineup-path distributions can be audited before
use in predictive models.

Observed batting-order slots are reconstructed from each team's chronological
plate-appearance sequence within a game. Because substitutions inherit the
replaced player's batting-order position, PA ordinal modulo nine preserves the
lineup slot without relying on the final-feed player identity list.

Outputs
-------
- batting_order_path_asof.parquet
    Raw team/inning prior-date distribution of the lineup slot that led off an
    inning. Rows with no prior team/inning history retain NA probabilities.
- batting_order_empirical_all_innings.parquet
    Overall 2021-2025 empirical start-slot counts/probabilities for I1-I9.
- batting_order_empirical_by_season.parquet
    Same distributions by season for stability analysis.
- batting_order_transition_empirical.parquet
    Empirical transition distribution from the prior inning's starting slot to
    the next inning's starting slot, by destination inning.
- batting_order_walkforward_validation.parquet
    Leave-one-season-forward validation against the uniform 1/9 benchmark.
- batting_order_stability.parquet
    Season-to-full-sample absolute probability drift diagnostics.
- bullpen_team_asof.parquet
- bullpen_pitcher_asof.parquet

Historical final-feed starter identities classify completed historical outcomes
only; they are not claimed as verified pregame identities. Same-day prior games
are intentionally excluded from as-of predictors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    p.add_argument("--shrink-strength", type=float, default=75.0,
                   help="Bullpen quality shrinkage only; never used for batting-order probabilities.")
    return p.parse_args()


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def prep_pa(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()
    x["game_date"] = pd.to_datetime(x["game_date"], errors="coerce").dt.normalize()
    x["season"] = x["game_date"].dt.year.astype("Int64")
    x["inning"] = pd.to_numeric(x["inning"], errors="coerce").astype("Int64")
    for e in EVENTS:
        x[f"ev_{e}"] = (x["event"].astype(str) == e).astype("int16")
    x["ev_hit"] = x[["ev_single", "ev_double", "ev_triple", "ev_home_run"]].sum(axis=1)
    x["ev_xbh"] = x[["ev_double", "ev_triple", "ev_home_run"]].sum(axis=1)
    x["ev_onbase"] = x[["ev_single", "ev_double", "ev_triple", "ev_home_run", "ev_walk", "ev_hit_by_pitch"]].sum(axis=1)
    x["ev_contact"] = 1 - x["ev_strikeout"]
    return x, [f"ev_{e}" for e in EVENTS + DERIVED]


def inning_start_observations(pa: pd.DataFrame, lineups: pd.DataFrame) -> pd.DataFrame:
    """One observed batting-order start slot per team-game-inning.

    Reconstruct slot from chronological PA ordinal rather than final-feed batter
    identity. A substitute occupies the batting-order slot of the player replaced,
    so the team PA sequence remains cyclic 1..9 throughout the game. This avoids
    losing observations when the final box-score battingOrder contains substitute
    identities instead of the original starter who took an earlier PA.
    """
    del lineups  # retained in the call signature for backward-compatible workflow inputs
    x = pa.dropna(subset=["game_id", "batting_team_id", "inning", "play_index"]).copy()
    x = x.sort_values(["game_id", "batting_team_id", "play_index"], kind="stable")
    x["team_pa_ordinal"] = x.groupby(["game_id", "batting_team_id"]).cumcount()
    x["batting_order_slot"] = (x["team_pa_ordinal"] % 9) + 1

    first = (x[x["inning"].between(1, 9)]
             .sort_values(["game_id", "batting_team_id", "inning", "play_index"], kind="stable")
             .groupby(["game_id", "game_date", "season", "batting_team_id", "inning"], as_index=False)
             .first())
    first["team_id"] = first["batting_team_id"]
    first["batting_order_slot"] = first["batting_order_slot"].astype(int)
    first["slot_source"] = "team_plate_appearance_ordinal_mod9"
    return first[["game_id", "game_date", "season", "team_id", "inning", "batting_order_slot", "slot_source"]]


def empirical_distribution(first: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = first.groupby(group_cols + ["batting_order_slot"]).size().rename("n").reset_index()
    totals = counts.groupby(group_cols)["n"].transform("sum")
    counts["probability"] = counts["n"] / totals
    counts["sample_n"] = totals
    # Wilson 95% interval: descriptive uncertainty, not smoothing.
    z = 1.959963984540054
    p = counts["probability"].astype(float)
    n = counts["sample_n"].astype(float)
    denom = 1.0 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*np.sqrt((p*(1-p)/n) + z*z/(4*n*n)) / denom
    counts["wilson95_low"] = (center-half).clip(0, 1)
    counts["wilson95_high"] = (center+half).clip(0, 1)
    return counts


def build_order_path_asof(first: pd.DataFrame) -> pd.DataFrame:
    """Raw team/inning empirical probabilities using strictly prior dates."""
    daily = (first.groupby(["team_id", "inning", "game_date", "batting_order_slot"])
             .size().rename("n").reset_index())
    dates = first[["team_id", "game_date"]].drop_duplicates()
    rows = []
    for team, td in dates.groupby("team_id", sort=False):
        hist = daily[daily["team_id"] == team]
        for date in sorted(td["game_date"].dropna().unique()):
            date = pd.Timestamp(date)
            for inn in range(1, 10):
                h = hist[(hist["inning"] == inn) & (hist["game_date"] < date)]
                counts = h.groupby("batting_order_slot")["n"].sum() if len(h) else pd.Series(dtype=float)
                n = int(counts.sum())
                rec = {
                    "team_id": team,
                    "as_of_date": date,
                    "inning": inn,
                    "prior_team_inning_games": n,
                    "reliability": 1.0 if n > 0 else 0.0,
                    "source_class": "raw_empirical_outcomes_strictly_prior_date_no_smoothing",
                }
                if n:
                    probs = {s: float(counts.get(s, 0)) / n for s in range(1, 10)}
                    rec["expected_start_slot"] = sum(s * probs[s] for s in range(1, 10))
                    for s in range(1, 10):
                        rec[f"p_start_slot_{s}"] = probs[s]
                else:
                    rec["expected_start_slot"] = np.nan
                    for s in range(1, 10):
                        rec[f"p_start_slot_{s}"] = np.nan
                rows.append(rec)
    return pd.DataFrame(rows)


def build_transitions(first: pd.DataFrame) -> pd.DataFrame:
    x = first.sort_values(["game_id", "team_id", "inning"]).copy()
    x["prev_inning"] = x.groupby(["game_id", "team_id"])["inning"].shift(1)
    x["prev_start_slot"] = x.groupby(["game_id", "team_id"])["batting_order_slot"].shift(1)
    x = x[(x["inning"] >= 2) & (x["prev_inning"] == x["inning"] - 1)].copy()
    x["prev_start_slot"] = x["prev_start_slot"].astype(int)
    g = (x.groupby(["inning", "prev_start_slot", "batting_order_slot"])
         .size().rename("n").reset_index())
    g["sample_n"] = g.groupby(["inning", "prev_start_slot"])["n"].transform("sum")
    g["probability"] = g["n"] / g["sample_n"]
    return g


def walkforward_validate(first: pd.DataFrame) -> pd.DataFrame:
    """Forward season validation of raw empirical inning distributions.

    Evaluation uses a tiny numerical floor only to make log loss finite when a
    previously unseen slot occurs. The floor is NOT used in production feature
    probabilities and is reported explicitly.
    """
    eps = 1e-12
    seasons = sorted(int(s) for s in first["season"].dropna().unique())
    rows = []
    for test_season in seasons[1:]:
        train = first[first["season"] < test_season]
        test = first[first["season"] == test_season]
        for inn in range(1, 10):
            tr = train[train["inning"] == inn]
            te = test[test["inning"] == inn]
            if tr.empty or te.empty:
                continue
            c = tr["batting_order_slot"].value_counts()
            probs = {s: float(c.get(s, 0)) / len(tr) for s in range(1, 10)}
            y = te["batting_order_slot"].astype(int).to_numpy()
            pp = np.array([max(probs[int(s)], eps) for s in y], dtype=float)
            empirical_logloss = float(-np.log(pp).mean())
            uniform_logloss = float(np.log(9.0))
            empirical_brier = float(np.mean([
                sum((probs[s] - (1.0 if s == int(obs) else 0.0))**2 for s in range(1,10))
                for obs in y
            ]))
            uniform_brier = float(8.0/9.0)
            rows.append({
                "train_through_season": test_season - 1,
                "test_season": test_season,
                "inning": inn,
                "train_n": int(len(tr)),
                "test_n": int(len(te)),
                "empirical_log_loss": empirical_logloss,
                "uniform_log_loss": uniform_logloss,
                "log_loss_improvement": uniform_logloss - empirical_logloss,
                "empirical_brier": empirical_brier,
                "uniform_brier": uniform_brier,
                "brier_improvement": uniform_brier - empirical_brier,
                "evaluation_probability_floor": eps,
            })
    return pd.DataFrame(rows)


def build_stability(overall: pd.DataFrame, by_season: pd.DataFrame) -> pd.DataFrame:
    ref = overall[["inning", "batting_order_slot", "probability"]].rename(columns={"probability":"full_probability"})
    x = by_season.merge(ref, on=["inning", "batting_order_slot"], how="left")
    x["abs_probability_drift"] = (x["probability"] - x["full_probability"]).abs()
    rows = []
    for (season, inn), g in x.groupby(["season", "inning"]):
        rows.append({
            "season": int(season),
            "inning": int(inn),
            "sample_n": int(g["sample_n"].max()),
            "max_abs_probability_drift": float(g["abs_probability_drift"].max()),
            "mean_abs_probability_drift": float(g["abs_probability_drift"].mean()),
            "l1_distribution_distance": float(g["abs_probability_drift"].sum()),
        })
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

    first = inning_start_observations(pa, lineups)
    order = build_order_path_asof(first)
    overall = empirical_distribution(first, ["inning"])
    by_season = empirical_distribution(first, ["season", "inning"])
    transitions = build_transitions(first)
    walkforward = walkforward_validate(first)
    stability = build_stability(overall, by_season)

    classified = classify_relief(pa, starters)
    bp_pitcher = build_bullpen_pitcher(classified, metrics, a.shrink_strength)
    bp_team = build_bullpen_team(classified, metrics, a.shrink_strength * 2)

    order.to_parquet(a.output_dir / "batting_order_path_asof.parquet", index=False)
    overall.to_parquet(a.output_dir / "batting_order_empirical_all_innings.parquet", index=False)
    by_season.to_parquet(a.output_dir / "batting_order_empirical_by_season.parquet", index=False)
    transitions.to_parquet(a.output_dir / "batting_order_transition_empirical.parquet", index=False)
    walkforward.to_parquet(a.output_dir / "batting_order_walkforward_validation.parquet", index=False)
    stability.to_parquet(a.output_dir / "batting_order_stability.parquet", index=False)
    bp_pitcher.to_parquet(a.output_dir / "bullpen_pitcher_asof.parquet", index=False)
    bp_team.to_parquet(a.output_dir / "bullpen_team_asof.parquet", index=False)

    manifest = {
        "strict_cutoff": "source game_date < target game_date",
        "same_day_prior_games_included": False,
        "market_data_used": False,
        "pregame_identity_claimed": False,
        "batting_order_smoothing_used": False,
        "batting_order_slot_source": "chronological team PA ordinal modulo 9; substitution-safe",
        "batting_order_method": "raw empirical all-inning start-slot and transition frequencies",
        "batting_order_validation": "Wilson intervals + season stability + forward-season log-loss/Brier validation",
        "outputs": {
            "batting_order_path_rows": int(len(order)),
            "batting_order_empirical_rows": int(len(overall)),
            "batting_order_by_season_rows": int(len(by_season)),
            "batting_order_transition_rows": int(len(transitions)),
            "batting_order_walkforward_rows": int(len(walkforward)),
            "batting_order_stability_rows": int(len(stability)),
            "bullpen_pitcher_rows": int(len(bp_pitcher)),
            "bullpen_team_rows": int(len(bp_team)),
        },
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
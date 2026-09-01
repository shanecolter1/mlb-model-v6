#!/usr/bin/env python3
"""Build the first integrated I1-I9 matchup-state residual challenger.

Development only. 2025 is never loaded.

This challenger combines the already validated development layers:
- M0 raw empirical opening-total x inning prior;
- M1 365-day batter/pitcher K, BB/HBP, HR and non-HR-hit skill dimensions;
- M2/M3 joint starter-state x batting-start-slot probabilities;
- M3 unconditional inning start-slot distributions (transition propagation was
  tested and rejected as a wholesale replacement);
- M4 probability-weighted bullpen skill mixtures using the inning-specific
  windows/blends selected by the bullpen skill-fidelity validation.

The internal unit is a played half inning. For each possible batting start slot,
actual historical lineup identity is reconstructed from the first batter to
occupy each modulo-nine slot in that game (Tier-B retrospective identity). An
empirical prior-season PA-exposure distribution P(PA slot | inning,start slot)
then maps each possible start state to a probability-weighted batter skill
vector without assuming an arbitrary number of hitters.

The pitcher object is the first-pitcher state for this initial integrated test:
starter skill if the joint state says starter; bullpen expected skill otherwise.
No assumption about a mid-inning pitching-change rate is introduced here.

Full-inning matchup features are the symmetric sums of the corresponding top and
bottom half features. The residual model uses M0 as a logit offset. Features are
training-fold standardized and centered, so zero residual corresponds to an
average matchup. Ridge strength is selected only on 2023/2024 development folds.
2025 is untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["k", "baserunner", "hr", "nonhr_hit"]
EVENTS = {
    "k": {"strikeout"},
    "baserunner": {"walk", "hit_by_pitch"},
    "hr": {"home_run"},
    "nonhr_hit": {"single", "double", "triple"},
}
INTERACTION_METRICS = {"k", "baserunner", "hr"}
RIDGES = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
TEST_YEARS = [2023, 2024]
EPS = 1e-8


def read(path: Path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)


def prep_pa(pa: pd.DataFrame) -> pd.DataFrame:
    x = pa.copy()
    x["game_date"] = pd.to_datetime(x.game_date, errors="coerce").dt.normalize()
    x["season"] = x.game_date.dt.year.astype("Int64")
    x["inning"] = pd.to_numeric(x.inning, errors="coerce").astype("Int64")
    x = x[x.inning.between(1, 9)].copy()
    x = x.sort_values(["game_id", "batting_team_id", "play_index"], kind="stable")
    x["team_pa_ordinal"] = x.groupby(["game_id", "batting_team_id"]).cumcount()
    x["batting_order_slot"] = (x.team_pa_ordinal % 9) + 1
    # Realized inning start slot, used only to build prior-season exposure tables.
    first = (x.sort_values(["game_id", "batting_team_id", "inning", "play_index"], kind="stable")
             .groupby(["game_id", "batting_team_id", "inning"], as_index=False)
             .first()[["game_id", "batting_team_id", "inning", "batting_order_slot"]]
             .rename(columns={"batting_order_slot": "start_slot_realized"}))
    x = x.merge(first, on=["game_id", "batting_team_id", "inning"], how="left", validate="many_to_one")
    return x


def lineup_identity(pa: pd.DataFrame) -> pd.DataFrame:
    # First realized occupant of each modulo-nine slot. This is a retrospective
    # research identity, not a verified historical pregame timestamp claim.
    z = (pa.sort_values(["game_id", "batting_team_id", "play_index"], kind="stable")
         .groupby(["game_id", "batting_team_id", "batting_order_slot"], as_index=False)
         .first())
    return z[["game_id", "game_date", "season", "batting_team_id", "batting_order_slot", "batter_id"]]


def daily_player_rates(pa: pd.DataFrame, id_col: str) -> pd.DataFrame:
    x = pa.dropna(subset=[id_col, "game_date"]).copy()
    ev = x.event.astype(str)
    for m, vals in EVENTS.items():
        x[f"n_{m}"] = ev.isin(vals).astype("int16")
    cols = [f"n_{m}" for m in METRICS]
    g = x.groupby([id_col, "game_date"])[cols].sum().reset_index()
    n = x.groupby([id_col, "game_date"]).size().rename("pa").reset_index()
    return g.merge(n, on=[id_col, "game_date"], how="inner")


def make_rate_index(daily: pd.DataFrame, id_col: str):
    out = {}
    cols = ["pa"] + [f"n_{m}" for m in METRICS]
    for pid, g in daily.groupby(id_col, sort=False):
        g = g.sort_values("game_date")
        dates = g.game_date.to_numpy(dtype="datetime64[ns]")
        vals = g[cols].to_numpy(float)
        cs = np.vstack([np.zeros((1, len(cols))), np.cumsum(vals, axis=0)])
        out[pid] = (dates, cs)
    return out


def rates_asof(index, pid, date):
    rec = index.get(pid)
    if rec is None or pd.isna(pid):
        return None
    dates, cs = rec
    d = np.datetime64(pd.Timestamp(date).to_datetime64())
    lo = d - np.timedelta64(365, "D")
    a = int(np.searchsorted(dates, lo, side="left"))
    b = int(np.searchsorted(dates, d, side="left"))
    v = cs[b] - cs[a]
    if v[0] <= 0:
        return None
    return np.array([v[i + 1] / v[0] for i in range(len(METRICS))], float)


def starter_map(starters: pd.DataFrame) -> dict:
    s = starters.copy()
    # normalized starter table uses team_id/pitcher_id in the reusable feed.
    return {(r.game_id, r.team_id): r.pitcher_id for r in s.itertuples() if pd.notna(r.pitcher_id)}


def classify_relief(pa: pd.DataFrame, starters: pd.DataFrame) -> pd.DataFrame:
    sm = starters[["game_id", "team_id", "pitcher_id"]].dropna(subset=["pitcher_id"]).rename(
        columns={"team_id": "pitching_team_id", "pitcher_id": "starter_id"}
    )
    x = pa.merge(sm, on=["game_id", "pitching_team_id"], how="left", validate="many_to_one")
    x["is_relief"] = x.pitcher_id != x.starter_id
    return x


def bullpen_history(pa: pd.DataFrame, starters: pd.DataFrame):
    x = classify_relief(pa, starters)
    r = x[x.is_relief].copy()
    bf = (r.groupby(["pitching_team_id", "pitcher_id", "game_date"], as_index=False)
          .size().rename(columns={"size": "relief_pa"}))
    # First reliever appearing in an inning, whether or not the starter began it.
    first = (r.sort_values(["game_id", "inning", "pitching_team_id", "play_index"], kind="stable")
             .groupby(["game_id", "game_date", "pitching_team_id", "inning"], as_index=False)
             .first())
    fi = (first.groupby(["pitching_team_id", "pitcher_id", "game_date", "inning"], as_index=False)
          .size().rename(columns={"size": "first_count"}))
    return bf, fi


def bullpen_skill(team, date, inning, window, alpha, bf_groups, fi_groups, pitcher_index):
    hb = bf_groups.get(team)
    if hb is None:
        return None, 0.0, 0
    lo = date - pd.Timedelta(days=int(window))
    h = hb[(hb.game_date >= lo) & (hb.game_date < date)]
    if h.empty:
        return None, 0.0, 0
    g = h.groupby("pitcher_id", as_index=False).relief_pa.sum().rename(columns={"relief_pa": "global_bf"})
    hf = fi_groups.get(team)
    if hf is not None:
        q = hf[(hf.game_date >= lo) & (hf.game_date < date) & (hf.inning == inning)]
    else:
        q = None
    if q is not None and len(q):
        q = q.groupby("pitcher_id", as_index=False).first_count.sum().rename(columns={"first_count": "inning_first"})
        g = g.merge(q, on="pitcher_id", how="left")
    else:
        g["inning_first"] = 0.0
    g["inning_first"] = g.inning_first.fillna(0.0)
    gs = float(g.global_bf.sum())
    ins = float(g.inning_first.sum())
    glob = g.global_bf.to_numpy(float) / gs if gs > 0 else np.zeros(len(g))
    inn = g.inning_first.to_numpy(float) / ins if ins > 0 else glob
    w = (1.0 - alpha) * glob + alpha * inn
    skills = []
    valid_w = []
    for ww, pid in zip(w, g.pitcher_id):
        v = rates_asof(pitcher_index, pid, date)
        if v is not None and np.isfinite(v).all() and ww > 0:
            skills.append(v)
            valid_w.append(float(ww))
    coverage = float(sum(valid_w))
    if not skills or coverage <= 0:
        return None, coverage, int(len(g))
    ww = np.asarray(valid_w, float) / coverage
    arr = np.vstack(skills)
    return (arr * ww[:, None]).sum(axis=0), coverage, int(len(g))


def exposure_tables(pa: pd.DataFrame):
    # Prior-season empirical PA-slot composition conditional on destination inning
    # and realized start slot. The object is a 9-vector summing to one.
    out = {}
    for year in [2022, 2023, 2024]:
        tr = pa[pa.season < year]
        for inn in range(1, 10):
            z = tr[tr.inning == inn]
            fallback = z.batting_order_slot.value_counts()
            fallback = np.array([float(fallback.get(s, 0)) for s in range(1, 10)], float)
            fallback = fallback / fallback.sum() if fallback.sum() else np.full(9, 1 / 9)
            for start in range(1, 10):
                q = z[z.start_slot_realized == start].batting_order_slot.value_counts()
                v = np.array([float(q.get(s, 0)) for s in range(1, 10)], float)
                # Missing support falls back to the same-inning prior-season PA composition.
                v = v / v.sum() if v.sum() else fallback
                out[(year, inn, start)] = v
    return out


def build_half_features(half, joint, pa, starters, bpbest):
    h = half[half.half_played.astype(bool) & half.season.isin([2022, 2023, 2024])].copy()
    jcols = [c for c in joint.columns if c.startswith("p_slot") and (c.endswith("_starter") or c.endswith("_bullpen"))]
    jk = ["game_id", "season", "inning", "batting_team_id", "pitching_team_id"]
    j = joint[jk + jcols].copy()
    h = h.merge(j, on=jk, how="left", validate="one_to_one")

    line = lineup_identity(pa)
    batter_daily = daily_player_rates(pa, "batter_id")
    pitcher_daily = daily_player_rates(pa, "pitcher_id")
    bidx = make_rate_index(batter_daily, "batter_id")
    pidx = make_rate_index(pitcher_daily, "pitcher_id")
    smap = starter_map(starters)
    exposures = exposure_tables(pa)

    # Cache lineup player skill by game/team/slot.
    lineup_skill = {}
    for r in line.itertuples():
        v = rates_asof(bidx, r.batter_id, r.game_date)
        lineup_skill[(r.game_id, r.batting_team_id, int(r.batting_order_slot))] = v

    bf, fi = bullpen_history(pa, starters)
    bf_groups = {k: g.copy() for k, g in bf.groupby("pitching_team_id", sort=False)}
    fi_groups = {k: g.copy() for k, g in fi.groupby("pitching_team_id", sort=False)}
    specs = {int(r.inning): (int(r.window), float(r.alpha_inning_usage)) for r in bpbest.itertuples()}

    rows = []
    for r in h.itertuples():
        year = int(r.season); inn = int(r.inning); date = pd.Timestamp(r.game_date)
        starter_id = smap.get((r.game_id, r.pitching_team_id))
        starter_skill = rates_asof(pidx, starter_id, date)
        if inn == 1:
            # I1 bullpen branch is structurally negligible but can occur for openers/injury.
            # Use the I2 validated bullpen mixture spec rather than inventing an I1 tuning value.
            bwin, balpha = specs.get(2, (30, 0.25))
        else:
            bwin, balpha = specs[inn]
        bp_skill, bp_cov, bp_n = bullpen_skill(r.pitching_team_id, date, inn, bwin, balpha, bf_groups, fi_groups, pidx)

        feat = {m: {"batter": 0.0, "pitcher": 0.0, "interaction": 0.0} for m in METRICS}
        mass_used = 0.0
        for start in range(1, 10):
            exposure = exposures[(year, inn, start)]
            bskills = []
            bw = []
            for slot in range(1, 10):
                v = lineup_skill.get((r.game_id, r.batting_team_id, slot))
                if v is not None and np.isfinite(v).all() and exposure[slot - 1] > 0:
                    bskills.append(v); bw.append(exposure[slot - 1])
            if not bskills or sum(bw) <= 0:
                continue
            bw = np.asarray(bw, float); bw = bw / bw.sum()
            bskill = (np.vstack(bskills) * bw[:, None]).sum(axis=0)
            ps = float(getattr(r, f"p_slot{start}_starter")) if pd.notna(getattr(r, f"p_slot{start}_starter")) else 0.0
            pb = float(getattr(r, f"p_slot{start}_bullpen")) if pd.notna(getattr(r, f"p_slot{start}_bullpen")) else 0.0
            if starter_skill is not None and np.isfinite(starter_skill).all() and ps > 0:
                for k, m in enumerate(METRICS):
                    feat[m]["batter"] += ps * bskill[k]
                    feat[m]["pitcher"] += ps * starter_skill[k]
                    if m in INTERACTION_METRICS:
                        feat[m]["interaction"] += ps * bskill[k] * starter_skill[k]
                mass_used += ps
            if bp_skill is not None and np.isfinite(bp_skill).all() and pb > 0:
                for k, m in enumerate(METRICS):
                    feat[m]["batter"] += pb * bskill[k]
                    feat[m]["pitcher"] += pb * bp_skill[k]
                    if m in INTERACTION_METRICS:
                        feat[m]["interaction"] += pb * bskill[k] * bp_skill[k]
                mass_used += pb
        rec = {
            "game_id": r.game_id, "game_date": date, "season": year, "inning": inn,
            "half": r.half, "batting_team_id": r.batting_team_id, "pitching_team_id": r.pitching_team_id,
            "dk_total_open_total": float(r.dk_total_open_total), "runs_half": float(r.runs_half),
            "matchup_state_mass_with_skill": mass_used, "bullpen_skill_mass_coverage": bp_cov,
            "bullpen_candidate_n": bp_n, "bullpen_window": bwin, "bullpen_alpha": balpha,
            "participant_identity_class": "retrospective_realized_lineup_and_starter_identity_Tier_B_unverified_pregame",
        }
        if mass_used > 0:
            for m in METRICS:
                rec[f"expected_batter_{m}"] = feat[m]["batter"] / mass_used
                rec[f"expected_pitcher_{m}"] = feat[m]["pitcher"] / mass_used
                if m in INTERACTION_METRICS:
                    rec[f"expected_interaction_{m}"] = feat[m]["interaction"] / mass_used
        rows.append(rec)
    return pd.DataFrame(rows)


def full_inning_matrix(hf: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in hf.columns if c.startswith("expected_")]
    keys = ["game_id", "game_date", "season", "inning", "dk_total_open_total"]
    top = hf[hf.half == "top"][keys + ["runs_half"] + features].copy()
    bot = hf[hf.half == "bottom"][keys + ["runs_half"] + features].copy()
    x = top.merge(bot, on=keys, suffixes=("_top", "_bottom"), how="inner", validate="one_to_one")
    x["full_inning_runs"] = x.runs_half_top + x.runs_half_bottom
    x["any_run"] = (x.full_inning_runs >= 1).astype(int)
    for c in features:
        x[c] = pd.to_numeric(x[f"{c}_top"], errors="coerce") + pd.to_numeric(x[f"{c}_bottom"], errors="coerce")
    return x[keys + ["full_inning_runs", "any_run"] + features]


def prior_m0_probability(all_half: pd.DataFrame, target_year: int) -> pd.DataFrame:
    x = all_half[(all_half.season < target_year) & all_half.half_played.astype(bool)].copy()
    p = x.pivot(index=["game_id", "inning", "dk_total_open_total"], columns="half", values="runs_half").reset_index()
    p = p.dropna(subset=["top", "bottom"])
    p["any_run"] = ((p.top + p.bottom) >= 1).astype(int)
    return (p.groupby(["dk_total_open_total", "inning"], as_index=False).any_run.mean()
            .rename(columns={"any_run": "m0_p_any"}))


def attach_oos_m0(full: pd.DataFrame, all_half: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for year in [2022, 2023, 2024]:
        z = full[full.season == year].copy()
        b = prior_m0_probability(all_half, year)
        z = z.merge(b, on=["dk_total_open_total", "inning"], how="left", validate="many_to_one")
        parts.append(z)
    return pd.concat(parts, ignore_index=True)


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    z = np.clip(np.asarray(z, float), -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS); y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def fit_offset_ridge(X, y, offset, ridge):
    beta = np.zeros(X.shape[1], float)
    eye = np.eye(X.shape[1])
    for _ in range(60):
        p = sigmoid(offset + X @ beta)
        w = np.maximum(p * (1 - p), 1e-7)
        grad = X.T @ (y - p) - ridge * beta
        H = (X.T * w) @ X + ridge * eye + 1e-9 * eye
        step = np.linalg.solve(H, grad)
        beta2 = beta + step
        if np.max(np.abs(step)) < 1e-8:
            beta = beta2; break
        beta = beta2
    return beta


def validate_residual(full: pd.DataFrame):
    feature_cols = [c for c in full.columns if c.startswith("expected_")]
    rows = []
    coeff_rows = []
    for year in TEST_YEARS:
        tr = full[(full.season < year) & (full.season >= 2022)].copy()
        te = full[full.season == year].copy()
        needed = feature_cols + ["m0_p_any"]
        tr = tr.dropna(subset=needed); te = te.dropna(subset=needed)
        if len(tr) < 500 or len(te) < 500:
            continue
        mu = tr[feature_cols].mean().to_numpy(float)
        sd = tr[feature_cols].std(ddof=0).to_numpy(float); sd[sd < 1e-9] = 1.0
        Xtr = (tr[feature_cols].to_numpy(float) - mu) / sd
        Xte = (te[feature_cols].to_numpy(float) - mu) / sd
        ytr = tr.any_run.to_numpy(float); yte = te.any_run.to_numpy(float)
        otr = logit(tr.m0_p_any.to_numpy(float)); ote = logit(te.m0_p_any.to_numpy(float))
        p0 = sigmoid(ote)
        for ridge in RIDGES:
            beta = fit_offset_ridge(Xtr, ytr, otr, ridge)
            p = sigmoid(ote + Xte @ beta)
            rec = {
                "test_year": year, "ridge": ridge, "n_train": len(tr), "n_test": len(te),
                "feature_count": len(feature_cols), "m0_logloss": logloss(yte, p0),
                "matchup_logloss": logloss(yte, p), "logloss_improvement": logloss(yte, p0) - logloss(yte, p),
                "m0_brier": brier(yte, p0), "matchup_brier": brier(yte, p),
                "brier_improvement": brier(yte, p0) - brier(yte, p),
            }
            rows.append(rec)
            for c, b in zip(feature_cols, beta):
                coeff_rows.append({"test_year": year, "ridge": ridge, "feature": c, "standardized_coefficient": float(b)})
    folds = pd.DataFrame(rows)
    coeffs = pd.DataFrame(coeff_rows)
    summary = (folds.groupby("ridge", as_index=False).agg(
        mean_logloss_improvement=("logloss_improvement", "mean"),
        worst_year_logloss_improvement=("logloss_improvement", "min"),
        mean_brier_improvement=("brier_improvement", "mean"),
        worst_year_brier_improvement=("brier_improvement", "min"),
    ))
    summary["all_years_logloss_positive"] = summary.worst_year_logloss_improvement > 0
    summary["all_years_brier_positive"] = summary.worst_year_brier_improvement > 0
    best = summary.sort_values(["mean_logloss_improvement", "worst_year_logloss_improvement"], ascending=False).head(1)
    return folds, summary, best, coeffs, feature_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate-appearances", type=Path, required=True)
    ap.add_argument("--starters", type=Path, required=True)
    ap.add_argument("--half-matrix", type=Path, required=True)
    ap.add_argument("--joint-state", type=Path, required=True)
    ap.add_argument("--bullpen-best", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)

    pa = prep_pa(read(a.plate_appearances)); starters = read(a.starters)
    half = read(a.half_matrix); joint = read(a.joint_state); bpbest = read(a.bullpen_best)
    for d in [pa, half, joint]:
        if "season" in d.columns: d["season"] = pd.to_numeric(d.season, errors="coerce").astype("Int64")
    if (pa.season >= 2025).any() or (half.season >= 2025).any() or (joint.season >= 2025).any():
        raise RuntimeError("2025 holdout leakage")
    if set(pa.season.dropna().astype(int).unique()) != {2021, 2022, 2023, 2024}:
        raise RuntimeError("PA development seasons must be exactly 2021-2024")

    hf = build_half_features(half, joint, pa, starters, bpbest)
    full = full_inning_matrix(hf)
    full = attach_oos_m0(full, half)
    folds, summary, best, coeffs, feature_cols = validate_residual(full)

    hf.to_parquet(a.output_dir / "m4_integrated_half_matchup_features.parquet", index=False)
    full.to_parquet(a.output_dir / "m4_integrated_full_inning_features.parquet", index=False)
    folds.to_csv(a.output_dir / "m4_integrated_residual_folds.csv", index=False)
    summary.to_csv(a.output_dir / "m4_integrated_residual_ridge_summary.csv", index=False)
    best.to_csv(a.output_dir / "m4_integrated_residual_best.csv", index=False)
    coeffs.to_csv(a.output_dir / "m4_integrated_residual_coefficients.csv", index=False)

    best_rec = best.iloc[0].to_dict() if len(best) else None
    manifest = {
        "status": "PASS",
        "architecture": "M4_integrated_joint_matchup_first_pitcher_residual_vs_M0",
        "development_seasons_loaded": [2021, 2022, 2023, 2024],
        "residual_test_years": TEST_YEARS,
        "holdout_season": 2025,
        "holdout_opened": False,
        "m0_method": "raw prior-season empirical opening-total x inning P(any run), evaluation floor only",
        "m1_skill_window_days": 365,
        "m1_skill_dimensions": METRICS,
        "interaction_dimensions": sorted(INTERACTION_METRICS),
        "m3_path_representation": "unconditional prior-season inning start-slot distribution; recursive transition challenger rejected",
        "batter_sequence_weighting": "prior-season empirical PA-slot composition conditional on inning and start slot",
        "pitcher_representation": "joint first-pitcher starter/bullpen state; no mid-inning change assumption",
        "bullpen_representation": "probability-weighted prior-usage skill mixture using empirically selected inning specs",
        "participant_identity_class": "retrospective_realized_lineup_and_starter_identity_Tier_B_unverified_pregame",
        "full_inning_feature_aggregation": "symmetric top+bottom sums",
        "feature_standardization": "training-fold mean/SD",
        "residual_centering": "no fitted intercept; standardized average matchup has zero residual",
        "ridge_candidates": RIDGES,
        "feature_columns": feature_cols,
        "half_feature_rows": int(len(hf)),
        "full_inning_rows": int(len(full)),
        "best_development_spec": best_rec,
        "market_data_used": False,
        "only_market_context": "isolated dk_total_open_total for M0 baseline cell",
        "automatic_production_promotion": False,
        "note": "First integrated matchup residual gate. A positive result supports proceeding to discrete 0/1/2/3/4+ modeling; a negative result diagnoses state/aggregation construction without invalidating M1-M3 identification layers."
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda z: float(z) if hasattr(z, "item") else str(z)), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=lambda z: float(z) if hasattr(z, "item") else str(z)))


if __name__ == "__main__":
    main()

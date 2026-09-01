#!/usr/bin/env python3
"""Execution wrapper for the integrated M4 matchup residual validation.

This wrapper makes two implementation-only changes to the governing validator:
1. fixes pandas 3 read-only ndarray behavior by copying the training SD vector;
2. replaces repeated pandas bullpen-history filtering with mathematically
   equivalent cumulative history indexes and caches.

No research specification, data window, feature, candidate, or model-selection
rule is changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.analysis.all_inning import m4_integrated_matchup_residual_validation as base

_BF_INDEX = None
_FI_INDEX = None
_GLOBAL_CACHE = {}


def _cum_index(groups, value_col, include_inning=False):
    out = {}
    for team, g in groups.items():
        td = {}
        keys = ["pitcher_id", "inning"] if include_inning else ["pitcher_id"]
        for key, q in g.groupby(keys, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            q = q.sort_values("game_date")
            dates = q.game_date.to_numpy(dtype="datetime64[ns]")
            vals = q[value_col].to_numpy(float)
            cs = np.concatenate([[0.0], np.cumsum(vals)])
            td[key] = (dates, cs)
        out[team] = td
    return out


def _ensure_indexes(bf_groups, fi_groups):
    global _BF_INDEX, _FI_INDEX
    if _BF_INDEX is None:
        _BF_INDEX = _cum_index(bf_groups, "relief_pa", include_inning=False)
        _FI_INDEX = _cum_index(fi_groups, "first_count", include_inning=True)


def _window_sum(rec, date, days):
    if rec is None:
        return 0.0
    dates, cs = rec
    d = np.datetime64(pd.Timestamp(date).to_datetime64())
    lo = d - np.timedelta64(int(days), "D")
    a = int(np.searchsorted(dates, lo, side="left"))
    b = int(np.searchsorted(dates, d, side="left"))
    return float(cs[b] - cs[a])


def fast_bullpen_skill(team, date, inning, window, alpha, bf_groups, fi_groups, pitcher_index):
    _ensure_indexes(bf_groups, fi_groups)
    d = pd.Timestamp(date)
    cache_key = (team, int(d.value), int(window), id(pitcher_index))
    rec = _GLOBAL_CACHE.get(cache_key)
    if rec is None:
        team_bf = _BF_INDEX.get(team, {})
        pids = []
        counts = []
        skills = []
        for (pid,), hist in team_bf.items():
            n = _window_sum(hist, d, window)
            if n <= 0:
                continue
            pids.append(pid)
            counts.append(n)
            skills.append(base.rates_asof(pitcher_index, pid, d))
        if not pids:
            rec = ([], np.array([], float), [])
        else:
            c = np.asarray(counts, float)
            rec = (pids, c / c.sum(), skills)
        _GLOBAL_CACHE[cache_key] = rec
    pids, global_share, skills = rec
    if not pids:
        return None, 0.0, 0

    team_fi = _FI_INDEX.get(team, {})
    first = np.array([
        _window_sum(team_fi.get((pid, int(inning))), d, window) for pid in pids
    ], float)
    inning_share = first / first.sum() if first.sum() > 0 else global_share
    weights = (1.0 - float(alpha)) * global_share + float(alpha) * inning_share

    valid_idx = [i for i, v in enumerate(skills) if v is not None and np.isfinite(v).all() and weights[i] > 0]
    if not valid_idx:
        return None, 0.0, len(pids)
    coverage = float(weights[valid_idx].sum())
    w = weights[valid_idx] / coverage
    arr = np.vstack([skills[i] for i in valid_idx])
    return (arr * w[:, None]).sum(axis=0), coverage, len(pids)


def fixed_validate_residual(full: pd.DataFrame):
    feature_cols = [c for c in full.columns if c.startswith("expected_")]
    rows = []
    coeff_rows = []
    for year in base.TEST_YEARS:
        tr = full[(full.season < year) & (full.season >= 2022)].copy()
        te = full[full.season == year].copy()
        needed = feature_cols + ["m0_p_any"]
        tr = tr.dropna(subset=needed); te = te.dropna(subset=needed)
        if len(tr) < 500 or len(te) < 500:
            continue
        mu = tr[feature_cols].mean().to_numpy(float).copy()
        sd = tr[feature_cols].std(ddof=0).to_numpy(float).copy()
        sd[sd < 1e-9] = 1.0
        Xtr = (tr[feature_cols].to_numpy(float) - mu) / sd
        Xte = (te[feature_cols].to_numpy(float) - mu) / sd
        ytr = tr.any_run.to_numpy(float); yte = te.any_run.to_numpy(float)
        otr = base.logit(tr.m0_p_any.to_numpy(float)); ote = base.logit(te.m0_p_any.to_numpy(float))
        p0 = base.sigmoid(ote)
        for ridge in base.RIDGES:
            beta = base.fit_offset_ridge(Xtr, ytr, otr, ridge)
            p = base.sigmoid(ote + Xte @ beta)
            rec = {
                "test_year": year, "ridge": ridge, "n_train": len(tr), "n_test": len(te),
                "feature_count": len(feature_cols), "m0_logloss": base.logloss(yte, p0),
                "matchup_logloss": base.logloss(yte, p), "logloss_improvement": base.logloss(yte, p0) - base.logloss(yte, p),
                "m0_brier": base.brier(yte, p0), "matchup_brier": base.brier(yte, p),
                "brier_improvement": base.brier(yte, p0) - base.brier(yte, p),
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


base.bullpen_skill = fast_bullpen_skill
base.validate_residual = fixed_validate_residual

if __name__ == "__main__":
    base.main()

#!/usr/bin/env python3
"""Test jointly useful Statcast families after the marginal M1 screen.

The marginal screen fixes each family's window and ridge on 2022. This gate
uses 2023 to select an event-specific subset from only the families that passed
the marginal 2022/2023/2024 sign gate, then confirms that fixed subset on 2024.
Every subset for an event is scored on the same complete-case rows. The locked
365-day M1 event-rate model remains the comparator, and 2025 is never loaded.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.all_inning import m1_statcast_family_incremental_validation as base

EPS = 1e-10


def fit_logistic(X, y, penalty, max_iter=30):
    beta = np.zeros(X.shape[1], dtype=float)
    penalty = np.asarray(penalty, dtype=float)
    if len(penalty) != X.shape[1]:
        raise ValueError("penalty vector does not match design")
    for _ in range(max_iter):
        p = base.sigmoid(X @ beta)
        w = np.maximum(p * (1 - p), 1e-7)
        grad = X.T @ (p - y) + penalty * beta
        hessian = X.T @ (X * w[:, None]) + np.diag(penalty) + 1e-8 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ grad
        updated = beta - step
        if np.max(np.abs(updated - beta)) < 1e-7:
            beta = updated
            break
        beta = updated
    return beta


def family_specifications(selected, event):
    # These specifications were selected on 2022 only. Do not filter them with
    # the later 2023/2024 marginal results before the independent joint gate.
    specs = selected[selected.event == event].copy()
    if specs.family.duplicated().any():
        raise RuntimeError(f"duplicate family specifications for {event}")
    return {
        str(r.family): {"window": str(r.window), "ridge": float(r.ridge)}
        for r in specs.itertuples()
    }


def prepare_event_year(x, event, specs, year):
    ycol = f"y_{event}"
    families = sorted(specs)
    needed = ["season", "inning", "batter_side", "pitcher_hand", ycol]
    needed += base.core_rate_cols(event)
    for family in families:
        needed += base.family_cols(family, specs[family]["window"])
    needed = list(dict.fromkeys(needed))

    train = x[x.season < year][needed].dropna().copy()
    test = x[x.season == year][needed].dropna().copy()
    if len(train) < 5000 or len(test) < 1000:
        raise RuntimeError(f"insufficient joint complete cases for {event} {year}")

    core_train = base.core_matrix(train, event)
    core_test = base.core_matrix(test, event)
    blocks = {}
    for family in families:
        window = specs[family]["window"]
        cols = base.family_cols(family, window)
        f_train, f_test = base.standardize(train, test, cols)
        if family == "pitch_mix_matchup":
            f_train, f_test = base.add_pitch_mix_interactions(
                train, test, window, f_train, f_test
            )
        blocks[family] = (f_train, f_test)

    y_train = train[ycol].to_numpy(float)
    y_test = test[ycol].to_numpy(float)
    core_beta = fit_logistic(core_train, y_train, np.zeros(core_train.shape[1]))
    core_pred = base.sigmoid(core_test @ core_beta)
    return {
        "event": event,
        "year": year,
        "families": families,
        "specs": specs,
        "core_train": core_train,
        "core_test": core_test,
        "blocks": blocks,
        "y_train": y_train,
        "y_test": y_test,
        "core_logloss": base.ll(y_test, core_pred),
        "core_brier": base.br(y_test, core_pred),
    }


def score_subset(prepared, subset):
    subset = tuple(sorted(subset))
    train_parts = [prepared["core_train"]]
    test_parts = [prepared["core_test"]]
    penalty = [0.0] * prepared["core_train"].shape[1]
    for family in subset:
        f_train, f_test = prepared["blocks"][family]
        train_parts.append(f_train)
        test_parts.append(f_test)
        penalty.extend([prepared["specs"][family]["ridge"]] * f_train.shape[1])
    X_train = np.column_stack(train_parts)
    X_test = np.column_stack(test_parts)
    beta = fit_logistic(X_train, prepared["y_train"], penalty)
    pred = base.sigmoid(X_test @ beta)
    challenger_ll = base.ll(prepared["y_test"], pred)
    challenger_br = base.br(prepared["y_test"], pred)
    return {
        "event": prepared["event"],
        "test_year": prepared["year"],
        "families": "|".join(subset) if subset else "core_only",
        "family_count": len(subset),
        "feature_count": X_train.shape[1] - prepared["core_train"].shape[1],
        "n_train": len(prepared["y_train"]),
        "n_test": len(prepared["y_test"]),
        "core_logloss": prepared["core_logloss"],
        "challenger_logloss": challenger_ll,
        "logloss_improvement": prepared["core_logloss"] - challenger_ll,
        "core_brier": prepared["core_brier"],
        "challenger_brier": challenger_br,
        "brier_improvement": prepared["core_brier"] - challenger_br,
    }


def subsets(families):
    for size in range(len(families) + 1):
        yield from itertools.combinations(families, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-matrix", type=Path, required=True)
    parser.add_argument("--batter-asof", type=Path, required=True)
    parser.add_argument("--pitcher-asof", type=Path, required=True)
    parser.add_argument("--selected-specs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = pd.read_parquet(args.m1_matrix)
    batter = pd.read_parquet(args.batter_asof)
    pitcher = pd.read_parquet(args.pitcher_asof)
    for frame in (x, batter, pitcher):
        frame["game_date"] = pd.to_datetime(frame.game_date, errors="coerce").dt.normalize()
    x["season"] = pd.to_numeric(x.season, errors="raise").astype(int)
    x["inning"] = pd.to_numeric(x.inning, errors="raise").astype(int)
    if set(x.season.unique()) != {2021, 2022, 2023, 2024} or (x.season >= 2025).any():
        raise RuntimeError("development seasons must be exactly 2021-2024")
    if set(x.inning.unique()) != set(range(1, 10)):
        raise RuntimeError("I1-I9 coverage incomplete")
    if "market_data_used" not in x or x.market_data_used.astype(bool).any():
        raise RuntimeError("market isolation audit failed")
    x = x.merge(
        batter, on=["batter_id", "game_date"], how="left", validate="many_to_one"
    ).merge(
        pitcher, on=["pitcher_id", "game_date"], how="left", validate="many_to_one"
    )

    selected = pd.read_csv(args.selected_specs)
    if set(selected.event) != set(base.EVENTS):
        raise RuntimeError("2022 marginal specification screen is missing an event")

    selection_rows = []
    chosen_rows = []
    confirmation_rows = []
    ablation_rows = []
    stable_rows = []
    event_status = []
    for event in base.EVENTS:
        specs = family_specifications(selected, event)
        prepared_2023 = prepare_event_year(x, event, specs, 2023)
        grid = pd.DataFrame(
            [score_subset(prepared_2023, subset) for subset in subsets(sorted(specs))]
        )
        selection_rows.extend(grid.to_dict("records"))
        chosen = grid.sort_values(
            ["challenger_logloss", "challenger_brier", "family_count"],
            ascending=[True, True, True],
        ).iloc[0].to_dict()
        chosen_rows.append(chosen)
        chosen_families = [] if chosen["families"] == "core_only" else chosen["families"].split("|")

        prepared_2024 = prepare_event_year(x, event, specs, 2024)
        confirmation = score_subset(prepared_2024, chosen_families)
        confirmation_rows.append(confirmation)

        # Use 2023 and 2024 only as development/model-selection evidence here.
        # Iteratively remove blocks that fail the predeclared conditional rule;
        # the final independent architecture holdout remains untouched 2025.
        stable_families = list(chosen_families)
        pruning_round = 0
        while stable_families:
            full_by_year = {
                2023: score_subset(prepared_2023, stable_families),
                2024: score_subset(prepared_2024, stable_families),
            }
            failing = []
            for family in stable_families:
                reduced = [f for f in stable_families if f != family]
                rows = []
                for year, prepared in ((2023, prepared_2023), (2024, prepared_2024)):
                    ablated = score_subset(prepared, reduced)
                    full = full_by_year[year]
                    row = {
                        "event": event,
                        "pruning_round": pruning_round,
                        "family_removed": family,
                        "test_year": year,
                        "selected_families": "|".join(stable_families),
                        "ablated_families": ablated["families"],
                        "n_test": full["n_test"],
                        "conditional_logloss_improvement": ablated["challenger_logloss"] - full["challenger_logloss"],
                        "conditional_brier_improvement": ablated["challenger_brier"] - full["challenger_brier"],
                    }
                    rows.append(row)
                    ablation_rows.append(row)
                passed = (
                    rows[0]["conditional_logloss_improvement"] > 0
                    and rows[1]["conditional_logloss_improvement"] > 0
                    and rows[1]["conditional_brier_improvement"] > 0
                )
                if not passed:
                    failing.append(family)
            if not failing:
                break
            stable_families = [f for f in stable_families if f not in failing]
            pruning_round += 1

        stable_specs = {family: specs[family] for family in stable_families}
        stable_prepared_2023 = prepare_event_year(x, event, stable_specs, 2023)
        stable_prepared_2024 = prepare_event_year(x, event, stable_specs, 2024)
        stable_by_year = {
            2023: score_subset(stable_prepared_2023, stable_families),
            2024: score_subset(stable_prepared_2024, stable_families),
        }
        for year, row in stable_by_year.items():
            stable_rows.append({**row, "development_selection_role": "joint_stability"})

        full_confirmed = (
            confirmation["logloss_improvement"] > 0
            and confirmation["brier_improvement"] > 0
        )
        event_status.append(
            {
                "event": event,
                "marginal_candidates": sorted(specs),
                "selected_families_2023": chosen_families,
                "selection_2023_logloss_improvement": chosen["logloss_improvement"],
                "confirmation_2024_logloss_improvement": confirmation["logloss_improvement"],
                "confirmation_2024_brier_improvement": confirmation["brier_improvement"],
                "full_subset_confirmed": bool(full_confirmed),
                "stable_families_2021_2024": stable_families,
                "families_removed_by_conditional_gate": sorted(set(chosen_families) - set(stable_families)),
                "stable_subset_2023_logloss_improvement": stable_by_year[2023]["logloss_improvement"],
                "stable_subset_2024_logloss_improvement": stable_by_year[2024]["logloss_improvement"],
                "stable_subset_2024_brier_improvement": stable_by_year[2024]["brier_improvement"],
                "joint_feature_set_development_selected": bool(
                    stable_families
                    and stable_by_year[2023]["logloss_improvement"] > 0
                    and stable_by_year[2024]["logloss_improvement"] > 0
                    and stable_by_year[2024]["brier_improvement"] > 0
                ),
            }
        )

    selection_df = pd.DataFrame(selection_rows)
    chosen_df = pd.DataFrame(chosen_rows)
    confirmation_df = pd.DataFrame(confirmation_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    stable_df = pd.DataFrame(stable_rows)
    selection_df.to_csv(args.output_dir / "m1_statcast_joint_subset_selection_2023.csv", index=False)
    chosen_df.to_csv(args.output_dir / "m1_statcast_joint_selected_by_event.csv", index=False)
    confirmation_df.to_csv(args.output_dir / "m1_statcast_joint_confirmation_2024.csv", index=False)
    ablation_df.to_csv(args.output_dir / "m1_statcast_joint_conditional_ablation.csv", index=False)
    stable_df.to_csv(args.output_dir / "m1_statcast_joint_stable_subset_metrics.csv", index=False)

    manifest = {
        "status": "PASS",
        "architecture": "M1_joint_statcast_subset_and_conditional_redundancy_gate",
        "development_seasons": [2021, 2022, 2023, 2024],
        "marginal_window_ridge_selection_year": 2022,
        "joint_subset_selection_year": 2023,
        "confirmation_year": 2024,
        "final_feature_selection_years": [2022, 2023, 2024],
        "confirmation_role": "2024 confirms the 2023-selected subset before any conditional pruning",
        "stable_subset_role": "2021-2024 development-selected input for downstream M4; not final-holdout evidence",
        "holdout_season": 2025,
        "holdout_opened": False,
        "market_data_used": False,
        "core_specs": base.CORE_SPECS,
        "comparison_rows": "common complete-case intersection across all eligible families within event/year",
        "subset_rule": "minimum 2023 log loss; Brier then smaller subset are tie-breakers",
        "conditional_rule": "selected family must improve log loss in 2023 and 2024 and Brier in 2024 versus its leave-one-family-out ablation",
        "automatic_production_promotion": False,
        "event_status": event_status,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(confirmation_df.to_string(index=False))
    if len(ablation_df):
        print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Chronologically screen PA-history and reliability families for M1.

The comparator is the retained 365-day event-rate core, validated platoon
terms, and the event-specific stable Statcast family set. Candidate windows,
shrinkage strengths, and incremental ridge penalties are selected on 2022;
2023 selects a joint subset and 2024 supplies development confirmation. The
2025 architecture holdout is never loaded.
"""
from __future__ import annotations

import argparse
import gc
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

from src.analysis.all_inning import m1_statcast_family_incremental_validation as statcast
from src.analysis.all_inning.m1_statcast_joint_redundancy_validation import fit_logistic


EVENTS = ["k", "baserunner", "hr", "nonhr_hit"]
FAMILIES = [
    "target_rate_history",
    "cross_event_profile",
    "outcome_composition",
    "sample_reliability",
]
WINDOWS_BY_FAMILY = {
    "target_rate_history": ["30d", "90d", "365d", "season"],
    "cross_event_profile": ["90d", "365d", "season"],
    "outcome_composition": ["365d", "season"],
    "sample_reliability": ["90d", "365d", "season"],
}
ALPHAS_BY_FAMILY = {
    "target_rate_history": [0.0, 25.0, 100.0, 400.0],
    "cross_event_profile": [0.0, 25.0, 100.0, 400.0],
    "outcome_composition": [0.0, 25.0, 100.0, 400.0],
    "sample_reliability": [25.0, 100.0, 400.0],
}
RIDGES = [0.0, 10.0, 100.0, 1000.0]
BROAD_EVENTS = ["k", "baserunner", "hr", "nonhr_hit"]
COMPOSITION_EVENTS = ["k", "walk", "hbp", "hr", "single", "double", "triple"]


def load_governance(selected_path: Path, joint_manifest_path: Path):
    selected = pd.read_csv(selected_path)
    manifest = json.loads(joint_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("holdout_opened") is not False or manifest.get("market_data_used") is not False:
        raise RuntimeError("Statcast governance manifest failed holdout/market audit")
    selected_specs = {
        (str(row.event), str(row.family)): {
            "window": str(row.window),
            "ridge": float(row.ridge),
        }
        for row in selected.itertuples()
    }
    stable = {
        str(row["event"]): list(row["stable_families_2021_2024"])
        for row in manifest["event_status"]
    }
    if set(stable) != set(EVENTS):
        raise RuntimeError("stable Statcast governance is missing an event")
    for event, families in stable.items():
        for family in families:
            if (event, family) not in selected_specs:
                raise RuntimeError(f"missing selected Statcast spec for {event}/{family}")
    return selected_specs, stable


def validate_inputs(x: pd.DataFrame, history_manifest: dict) -> None:
    x["season"] = pd.to_numeric(x.season, errors="raise").astype(int)
    x["inning"] = pd.to_numeric(x.inning, errors="raise").astype(int)
    if set(x.season.unique()) != {2021, 2022, 2023, 2024} or (x.season >= 2025).any():
        raise RuntimeError("development seasons must be exactly 2021-2024")
    if set(x.inning.unique()) != set(range(1, 10)):
        raise RuntimeError("I1-I9 target coverage is incomplete")
    if "market_data_used" not in x or x.market_data_used.astype(bool).any():
        raise RuntimeError("market isolation audit failed")
    if history_manifest.get("development_seasons") != [2021, 2022, 2023, 2024]:
        raise RuntimeError("PA-history source season audit failed")
    if history_manifest.get("holdout_opened") is not False:
        raise RuntimeError("PA-history manifest reports 2025 access")
    if history_manifest.get("same_day_history_included") is not False:
        raise RuntimeError("PA-history manifest reports same-day leakage")
    if history_manifest.get("fixed_shrinkage_used") is not False:
        raise RuntimeError("PA-history store must not assume a shrinkage strength")
    if history_manifest.get("market_data_used") is not False:
        raise RuntimeError("PA-history manifest reports market data")


def audit_core_reproduction(x):
    """Verify the new 365-day counts exactly reproduce the locked raw rates."""
    errors = {}
    for event in EVENTS:
        for entity in ("batter", "pitcher"):
            old = x[f"{entity}_365d_{event}_rate"].to_numpy(float)
            count = x[f"{entity}_365d_{event}_count"].to_numpy(float)
            opportunities = x[f"{entity}_365d_pa_count"].to_numpy(float)
            rebuilt = np.divide(
                count,
                opportunities,
                out=np.full(len(x), np.nan, dtype=float),
                where=opportunities > 0,
            )
            valid = np.isfinite(old) & np.isfinite(rebuilt)
            if valid.sum() < 1000:
                raise RuntimeError(f"insufficient core-reproduction rows for {event}/{entity}")
            error = float(np.max(np.abs(old[valid] - rebuilt[valid])))
            if error > 1e-12:
                raise RuntimeError(
                    f"365-day history does not reproduce locked core for {event}/{entity}: {error}"
                )
            errors[f"{event}_{entity}"] = error
    return errors


def statcast_specs_for_event(event, selected_specs, stable):
    return {
        family: selected_specs[(event, family)]
        for family in sorted(stable[event])
    }


def fixed_base_columns(event, specifications):
    columns = ["season", "inning", "batter_side", "pitcher_hand", f"y_{event}"]
    columns += statcast.core_rate_cols(event)
    for family, spec in specifications.items():
        columns += statcast.family_cols(family, spec["window"])
    return list(dict.fromkeys(columns))


def candidate_metrics(event, family):
    if family == "target_rate_history":
        return [event]
    if family == "cross_event_profile":
        return [metric for metric in BROAD_EVENTS if metric != event]
    if family == "outcome_composition":
        return COMPOSITION_EVENTS
    if family == "sample_reliability":
        return [event]
    raise ValueError(f"unknown PA-history family {family}")


def candidate_source_columns(event, family, windows):
    metrics = candidate_metrics(event, family)
    columns = []
    for window in windows:
        columns.append(f"league_{window}_pa_count")
        for metric in metrics:
            columns.append(f"league_{window}_{metric}_rate")
        for entity in ("batter", "pitcher"):
            columns.append(f"{entity}_{window}_pa_count")
            for metric in metrics:
                columns.append(f"{entity}_{window}_{metric}_count")
    return list(dict.fromkeys(columns))


def shrunken_rate(frame, entity, window, metric, alpha):
    count = frame[f"{entity}_{window}_{metric}_count"].to_numpy(float)
    opportunities = frame[f"{entity}_{window}_pa_count"].to_numpy(float)
    prior = frame[f"league_{window}_{metric}_rate"].to_numpy(float)
    if alpha == 0:
        return np.divide(
            count,
            opportunities,
            out=np.full(len(frame), np.nan, dtype=float),
            where=opportunities > 0,
        )
    return (count + alpha * prior) / (opportunities + alpha)


def candidate_frame(frame, event, family, window, alpha):
    values = {}
    if family in {"target_rate_history", "cross_event_profile", "outcome_composition"}:
        for entity in ("batter", "pitcher"):
            for metric in candidate_metrics(event, family):
                values[f"{entity}_{metric}"] = shrunken_rate(
                    frame, entity, window, metric, alpha
                )
    elif family == "sample_reliability":
        for entity in ("batter", "pitcher"):
            count = frame[f"{entity}_{window}_{event}_count"].to_numpy(float)
            opportunities = frame[f"{entity}_{window}_pa_count"].to_numpy(float)
            prior = frame[f"league_{window}_{event}_rate"].to_numpy(float)
            reliability = opportunities / (opportunities + alpha)
            raw = np.divide(
                count,
                opportunities,
                out=prior.copy(),
                where=opportunities > 0,
            )
            values[f"{entity}_log_pa"] = np.log1p(opportunities)
            values[f"{entity}_reliability"] = reliability
            values[f"{entity}_reliable_deviation"] = reliability * (raw - prior)
    else:
        raise ValueError(f"unknown PA-history family {family}")
    return pd.DataFrame(values, index=frame.index).replace([np.inf, -np.inf], np.nan)


def standardize_array(train, test):
    train_values = np.asarray(train, dtype=float)
    test_values = np.asarray(test, dtype=float)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-10)] = 1.0
    return (train_values - mean) / scale, (test_values - mean) / scale


def fixed_base_design(train, test, event, specifications):
    core_train = statcast.core_matrix(train, event)
    core_test = statcast.core_matrix(test, event)
    train_parts = [core_train]
    test_parts = [core_test]
    penalty = [0.0] * core_train.shape[1]
    for family in sorted(specifications):
        spec = specifications[family]
        columns = statcast.family_cols(family, spec["window"])
        block_train, block_test = statcast.standardize(train, test, columns)
        if family == "pitch_mix_matchup":
            block_train, block_test = statcast.add_pitch_mix_interactions(
                train, test, spec["window"], block_train, block_test
            )
        train_parts.append(block_train)
        test_parts.append(block_test)
        penalty.extend([spec["ridge"]] * block_train.shape[1])
    return np.column_stack(train_parts), np.column_stack(test_parts), np.asarray(penalty)


def prepare_family_year(x, event, family, year, specifications):
    windows = WINDOWS_BY_FAMILY[family]
    needed = fixed_base_columns(event, specifications)
    needed += candidate_source_columns(event, family, windows)
    needed = list(dict.fromkeys(needed))
    train = x[x.season < year][needed].dropna().copy()
    test = x[x.season == year][needed].dropna().copy()

    # Alpha=0 candidates require a real player sample. Enforce the condition
    # for every registered window so all candidate specifications in a family
    # are selected on exactly the same rows.
    if 0.0 in ALPHAS_BY_FAMILY[family]:
        for entity in ("batter", "pitcher"):
            for window in windows:
                train = train[train[f"{entity}_{window}_pa_count"] > 0]
                test = test[test[f"{entity}_{window}_pa_count"] > 0]
    if len(train) < 5000 or len(test) < 1000:
        raise RuntimeError(f"insufficient common rows for {event}/{family}/{year}")
    base_train, base_test, penalty = fixed_base_design(
        train, test, event, specifications
    )
    y_train = train[f"y_{event}"].to_numpy(float)
    y_test = test[f"y_{event}"].to_numpy(float)
    beta = fit_logistic(base_train, y_train, penalty)
    prediction = statcast.sigmoid(base_test @ beta)
    return {
        "event": event,
        "family": family,
        "year": year,
        "train": train,
        "test": test,
        "base_train": base_train,
        "base_test": base_test,
        "base_penalty": penalty,
        "y_train": y_train,
        "y_test": y_test,
        "comparator_logloss": statcast.ll(y_test, prediction),
        "comparator_brier": statcast.br(y_test, prediction),
    }


def score_candidate(prepared, window, alpha, ridge):
    family = prepared["family"]
    event = prepared["event"]
    feature_train = candidate_frame(prepared["train"], event, family, window, alpha)
    feature_test = candidate_frame(prepared["test"], event, family, window, alpha)
    if feature_train.isna().any().any() or feature_test.isna().any().any():
        raise RuntimeError(f"candidate feature missingness survived common-row filter: {event}/{family}")
    block_train, block_test = standardize_array(feature_train, feature_test)
    design_train = np.column_stack([prepared["base_train"], block_train])
    design_test = np.column_stack([prepared["base_test"], block_test])
    penalty = np.concatenate(
        [prepared["base_penalty"], np.full(block_train.shape[1], ridge, dtype=float)]
    )
    beta = fit_logistic(design_train, prepared["y_train"], penalty)
    prediction = statcast.sigmoid(design_test @ beta)
    challenger_logloss = statcast.ll(prepared["y_test"], prediction)
    challenger_brier = statcast.br(prepared["y_test"], prediction)
    return {
        "event": event,
        "family": family,
        "window": window,
        "alpha": float(alpha),
        "ridge": float(ridge),
        "test_year": prepared["year"],
        "feature_count": int(block_train.shape[1]),
        "n_train": int(len(prepared["train"])),
        "n_test": int(len(prepared["test"])),
        "comparator_logloss": prepared["comparator_logloss"],
        "challenger_logloss": challenger_logloss,
        "logloss_improvement": prepared["comparator_logloss"] - challenger_logloss,
        "comparator_brier": prepared["comparator_brier"],
        "challenger_brier": challenger_brier,
        "brier_improvement": prepared["comparator_brier"] - challenger_brier,
    }


def selected_specifications(selection_grid):
    return (
        selection_grid.sort_values(
            ["event", "family", "challenger_logloss", "challenger_brier"],
            ascending=[True, True, True, True],
        )
        .groupby(["event", "family"], as_index=False)
        .head(1)
        .sort_values(["event", "family"])
        .reset_index(drop=True)
    )


def marginal_status(selected, validation, confirmation):
    keys = ["event", "family"]
    status = selected[
        keys + ["window", "alpha", "ridge", "logloss_improvement", "brier_improvement", "n_test"]
    ].rename(
        columns={
            "logloss_improvement": "selection_2022_logloss_improvement",
            "brier_improvement": "selection_2022_brier_improvement",
            "n_test": "selection_2022_n",
        }
    )
    status = status.merge(
        validation[keys + ["logloss_improvement", "brier_improvement", "n_test"]].rename(
            columns={
                "logloss_improvement": "validation_2023_logloss_improvement",
                "brier_improvement": "validation_2023_brier_improvement",
                "n_test": "validation_2023_n",
            }
        ),
        on=keys,
        validate="one_to_one",
    ).merge(
        confirmation[keys + ["logloss_improvement", "brier_improvement", "n_test"]].rename(
            columns={
                "logloss_improvement": "confirmation_2024_logloss_improvement",
                "brier_improvement": "confirmation_2024_brier_improvement",
                "n_test": "confirmation_2024_n",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    status["earns_marginal_candidate_status"] = (
        (status.selection_2022_logloss_improvement > 0)
        & (status.validation_2023_logloss_improvement > 0)
        & (status.confirmation_2024_logloss_improvement > 0)
        & (status.confirmation_2024_brier_improvement > 0)
    )
    return status


def event_candidate_specs(selected, event):
    rows = selected[selected.event == event]
    if set(rows.family) != set(FAMILIES):
        raise RuntimeError(f"missing 2022-selected PA-history family for {event}")
    return {
        str(row.family): {
            "window": str(row.window),
            "alpha": float(row.alpha),
            "ridge": float(row.ridge),
        }
        for row in rows.itertuples()
    }


def prepare_joint_year(x, event, year, statcast_specifications, candidate_specifications):
    needed = fixed_base_columns(event, statcast_specifications)
    for family, spec in candidate_specifications.items():
        needed += candidate_source_columns(event, family, [spec["window"]])
    needed = list(dict.fromkeys(needed))
    train = x[x.season < year][needed].dropna().copy()
    test = x[x.season == year][needed].dropna().copy()
    candidate_train = {}
    candidate_test = {}
    keep_train = pd.Series(True, index=train.index)
    keep_test = pd.Series(True, index=test.index)
    for family, spec in candidate_specifications.items():
        f_train = candidate_frame(train, event, family, spec["window"], spec["alpha"])
        f_test = candidate_frame(test, event, family, spec["window"], spec["alpha"])
        candidate_train[family] = f_train
        candidate_test[family] = f_test
        keep_train &= f_train.notna().all(axis=1)
        keep_test &= f_test.notna().all(axis=1)
    train = train.loc[keep_train].copy()
    test = test.loc[keep_test].copy()
    if len(train) < 5000 or len(test) < 1000:
        raise RuntimeError(f"insufficient joint complete cases for {event}/{year}")
    base_train, base_test, penalty = fixed_base_design(
        train, test, event, statcast_specifications
    )
    y_train = train[f"y_{event}"].to_numpy(float)
    y_test = test[f"y_{event}"].to_numpy(float)
    beta = fit_logistic(base_train, y_train, penalty)
    prediction = statcast.sigmoid(base_test @ beta)
    blocks = {}
    for family in sorted(candidate_specifications):
        block_train, block_test = standardize_array(
            candidate_train[family].loc[train.index],
            candidate_test[family].loc[test.index],
        )
        blocks[family] = (block_train, block_test)
    return {
        "event": event,
        "year": year,
        "specifications": candidate_specifications,
        "base_train": base_train,
        "base_test": base_test,
        "base_penalty": penalty,
        "blocks": blocks,
        "y_train": y_train,
        "y_test": y_test,
        "comparator_logloss": statcast.ll(y_test, prediction),
        "comparator_brier": statcast.br(y_test, prediction),
    }


def score_subset(prepared, subset):
    subset = tuple(sorted(subset))
    train_parts = [prepared["base_train"]]
    test_parts = [prepared["base_test"]]
    penalty = list(prepared["base_penalty"])
    feature_count = 0
    for family in subset:
        block_train, block_test = prepared["blocks"][family]
        train_parts.append(block_train)
        test_parts.append(block_test)
        penalty.extend([prepared["specifications"][family]["ridge"]] * block_train.shape[1])
        feature_count += block_train.shape[1]
    design_train = np.column_stack(train_parts)
    design_test = np.column_stack(test_parts)
    beta = fit_logistic(design_train, prepared["y_train"], np.asarray(penalty))
    prediction = statcast.sigmoid(design_test @ beta)
    challenger_logloss = statcast.ll(prepared["y_test"], prediction)
    challenger_brier = statcast.br(prepared["y_test"], prediction)
    return {
        "event": prepared["event"],
        "test_year": prepared["year"],
        "families": "|".join(subset) if subset else "comparator_only",
        "family_count": len(subset),
        "feature_count": int(feature_count),
        "n_train": int(len(prepared["y_train"])),
        "n_test": int(len(prepared["y_test"])),
        "comparator_logloss": prepared["comparator_logloss"],
        "challenger_logloss": challenger_logloss,
        "logloss_improvement": prepared["comparator_logloss"] - challenger_logloss,
        "comparator_brier": prepared["comparator_brier"],
        "challenger_brier": challenger_brier,
        "brier_improvement": prepared["comparator_brier"] - challenger_brier,
    }


def subsets(families):
    for size in range(len(families) + 1):
        yield from itertools.combinations(families, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1-matrix", type=Path, required=True)
    parser.add_argument("--statcast-batter-asof", type=Path, required=True)
    parser.add_argument("--statcast-pitcher-asof", type=Path, required=True)
    parser.add_argument("--history-batter-asof", type=Path, required=True)
    parser.add_argument("--history-pitcher-asof", type=Path, required=True)
    parser.add_argument("--history-league-asof", type=Path, required=True)
    parser.add_argument("--history-manifest", type=Path, required=True)
    parser.add_argument("--statcast-selected-specs", type=Path, required=True)
    parser.add_argument("--statcast-joint-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_statcast, stable_statcast = load_governance(
        args.statcast_selected_specs, args.statcast_joint_manifest
    )
    m1_columns = [
        "game_date",
        "season",
        "inning",
        "batter_id",
        "pitcher_id",
        "batter_side",
        "pitcher_hand",
        "market_data_used",
    ]
    m1_columns += [f"y_{event}" for event in EVENTS]
    for event in EVENTS:
        m1_columns += statcast.core_rate_cols(event)
    m1_columns = list(dict.fromkeys(m1_columns))
    x = pd.read_parquet(args.m1_matrix, columns=m1_columns)
    history_manifest = json.loads(args.history_manifest.read_text(encoding="utf-8"))
    validate_inputs(x, history_manifest)

    statcast_columns = set()
    for event in EVENTS:
        for family, spec in statcast_specs_for_event(
            event, selected_statcast, stable_statcast
        ).items():
            statcast_columns.update(statcast.family_cols(family, spec["window"]))
    history_columns = set()
    for event in EVENTS:
        for family in FAMILIES:
            history_columns.update(
                candidate_source_columns(event, family, WINDOWS_BY_FAMILY[family])
            )
    batter_statcast_columns = sorted(
        column for column in statcast_columns if column.startswith("batter_")
    )
    pitcher_statcast_columns = sorted(
        column for column in statcast_columns if column.startswith("pitcher_")
    )
    batter_history_columns = sorted(
        column for column in history_columns if column.startswith("batter_")
    )
    pitcher_history_columns = sorted(
        column for column in history_columns if column.startswith("pitcher_")
    )
    league_history_columns = sorted(
        column for column in history_columns if column.startswith("league_")
    )
    frames = [
        x,
        pd.read_parquet(
            args.statcast_batter_asof,
            columns=["batter_id", "game_date", *batter_statcast_columns],
        ),
        pd.read_parquet(
            args.statcast_pitcher_asof,
            columns=["pitcher_id", "game_date", *pitcher_statcast_columns],
        ),
        pd.read_parquet(
            args.history_batter_asof,
            columns=["batter_id", "game_date", *batter_history_columns],
        ),
        pd.read_parquet(
            args.history_pitcher_asof,
            columns=["pitcher_id", "game_date", *pitcher_history_columns],
        ),
        pd.read_parquet(
            args.history_league_asof,
            columns=["game_date", *league_history_columns],
        ),
    ]
    for frame in frames:
        frame["game_date"] = pd.to_datetime(frame.game_date, errors="raise").dt.normalize()
    x = (
        frames[0]
        .merge(frames[1], on=["batter_id", "game_date"], how="left", validate="many_to_one")
        .merge(frames[2], on=["pitcher_id", "game_date"], how="left", validate="many_to_one")
        .merge(frames[3], on=["batter_id", "game_date"], how="left", validate="many_to_one")
        .merge(frames[4], on=["pitcher_id", "game_date"], how="left", validate="many_to_one")
        .merge(frames[5], on="game_date", how="left", validate="many_to_one")
    )
    core_reproduction_errors = audit_core_reproduction(x)
    selection_rows = []
    for event in EVENTS:
        specifications = statcast_specs_for_event(event, selected_statcast, stable_statcast)
        for family in FAMILIES:
            print(f"selection 2022: {event}/{family}", flush=True)
            prepared = prepare_family_year(x, event, family, 2022, specifications)
            for window in WINDOWS_BY_FAMILY[family]:
                for alpha in ALPHAS_BY_FAMILY[family]:
                    for ridge in RIDGES:
                        selection_rows.append(score_candidate(prepared, window, alpha, ridge))
            del prepared
            gc.collect()
    selection_grid = pd.DataFrame(selection_rows)
    selected = selected_specifications(selection_grid)

    validation_rows = []
    confirmation_rows = []
    for row in selected.itertuples():
        specifications = statcast_specs_for_event(row.event, selected_statcast, stable_statcast)
        for year, destination in ((2023, validation_rows), (2024, confirmation_rows)):
            print(f"fixed-spec {year}: {row.event}/{row.family}", flush=True)
            prepared = prepare_family_year(x, row.event, row.family, year, specifications)
            destination.append(
                score_candidate(prepared, row.window, float(row.alpha), float(row.ridge))
            )
            del prepared
            gc.collect()
    validation = pd.DataFrame(validation_rows)
    confirmation = pd.DataFrame(confirmation_rows)
    marginal = marginal_status(selected, validation, confirmation)

    joint_selection_rows = []
    joint_chosen_rows = []
    joint_confirmation_rows = []
    ablation_rows = []
    stable_rows = []
    event_status = []
    for event in EVENTS:
        print(f"joint subset gate: {event}", flush=True)
        statcast_specifications = statcast_specs_for_event(
            event, selected_statcast, stable_statcast
        )
        candidate_specifications = event_candidate_specs(selected, event)
        joint_2023 = prepare_joint_year(
            x, event, 2023, statcast_specifications, candidate_specifications
        )
        grid = pd.DataFrame(
            [score_subset(joint_2023, subset) for subset in subsets(sorted(FAMILIES))]
        )
        joint_selection_rows.extend(grid.to_dict("records"))
        chosen = grid.sort_values(
            ["challenger_logloss", "challenger_brier", "family_count"],
            ascending=[True, True, True],
        ).iloc[0].to_dict()
        joint_chosen_rows.append(chosen)
        chosen_families = (
            [] if chosen["families"] == "comparator_only" else chosen["families"].split("|")
        )
        joint_2024 = prepare_joint_year(
            x, event, 2024, statcast_specifications, candidate_specifications
        )
        confirmation_row = score_subset(joint_2024, chosen_families)
        joint_confirmation_rows.append(confirmation_row)

        stable_families = list(chosen_families)
        pruning_round = 0
        while stable_families:
            full = {
                2023: score_subset(joint_2023, stable_families),
                2024: score_subset(joint_2024, stable_families),
            }
            failing = []
            for family in stable_families:
                reduced = [candidate for candidate in stable_families if candidate != family]
                rows = []
                for year, prepared in ((2023, joint_2023), (2024, joint_2024)):
                    ablated = score_subset(prepared, reduced)
                    row = {
                        "event": event,
                        "pruning_round": pruning_round,
                        "family_removed": family,
                        "test_year": year,
                        "selected_families": "|".join(stable_families),
                        "ablated_families": ablated["families"],
                        "n_test": full[year]["n_test"],
                        "conditional_logloss_improvement": (
                            ablated["challenger_logloss"] - full[year]["challenger_logloss"]
                        ),
                        "conditional_brier_improvement": (
                            ablated["challenger_brier"] - full[year]["challenger_brier"]
                        ),
                    }
                    rows.append(row)
                    ablation_rows.append(row)
                if not (
                    rows[0]["conditional_logloss_improvement"] > 0
                    and rows[1]["conditional_logloss_improvement"] > 0
                    and rows[1]["conditional_brier_improvement"] > 0
                ):
                    failing.append(family)
            if not failing:
                break
            stable_families = [family for family in stable_families if family not in failing]
            pruning_round += 1

        stable_specifications = {
            family: candidate_specifications[family] for family in stable_families
        }
        stable_by_year = {}
        for year in (2023, 2024):
            if stable_specifications:
                prepared = prepare_joint_year(
                    x, event, year, statcast_specifications, stable_specifications
                )
                row = score_subset(prepared, stable_families)
            else:
                prepared = prepare_joint_year(
                    x, event, year, statcast_specifications, candidate_specifications
                )
                row = score_subset(prepared, [])
            row["development_selection_role"] = "joint_stability"
            stable_by_year[year] = row
            stable_rows.append(row)
        event_status.append(
            {
                "event": event,
                "selected_specifications_2022": candidate_specifications,
                "selected_families_2023": chosen_families,
                "confirmation_2024_logloss_improvement": confirmation_row[
                    "logloss_improvement"
                ],
                "confirmation_2024_brier_improvement": confirmation_row[
                    "brier_improvement"
                ],
                "stable_families_2021_2024": stable_families,
                "families_removed_by_conditional_gate": sorted(
                    set(chosen_families) - set(stable_families)
                ),
                "stable_subset_2023_logloss_improvement": stable_by_year[2023][
                    "logloss_improvement"
                ],
                "stable_subset_2024_logloss_improvement": stable_by_year[2024][
                    "logloss_improvement"
                ],
                "stable_subset_2024_brier_improvement": stable_by_year[2024][
                    "brier_improvement"
                ],
                "joint_feature_set_development_selected": bool(
                    stable_families
                    and stable_by_year[2023]["logloss_improvement"] > 0
                    and stable_by_year[2024]["logloss_improvement"] > 0
                    and stable_by_year[2024]["brier_improvement"] > 0
                ),
            }
        )

    selection_grid.to_csv(
        args.output_dir / "m1_pa_history_selection_2022_grid.csv", index=False
    )
    selected.to_csv(args.output_dir / "m1_pa_history_selected_specs.csv", index=False)
    validation.to_csv(args.output_dir / "m1_pa_history_validation_2023.csv", index=False)
    confirmation.to_csv(args.output_dir / "m1_pa_history_confirmation_2024.csv", index=False)
    marginal.to_csv(args.output_dir / "m1_pa_history_marginal_status.csv", index=False)
    pd.DataFrame(joint_selection_rows).to_csv(
        args.output_dir / "m1_pa_history_joint_selection_2023.csv", index=False
    )
    pd.DataFrame(joint_chosen_rows).to_csv(
        args.output_dir / "m1_pa_history_joint_selected_by_event.csv", index=False
    )
    pd.DataFrame(joint_confirmation_rows).to_csv(
        args.output_dir / "m1_pa_history_joint_confirmation_2024.csv", index=False
    )
    pd.DataFrame(ablation_rows).to_csv(
        args.output_dir / "m1_pa_history_joint_conditional_ablation.csv", index=False
    )
    pd.DataFrame(stable_rows).to_csv(
        args.output_dir / "m1_pa_history_joint_stable_metrics.csv", index=False
    )

    manifest = {
        "status": "PASS",
        "architecture": "M1_PA_history_reliability_incremental_and_joint_gate",
        "development_seasons": [2021, 2022, 2023, 2024],
        "candidate_specification_selection_year": 2022,
        "joint_subset_selection_year": 2023,
        "development_confirmation_year": 2024,
        "final_feature_selection_years": [2022, 2023, 2024],
        "holdout_season": 2025,
        "holdout_opened": False,
        "market_data_used": False,
        "target_innings": list(range(1, 10)),
        "statistics_timing": "strictly prior date; same-day PA history excluded",
        "raw_365d_core_reproduction_max_abs_error": core_reproduction_errors,
        "comparator": "validated 365d event-rate core + retained platoon terms + event-specific stable Statcast families",
        "statcast_stable_families": stable_statcast,
        "feature_families": FAMILIES,
        "windows_by_family": WINDOWS_BY_FAMILY,
        "alpha_candidates_by_family": ALPHAS_BY_FAMILY,
        "ridge_candidates": RIDGES,
        "shrinkage_prior": "strictly prior-date league event rate for the same registered window",
        "fixed_shrinkage_used": False,
        "comparison_rows": "identical complete-case rows across all registered specifications within each event/family/year; common intersection across families for joint selection",
        "marginal_rule": "specification selected on 2022; positive log loss in 2022, 2023, 2024 and positive 2024 Brier earns marginal candidate status",
        "joint_subset_rule": "minimum 2023 log loss; Brier then smaller subset as tie-breakers",
        "conditional_rule": "family must improve log loss in 2023 and 2024 and Brier in 2024 versus leave-one-family-out ablation",
        "automatic_production_promotion": False,
        "participant_identity_use": "retrospective realized matchup oracle for M1 skill validation only",
        "event_status": event_status,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(marginal.to_string(index=False))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

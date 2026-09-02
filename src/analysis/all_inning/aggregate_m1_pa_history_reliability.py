#!/usr/bin/env python3
"""Aggregate four independently validated PA-history event artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


EVENTS = ["k", "baserunner", "hr", "nonhr_hit"]
CSV_FILES = [
    "m1_pa_history_selection_2022_grid.csv",
    "m1_pa_history_selected_specs.csv",
    "m1_pa_history_validation_2023.csv",
    "m1_pa_history_confirmation_2024.csv",
    "m1_pa_history_marginal_status.csv",
    "m1_pa_history_joint_selection_2023.csv",
    "m1_pa_history_joint_selected_by_event.csv",
    "m1_pa_history_joint_confirmation_2024.csv",
    "m1_pa_history_joint_conditional_ablation.csv",
    "m1_pa_history_joint_stable_metrics.csv",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = sorted(args.input_root.rglob("manifest.json"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    if len(manifests) != len(EVENTS):
        raise RuntimeError(
            f"expected {len(EVENTS)} event manifests under {args.input_root}; found {manifest_paths}"
        )
    by_event = {}
    for path, manifest in zip(manifest_paths, manifests):
        events_run = manifest.get("events_run")
        if not isinstance(events_run, list) or len(events_run) != 1:
            raise RuntimeError(f"event artifact is not single-event: {path}")
        event = str(events_run[0])
        if event in by_event:
            raise RuntimeError(f"duplicate event artifact: {event}")
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"event artifact failed: {event}")
        if manifest.get("holdout_opened") is not False:
            raise RuntimeError(f"event artifact opened 2025: {event}")
        if manifest.get("market_data_used") is not False:
            raise RuntimeError(f"event artifact used market data: {event}")
        by_event[event] = (path.parent, manifest)
    if set(by_event) != set(EVENTS):
        raise RuntimeError(f"event artifact set mismatch: {sorted(by_event)}")

    for filename in CSV_FILES:
        parts = []
        for event in EVENTS:
            path = by_event[event][0] / filename
            if not path.exists():
                raise RuntimeError(f"missing {filename} for {event}")
            parts.append(pd.read_csv(path))
        combined = pd.concat(parts, ignore_index=True)
        if "event" in combined.columns:
            combined["event"] = pd.Categorical(
                combined.event, categories=EVENTS, ordered=True
            )
            sort_columns = ["event"]
            if "test_year" in combined.columns:
                sort_columns.append("test_year")
            if "family" in combined.columns:
                sort_columns.append("family")
            combined = combined.sort_values(sort_columns, kind="mergesort")
            combined["event"] = combined.event.astype(str)
        combined.to_csv(args.output_dir / filename, index=False)

    base = dict(by_event[EVENTS[0]][1])
    base["events_run"] = EVENTS
    base["event_status"] = []
    base["raw_365d_core_reproduction_max_abs_error"] = {}
    for event in EVENTS:
        manifest = by_event[event][1]
        if manifest.get("development_seasons") != [2021, 2022, 2023, 2024]:
            raise RuntimeError(f"development season mismatch for {event}")
        if manifest.get("target_innings") != list(range(1, 10)):
            raise RuntimeError(f"inning coverage mismatch for {event}")
        if manifest.get("fixed_shrinkage_used") is not False:
            raise RuntimeError(f"fixed shrinkage reported for {event}")
        base["event_status"].extend(manifest["event_status"])
        base["raw_365d_core_reproduction_max_abs_error"].update(
            manifest["raw_365d_core_reproduction_max_abs_error"]
        )
    base["event_status"] = sorted(
        base["event_status"], key=lambda row: EVENTS.index(row["event"])
    )
    base["execution_phases"] = {
        "phase_1": "single reusable strictly-prior-date PA-history build",
        "phase_2": "four independent event validation jobs run in parallel",
        "phase_3": "artifact aggregation and cross-event governance audit",
    }
    base["source_event_artifact_count"] = len(manifests)

    marginal = pd.read_csv(args.output_dir / "m1_pa_history_marginal_status.csv")
    stable = pd.read_csv(args.output_dir / "m1_pa_history_joint_stable_metrics.csv")
    if len(marginal) != 16 or set(marginal.event) != set(EVENTS):
        raise RuntimeError("combined marginal result cardinality failed")
    if len(stable) != 8 or set(stable.test_year) != {2023, 2024}:
        raise RuntimeError("combined stable result cardinality failed")
    if max(base["raw_365d_core_reproduction_max_abs_error"].values()) > 1e-12:
        raise RuntimeError("combined core-reproduction audit failed")

    (args.output_dir / "manifest.json").write_text(
        json.dumps(base, indent=2), encoding="utf-8"
    )
    print(json.dumps(base, indent=2))
    print(stable.to_string(index=False))


if __name__ == "__main__":
    main()

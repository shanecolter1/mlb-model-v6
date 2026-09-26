#!/usr/bin/env python3
"""Build monthly point-in-time I2 vNext models for the 2025 replay.

The model specification and hyperparameters are frozen before 2025.  This
script only updates player coefficients with baseball observations that were
available before each monthly cutoff.  It never re-selects features,
regularization, recency, calibration, or any market-facing parameter.

Cadence is deliberately monthly for the bounded first production build:
- games before 2025-05-01 use the end-2024 frozen artifact;
- May through September use models fit strictly before the first of each month.

This is closer to the live expanding/as-of refit than replaying all of 2025
with a stale December-2024 model, while keeping the historical build bounded.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import fit_i2_vnext as base


CUTOFFS = [
    pd.Timestamp("2025-05-01"),
    pd.Timestamp("2025-06-01"),
    pd.Timestamp("2025-07-01"),
    pd.Timestamp("2025-08-01"),
    pd.Timestamp("2025-09-01"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/derived/i2_vnext/i2_pa_statcast.csv"),
    )
    p.add_argument(
        "--frozen-model",
        type=Path,
        default=Path("data/derived/i2_vnext/i2_vnext_event_model.json"),
    )
    p.add_argument(
        "--arsenal-dir",
        type=Path,
        default=Path("data/derived/i2_vnext/arsenal"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/i2_vnext/walkforward_2025"),
    )
    return p.parse_args()


def prepare_frame(path: Path, arsenal_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="raise")
    df = df[df["event_class"].isin(base.EVENTS)].copy()
    if df.empty:
        raise RuntimeError("No modeled I2 PA rows available for walk-forward fitting")

    numeric = df.copy()
    numeric["batter"] = pd.to_numeric(numeric["batter"], errors="raise").astype(int)
    numeric["pitcher"] = pd.to_numeric(numeric["pitcher"], errors="raise").astype(int)
    numeric = base.add_arsenal_feature(numeric, arsenal_dir)
    df["arsenal_matchup_xwoba"] = numeric["arsenal_matchup_xwoba"].to_numpy()

    df["batter"] = pd.to_numeric(df["batter"], errors="raise").astype(int).astype(str)
    df["pitcher"] = pd.to_numeric(df["pitcher"], errors="raise").astype(int).astype(str)
    df["platoon"] = df["platoon"].fillna("?v?").astype(str)
    df["home_team"] = df["home_team"].fillna("UNKNOWN").astype(str)
    for platoon in base.PLATOONS:
        df[f"arsenal_x_{platoon}"] = np.where(
            df["platoon"] == platoon,
            df["arsenal_matchup_xwoba"],
            0.0,
        )
    return df


def training_summary(df: pd.DataFrame) -> dict:
    return {
        "start": df["game_date"].min().date().isoformat(),
        "end": df["game_date"].max().date().isoformat(),
        "n": int(len(df)),
        "seasons": sorted(int(x) for x in df["season"].unique()),
        "event_counts": {
            str(k): int(v) for k, v in df["event_class"].value_counts().to_dict().items()
        },
        "batters": int(df["batter"].nunique()),
        "pitchers": int(df["pitcher"].nunique()),
    }


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.frozen_model.read_text(encoding="utf-8"))
    if frozen.get("market_inputs_used") is not False:
        raise RuntimeError("Frozen model is not explicitly market-isolated")

    selected = frozen.get("selected") or {}
    half_life = float(selected["half_life_days"])
    c_value = float(selected["C"])

    df = prepare_frame(args.input, args.arsenal_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_name = "model_end_2024_frozen.json"
    shutil.copyfile(args.frozen_model, args.output_dir / base_name)
    entries = [{
        "effective_from": "2025-01-01",
        "training_cutoff_exclusive": "2025-01-01",
        "model_file": base_name,
        "source": "end-2024 frozen model",
        "training_end": frozen.get("training", {}).get("end"),
    }]

    for cutoff in CUTOFFS:
        train = df[df["game_date"] < cutoff].copy()
        if train.empty:
            raise RuntimeError(f"No training rows before {cutoff.date().isoformat()}")
        if train["game_date"].max() >= cutoff:
            raise RuntimeError("Point-in-time cutoff violation")

        prep, model = base.fit_one(train, c_value, half_life)
        artifact = base.serialize_model(prep, model, selected, [], train)
        artifact["artifact_role"] = "WALKFORWARD_OOS_POINT_IN_TIME"
        artifact["market_inputs_used"] = False
        artifact["walkforward"] = {
            "cutoff_exclusive": cutoff.date().isoformat(),
            "max_training_date": train["game_date"].max().date().isoformat(),
            "hyperparameters_reselected": False,
            "features_reselected": False,
            "calibration_used": False,
            "market_inputs_used": False,
        }
        artifact["training"] = training_summary(train)

        file_name = f"model_before_{cutoff.date().isoformat()}.json"
        (args.output_dir / file_name).write_text(
            json.dumps(artifact, separators=(",", ":")),
            encoding="utf-8",
        )
        entries.append({
            "effective_from": cutoff.date().isoformat(),
            "training_cutoff_exclusive": cutoff.date().isoformat(),
            "model_file": file_name,
            "source": "monthly expanding point-in-time refit",
            "training_end": artifact["training"]["end"],
            "training_n": artifact["training"]["n"],
        })

    manifest = {
        "version": "i2-vnext-walkforward-2025-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_inputs_used": False,
        "season": 2025,
        "cadence": "monthly",
        "policy": (
            "Fixed pre-2025 specification/hyperparameters; update coefficients only "
            "with I2 PAs strictly before each monthly cutoff."
        ),
        "hyperparameters": {
            "half_life_days": half_life,
            "C": c_value,
        },
        "entries": entries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

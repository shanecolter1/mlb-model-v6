#!/usr/bin/env python3
"""Build leakage-safe historical I2 vNext walk-forward models without home_team.

Research-only challenger for a fit-only nuisance-control audit. The only
specification change from the governed direct-I2 model is removal of the
home_team categorical nuisance block. Batter identity, pitcher identity,
standalone platoon, and the four platoon-specific arsenal interaction numerics
are retained. Live inference behavior is otherwise unchanged.

No market data. Fixed governed hyperparameters. Target-season observations are
used only in expanding refits strictly before each monthly cutoff.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import fit_i2_vnext as base

CATS = ["batter", "pitcher", "platoon"]


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--arsenal-dir",type=Path,required=True)
    p.add_argument("--season",type=int,required=True)
    p.add_argument("--half-life",type=float,default=730.0)
    p.add_argument("--c",type=float,default=0.05)
    p.add_argument("--max-iter",type=int,default=7500)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--arsenal-output",type=Path,required=True)
    return p.parse_args()


def prepare(path, arsenal_dir):
    df=pd.read_csv(path,low_memory=False)
    df["game_date"]=pd.to_datetime(df["game_date"],errors="raise")
    df=df[df["event_class"].isin(base.EVENTS)].copy()
    df["season"]=pd.to_numeric(df["season"],errors="raise").astype(int)

    numeric=df.copy()
    numeric["batter"]=pd.to_numeric(numeric["batter"],errors="raise").astype(int)
    numeric["pitcher"]=pd.to_numeric(numeric["pitcher"],errors="raise").astype(int)
    numeric=base.add_arsenal_feature(numeric,arsenal_dir)
    df["arsenal_matchup_xwoba"]=numeric["arsenal_matchup_xwoba"].to_numpy()

    df["batter"]=pd.to_numeric(df["batter"],errors="raise").astype(int).astype(str)
    df["pitcher"]=pd.to_numeric(df["pitcher"],errors="raise").astype(int).astype(str)
    df["platoon"]=df["platoon"].fillna("?v?").astype(str)
    df["home_team"]=df["home_team"].fillna("UNKNOWN").astype(str)
    for platoon in base.PLATOONS:
        df[f"arsenal_x_{platoon}"]=np.where(
            df["platoon"]==platoon,df["arsenal_matchup_xwoba"],0.0
        )
    if df[base.NUM].isna().any().any():
        raise ValueError("Missing leakage-safe prior-season arsenal features")
    return df


def summary(df):
    return {
        "start":df["game_date"].min().date().isoformat(),
        "end":df["game_date"].max().date().isoformat(),
        "n":int(len(df)),
        "seasons":sorted(int(x) for x in df["season"].unique()),
        "batters":int(df["batter"].nunique()),
        "pitchers":int(df["pitcher"].nunique()),
    }


def artifact_for(train,c,half_life,target_season,cutoff,max_iter):
    base.MAX_MODEL_ITER=int(max_iter)
    prep,model=base.fit_one(train,c,half_life,categorical_features=CATS)
    selected={
        "half_life_days":half_life,
        "C":c,
        "selection_metric":"fixed governed hyperparameters; home_team removal not reselected",
    }
    artifact=base.serialize_model(prep,model,selected,[],train)
    artifact["version"]="i2-vnext-direct-talent-no-home-team-research-v1"
    artifact["artifact_role"]="HISTORICAL_POINT_IN_TIME_NO_HOME_TEAM_CHALLENGER"
    artifact["market_inputs_used"]=False
    artifact["categorical_features"]=CATS
    artifact["category_levels"]={
        feature:[str(v) for v in values]
        for feature,values in zip(CATS,prep.named_transformers_["cat"].categories_)
    }
    artifact["nuisance_controls"]={}
    artifact["research_specification_change"]={
        "removed_feature_family":"home_team fit-only nuisance",
        "retained":["batter identity","pitcher identity","standalone platoon","arsenal x platoon numeric interactions"],
        "production_changed":False,
    }
    artifact["historical_replication"]={
        "target_season":target_season,
        "cutoff_exclusive":cutoff,
        "hyperparameters_reselected":False,
        "features_reselected":False,
        "calibration_used":False,
        "market_inputs_used":False,
    }
    artifact["training"]=summary(train)
    return artifact


def main():
    a=parse_args()
    df=prepare(a.input,a.arsenal_dir)
    if a.season not in set(df["season"]):
        raise ValueError("Target season absent from I2 PA dataset")

    a.output_dir.mkdir(parents=True,exist_ok=True)
    entries=[]
    cutoffs=[f"{a.season}-01-01"]+[f"{a.season}-{m:02d}-01" for m in range(5,10)]
    for i,cutoff in enumerate(cutoffs):
        cutoff_ts=pd.Timestamp(cutoff)
        train=df[df["game_date"] < cutoff_ts].copy()
        if train.empty or train["game_date"].max() >= cutoff_ts:
            raise ValueError(f"Training cutoff violation: {cutoff}")
        artifact=artifact_for(train,a.c,a.half_life,a.season,cutoff,a.max_iter)
        name="model_preseason.json" if i==0 else f"model_before_{cutoff}.json"
        (a.output_dir/name).write_text(json.dumps(artifact,separators=(",",":")))
        entries.append({
            "effective_from":cutoff,
            "training_cutoff_exclusive":cutoff,
            "model_file":name,
            "training_end":artifact["training"]["end"],
            "training_n":artifact["training"]["n"],
        })

    manifest={
        "version":f"i2-vnext-walkforward-{a.season}-no-home-team-research-v1",
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "market_inputs_used":False,
        "season":a.season,
        "cadence":"preseason then monthly May-September",
        "policy":"Fixed governed hyperparameters; no-home-team challenger; expanding fits use only I2 PAs strictly before each cutoff.",
        "hyperparameters":{"half_life_days":a.half_life,"C":a.c,"solver":"saga","solver_max_iter":a.max_iter},
        "categorical_features":CATS,
        "removed_feature_family":"home_team",
        "entries":entries,
        "governance":{
            "target_season_outcomes_used_for_feature_selection":False,
            "target_season_outcomes_used_for_hyperparameter_selection_in_this_workflow":False,
            "market_inputs_used":False,
            "production_changed":False,
        },
    }
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

    a.arsenal_output.parent.mkdir(parents=True,exist_ok=True)
    a.arsenal_output.write_text(
        json.dumps(base.live_arsenal_payload(a.season-1,a.arsenal_dir),separators=(",",":"))
    )
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()

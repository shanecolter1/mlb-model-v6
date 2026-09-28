#!/usr/bin/env python3
"""Merge newly built early I2 Statcast history into the canonical later artifact."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--early",type=Path,required=True)
    p.add_argument("--canonical",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--manifest-output",type=Path,required=True)
    return p.parse_args()

def main():
    a=args()
    early=pd.read_csv(a.early,low_memory=False)
    later=pd.read_csv(a.canonical,low_memory=False)
    if int(early["season"].max()) >= int(later["season"].min()):
        overlap=set(early["season"].unique()) & set(later["season"].unique())
        if overlap:
            raise ValueError(f"Unexpected season overlap: {sorted(overlap)}")
    out=pd.concat([early,later],ignore_index=True)
    out["game_date"]=pd.to_datetime(out["game_date"],errors="raise")
    out=out.sort_values(["game_date","game_pk","at_bat_number"]).reset_index(drop=True)
    if out.duplicated(["game_pk","at_bat_number"]).any():
        raise ValueError("Duplicate PAs after merge")
    seasons=sorted(int(x) for x in out["season"].unique())
    if seasons != [2021,2022,2023,2024,2025,2026]:
        raise ValueError(f"Unexpected seasons: {seasons}")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    serial=out.copy()
    serial["game_date"]=serial["game_date"].dt.strftime("%Y-%m-%d")
    serial.to_csv(a.output,index=False)
    manifest={
        "version":"i2-vnext-statcast-merged-existing-plus-missing-history-v1",
        "market_inputs_used":False,
        "seasons":seasons,
        "rows":int(len(out)),
        "games":int(out["game_pk"].nunique()),
        "early_source":str(a.early),
        "canonical_later_source":str(a.canonical),
        "composition":{
            "2021_2022":"newly built missing I2 Statcast history",
            "2023_2026":"reused canonical governed Phase 2 artifact",
        },
        "year_2020_usage":"arsenal feature snapshots only; no 2020 PA rows",
    }
    a.manifest_output.parent.mkdir(parents=True,exist_ok=True)
    a.manifest_output.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()

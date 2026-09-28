#!/usr/bin/env python3
"""Build prior-season league PA event rates from a Retrosheet plays ZIP."""
from __future__ import annotations
import argparse, csv, io, json, zipfile
from collections import Counter
from pathlib import Path

EVENTS=("single","double","triple","home_run","walk","hit_by_pitch","strikeout","ball_in_play_out")
OUT_EVENTS={"field_out","force_out","grounded_into_double_play","double_play","triple_play","fielders_choice","fielders_choice_out","sac_fly","sac_bunt","strikeout_double_play"}

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--zip",type=Path,required=True)
    p.add_argument("--season",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    return p.parse_args()

def truth(v):
    return str(v).strip().lower() in {"1","true","t","yes","y"}

def classify(r):
    if not truth(r.get("pa")): return None
    if truth(r.get("single")): return "single"
    if truth(r.get("double")): return "double"
    if truth(r.get("triple")): return "triple"
    if truth(r.get("hr")): return "home_run"
    if truth(r.get("hbp")): return "hit_by_pitch"
    if truth(r.get("walk")): return "walk"
    if truth(r.get("k")): return "strikeout"
    return "ball_in_play_out"

def main():
    a=args(); counts=Counter()
    with zipfile.ZipFile(a.zip) as zf:
        member=f"{a.season}plays.csv"
        if member not in zf.namelist(): raise RuntimeError(f"Missing {member}")
        with zf.open(member) as raw:
            for r in csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8-sig",newline="")):
                if str(r.get("gametype") or "").lower()!="regular": continue
                ev=classify(r)
                if ev: counts[ev]+=1
    n=sum(counts.values())
    if n<=0: raise RuntimeError("No PA events")
    rates={k:counts[k]/n for k in EVENTS}
    out={"version":"retrosheet-prior-league-event-rates-v1","season":a.season,"market_inputs_used":False,"n_pa":n,"event_rates":rates}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=="__main__": main()

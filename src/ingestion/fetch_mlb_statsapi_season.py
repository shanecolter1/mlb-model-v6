#!/usr/bin/env python3
"""Fetch one MLB regular-season date range plus completed game feeds concurrently."""
from __future__ import annotations
import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE="https://statsapi.mlb.com/api"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--start-date",required=True)
    p.add_argument("--end-date",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--workers",type=int,default=16)
    return p.parse_args()

def get_json(url,params=None):
    r=requests.get(url,params=params,timeout=60,headers={"User-Agent":"MLB-V6-Research/0.2","Accept-Encoding":"gzip"})
    r.raise_for_status()
    return r.json()

def main():
    a=parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    schedule=get_json(f"{BASE}/v1/schedule",{
        "sportId":1,"startDate":a.start_date,"endDate":a.end_date,
        "gameTypes":"R","hydrate":"venue,probablePitcher,linescore",
    })
    (a.output_dir/"schedule_range.json").write_text(json.dumps(schedule,indent=2))
    games=[]
    for db in schedule.get("dates",[]):
        for g in db.get("games",[]):
            state=str(g.get("status",{}).get("abstractGameState") or "").lower()
            if state!="final": continue
            games.append({
                "game_id":int(g["gamePk"]),
                "game_date":str(g.get("officialDate") or db.get("date") or ""),
                "status":g.get("status",{}).get("detailedState"),
            })
    feeds=a.output_dir/"feeds"; feeds.mkdir(exist_ok=True)
    def fetch_one(rec):
        path=feeds/f"{rec['game_id']}.json"
        if path.exists():
            return {**rec,"feed_path":str(path),"cached":True}
        payload=get_json(f"{BASE}/v1.1/game/{rec['game_id']}/feed/live")
        path.write_text(json.dumps(payload))
        return {**rec,"feed_path":str(path),"cached":False}
    rows=[]
    with ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        futs=[ex.submit(fetch_one,g) for g in games]
        for i,fut in enumerate(as_completed(futs),1):
            rows.append(fut.result())
            if i%100==0 or i==len(futs): print(f"{i}/{len(futs)} feeds")
    rows.sort(key=lambda r:(r["game_date"],r["game_id"]))
    manifest={
        "version":"mlb-statsapi-season-range-v1",
        "start_date":a.start_date,"end_date":a.end_date,
        "game_types":["R"],"final_games":len(rows),
        "fetched_at":datetime.now(timezone.utc).isoformat(),
        "games":rows,
    }
    (a.output_dir/"fetch_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="games"},indent=2))
if __name__=="__main__": main()

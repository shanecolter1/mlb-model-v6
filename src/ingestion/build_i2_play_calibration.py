#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,io,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path

EVENTS=['single','double','triple','home_run','walk','hit_by_pitch','strikeout','ball_in_play_out']
ATTR='The information used here was obtained free of charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at 20 Sunset Rd., Newark, DE 19711.'

def i(v):
    try:return int(v or 0)
    except:return 0

def event(r):
    if i(r.get('single')): return 'single'
    if i(r.get('double')): return 'double'
    if i(r.get('triple')): return 'triple'
    if i(r.get('hr')): return 'home_run'
    if i(r.get('walk')): return 'walk'
    if i(r.get('hbp')): return 'hit_by_pitch'
    if i(r.get('k')): return 'strikeout'
    return 'ball_in_play_out'

def mask(r,prefix):
    return (1 if r.get(f'br1_{prefix}') else 0) | (2 if r.get(f'br2_{prefix}') else 0) | (4 if r.get(f'br3_{prefix}') else 0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--output',required=True);ap.add_argument('--summary',required=True);a=ap.parse_args()
    trans=defaultdict(Counter); pitches=defaultdict(Counter); event_counts=Counter(); total=0
    for zp in sorted(Path(a.input_dir).glob('20??csvs*.zip')):
        year=int(zp.name[:4])
        with zipfile.ZipFile(zp) as z:
            with z.open(f'{year}plays.csv') as raw:
                for r in csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig',newline='')):
                    if r.get('gametype')!='regular' or i(r.get('pa'))!=1: continue
                    outs_pre=i(r.get('outs_pre')); outs_post=i(r.get('outs_post'))
                    if outs_pre not in (0,1,2) or outs_post < outs_pre or outs_post>3: continue
                    ev=event(r); pre=mask(r,'pre'); post=mask(r,'post'); runs=i(r.get('runs')); oa=outs_post-outs_pre
                    trans[(ev,outs_pre,pre)][(oa,post,runs)] += 1
                    n=max(1,i(r.get('nump'))); pitches[ev][n]+=1; event_counts[ev]+=1; total+=1
    base={}
    for (ev,o,m),cnt in sorted(trans.items()):
        n=sum(cnt.values()); key=f'{ev}|{o}|{m}'
        base[key]=[{'outs_added':oa,'post_mask':post,'runs':runs,'p':c/n,'n':c} for (oa,post,runs),c in sorted(cnt.items())]
    pitch={}
    for ev,cnt in pitches.items():
        n=sum(cnt.values()); pitch[ev]=[{'pitches':k,'p':v/n,'n':v} for k,v in sorted(cnt.items())]
    payload={'version':'2026-08-15-v1','seasons':[2021,2022,2023,2024,2025],'regular_season_plate_appearances':total,'event_counts':dict(event_counts),'base_transition_states':len(base),'base_transitions':base,'pitch_count_pmf':pitch,'normalization':'single/double/triple/home_run/walk/HBP/K; all other PA outcomes -> ball_in_play_out, matching existing event-rate engine','retrosheet_attribution':ATTR}
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    with open(a.summary,'w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['event','n','mean_pitches'])
        for ev in EVENTS:
            cnt=pitches[ev];n=sum(cnt.values());mean=sum(k*v for k,v in cnt.items())/n if n else 0;w.writerow([ev,event_counts[ev],mean])
    print(json.dumps({'pa':total,'states':len(base),'events':dict(event_counts)},indent=2))
if __name__=='__main__':main()

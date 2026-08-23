#!/usr/bin/env python3
"""Season-specific Retrosheet transition calibration with chronological shrinkage validation.

Governance split:
- 2021-2023: estimation
- 2024: shrinkage hyperparameter selection
- 2025: locked chronological validation

Downloads official Retrosheet parsed plays by season and derives model-compatible
PA transitions without sportsbook/market inputs.

Outputs under data/derived/model_calibration/seasonal/.
"""
from __future__ import annotations

import csv, io, json, math, urllib.request, zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/derived/model_calibration/seasonal'
SEASONS = [2021, 2022, 2023, 2024, 2025]
K_GRID = [0, 5, 10, 20, 40, 80, 160, 320, 640, 1280]
EPS = 1e-12


def truth(v):
    return str(v).strip().lower() in {'1','true','t','yes','y'}


def intval(v, default=0):
    try: return int(float(v))
    except Exception: return default


def base_mask(row, suffix):
    return ((1 if str(row.get(f'br1_{suffix}','')).strip() else 0) |
            (2 if str(row.get(f'br2_{suffix}','')).strip() else 0) |
            (4 if str(row.get(f'br3_{suffix}','')).strip() else 0))


def classify_event(r):
    if not truth(r.get('pa')): return None
    if truth(r.get('single')): return 'single'
    if truth(r.get('double')): return 'double'
    if truth(r.get('triple')): return 'triple'
    if truth(r.get('hr')): return 'hr'
    if truth(r.get('walk')) and not truth(r.get('hbp')): return 'bb'
    if truth(r.get('k')): return 'strikeout'
    if truth(r.get('hbp')): return 'hbp'
    pre, post = intval(r.get('outs_pre')), intval(r.get('outs_post'))
    if post > pre: return 'ball_in_play_out'
    return 'other_pa'


def is_regular(gametype):
    g = str(gametype or '').strip().lower().replace('_',' ').replace('-',' ')
    # Current Retrosheet parsed files label regular-season games descriptively.
    return ('regular' in g) or g in {'r','rs','0'}


def fetch_season(year):
    url = f'https://www.retrosheet.org/downloads/plays/{year}plays.zip'
    req = urllib.request.Request(url, headers={'User-Agent':'mlb-model-v6 empirical research'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not csv_names:
            raise RuntimeError(f'No CSV in {url}')
        # Season download normally contains one plays CSV; concatenate if more than one.
        rows = []
        for name in csv_names:
            with z.open(name) as f:
                text = io.TextIOWrapper(f, encoding='utf-8-sig', newline='')
                rows.extend(csv.DictReader(text))
    return url, rows


def season_counts(year):
    url, rows = fetch_season(year)
    states = defaultdict(Counter)
    event_counts = Counter()
    accepted = 0
    gametypes = Counter(str(r.get('gametype','')) for r in rows)
    reg_rows = [r for r in rows if is_regular(r.get('gametype'))]
    if not reg_rows:
        raise RuntimeError(f'{year}: no regular-season rows recognized; observed gametype values={dict(gametypes)}')
    for r in reg_rows:
        ev = classify_event(r)
        if ev not in {'single','double','triple','hr','bb','strikeout','ball_in_play_out'}:
            continue
        outs = intval(r.get('outs_pre'))
        if outs not in (0,1,2): continue
        pre = base_mask(r, 'pre')
        post_outs = intval(r.get('outs_post'))
        outs_added = max(0, min(3-outs, post_outs-outs))
        post = 0 if post_outs >= 3 else base_mask(r, 'post')
        runs = max(0, intval(r.get('runs')))
        states[(ev, outs, pre)][(outs_added, post, runs)] += 1
        event_counts[ev] += 1
        accepted += 1
    return {
        'year': year, 'source_url': url, 'rows_total': len(rows), 'rows_regular': len(reg_rows),
        'accepted_pa': accepted, 'event_counts': dict(event_counts), 'states': states,
        'gametype_values': dict(gametypes),
    }


def merge(year_data, years):
    out = defaultdict(Counter)
    for y in years:
        for k,c in year_data[y]['states'].items(): out[k].update(c)
    return out


def parents(counts):
    eo = defaultdict(Counter)
    e = defaultdict(Counter)
    for (ev,outs,mask), c in counts.items():
        eo[(ev,outs)].update(c); e[ev].update(c)
    return eo,e


def norm(counter):
    n=sum(counter.values())
    return {k:v/n for k,v in counter.items()} if n else {}


def shrunk_dist(key, child, eo, e, k1, k2):
    ev,outs,_ = key
    grand = norm(e.get(ev, Counter()))
    pc = eo.get((ev,outs), Counter()); pn=sum(pc.values())
    outcomes = set(grand) | set(pc) | set(child)
    if not outcomes: return {}
    parent = {}
    for o in outcomes:
        gp=grand.get(o,0.0)
        parent[o]=(pc.get(o,0)+k2*gp)/(pn+k2) if pn+k2 else gp
    cn=sum(child.values())
    d={o:(child.get(o,0)+k1*parent.get(o,0.0))/(cn+k1) if cn+k1 else parent.get(o,0.0) for o in outcomes}
    s=sum(d.values())
    return {o:p/s for o,p in d.items()} if s else d


def score(train_counts, test_counts, k1, k2):
    eo,e=parents(train_counts)
    ll=0.0; n=0
    for key, tc in test_counts.items():
        d=shrunk_dist(key, train_counts.get(key,Counter()), eo,e,k1,k2)
        for o,c in tc.items():
            ll -= c*math.log(max(EPS,d.get(o,0.0))); n += c
    return ll/n if n else None


def tune(train_counts, tune_counts):
    best=None; rows=[]
    for k1 in K_GRID:
        for k2 in K_GRID:
            ll=score(train_counts,tune_counts,k1,k2)
            row={'k_exact_to_event_outs':k1,'k_event_outs_to_event':k2,'logloss_2024':ll}
            rows.append(row)
            if ll is not None and (best is None or ll<best['logloss_2024']): best=row
    return best,rows


def serialize_table(counts,k1,k2):
    eo,e=parents(counts); states={}; samples={}
    for ev in ['out','bb','single','double','triple','hr']:
        for outs in (0,1,2):
            for mask in range(8):
                # bridge generic out by observed K+BIP counts
                if ev=='out':
                    child=Counter(); child.update(counts.get(('strikeout',outs,mask),{})); child.update(counts.get(('ball_in_play_out',outs,mask),{}))
                    # Build temporary pooled parents for generic out separately.
                    generic=defaultdict(Counter)
                    for (hev,ho,hm),c in counts.items():
                        mev='out' if hev in {'strikeout','ball_in_play_out'} else hev
                        generic[(mev,ho,hm)].update(c)
                    geo,ge=parents(generic); key=('out',outs,mask)
                    d=shrunk_dist(key,child,geo,ge,k1,k2)
                else:
                    key=(ev,outs,mask); child=counts.get(key,Counter()); d=shrunk_dist(key,child,eo,e,k1,k2)
                states[f'{ev}|{outs}|{mask}']=[{'outs_added':o[0],'post_mask':o[1],'runs':o[2],'p':p} for o,p in sorted(d.items())]
                samples[f'{ev}|{outs}|{mask}']=sum(child.values())
    return states,samples


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    yd={}
    provenance=[]
    for y in SEASONS:
        d=season_counts(y); yd[y]=d
        serial={'year':y,'source_url':d['source_url'],'rows_total':d['rows_total'],'rows_regular':d['rows_regular'],'accepted_pa':d['accepted_pa'],'event_counts':d['event_counts'],'gametype_values':d['gametype_values']}
        provenance.append(serial)
        # compact season table for stability review
        outstates={}
        for key,c in d['states'].items():
            outstates['|'.join(map(str,key))]=[{'outs_added':o[0],'post_mask':o[1],'runs':o[2],'n':n,'p':n/sum(c.values())} for o,n in sorted(c.items())]
        (OUT/f'transitions_{y}.json').write_text(json.dumps({'metadata':serial,'states':outstates},separators=(',',':')),encoding='utf-8')

    train=merge(yd,[2021,2022,2023]); tune24=merge(yd,[2024]); val25=merge(yd,[2025])
    best,grid=tune(train,tune24)
    if not best: raise RuntimeError('Unable to tune shrinkage')
    # Freeze selected shrinkage then refit empirical counts on development+selection years, evaluate once on 2025.
    refit=merge(yd,[2021,2022,2023,2024])
    ll25=score(refit,val25,best['k_exact_to_event_outs'],best['k_event_outs_to_event'])
    raw25=score(refit,val25,0,0)
    states,samples=serialize_table(refit,best['k_exact_to_event_outs'],best['k_event_outs_to_event'])
    production={
      'version':'seasonal-shrunk-pa-transitions-v1','market_inputs_used':False,
      'training_years':[2021,2022,2023],'selection_year':[2024],'locked_validation_year':[2025],
      'shrinkage':best,'validation_2025':{'logloss_shrunk':ll25,'logloss_raw':raw25,'delta_vs_raw':None if raw25 is None else raw25-ll25},
      'state_definition':'model_event|outs_before|base_mask_before','states':states,'raw_state_sample_sizes_2021_2024':samples,
      'generic_out':'observed-count pool of strikeout + ball_in_play_out','hbp_status':'excluded until separately modeled'
    }
    (OUT/'production_pa_transition_table_shrunk.json').write_text(json.dumps(production,separators=(',',':')),encoding='utf-8')
    with (OUT/'shrinkage_grid_2024.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['k_exact_to_event_outs','k_event_outs_to_event','logloss_2024']); w.writeheader(); w.writerows(grid)
    manifest={
      'component':'PA base/out transition engine','governance_status':'PASS' if ll25 is not None and raw25 is not None and ll25 <= raw25 else 'WARNING',
      'data_source':'Retrosheet official parsed play-by-play season ZIPs','source_urls':[p['source_url'] for p in provenance],
      'development_years':[2021,2022,2023],'hyperparameter_selection_year':[2024],'locked_validation_year':[2025],
      'hyperparameters_frozen_before_2025':True,'selected_shrinkage':best,'validation_2025':production['validation_2025'],
      'market_inputs_used':False,'provenance':provenance,
      'promotion_note':'Production eligibility still requires artifact-integrity audit and live-engine regression tests.'
    }
    (OUT/'model_development_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()

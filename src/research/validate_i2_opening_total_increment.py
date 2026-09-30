#!/usr/bin/env python3
"""Leakage-safe, paired test of full-game opening-total conditioning on I2 replay.

Research only. Rebuilds each exact total bucket from earlier same-rule seasons;
never uses I2 derivative prices or a contemporaneous season to make its prior.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

try:
    from .audit_i2_date_regime import load_season
except ImportError:  # Direct script invocation.
    from audit_i2_date_regime import load_season

TRAIN = {2022: (2021,), 2024: (2023,), 2025: (2023, 2024)}
MASTER_SHA256 = '8512a4fc2fc8e8566d62080bcf006ab6dbdc953e95c3eb8725b55c47a8dafee5'
V04_RELEASE_SHA256 = 'a10284015798da6e797abca00b6df81e9c16cd72e30c7928e22860672b49a820'
TEAM_IDS = dict(zip(
    'LAA AZ BAL BOS CHC CIN CLE COL DET HOU KC LAD WSH NYM OAK PIT SD SEA SF STL TB TEX TOR MIN PHI ATL CWS MIA NYY MIL'.split(),
    [108,109,110,111,112,113,114,115,116,117,118,119,120,121,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,158]))
ALIASES = {'ARI':'AZ','CHW':'CWS','KCR':'KC','WAS':'WSH','ATH':'OAK','SDP':'SD','SFG':'SF','TBR':'TB'}


def schedule_crosswalk(master: pd.DataFrame, schedule_paths: dict[int, Path]) -> dict[str,int]:
    """Post-game identifier reconciliation only; final scores never enter forecasts."""
    by_key=defaultdict(set)
    for year,path in schedule_paths.items():
        payload=json.loads(path.read_text())
        for day in payload['dates']:
            for g in day['games']:
                if g.get('season')!=str(year) or g['status']['abstractGameState']!='Final':
                    continue
                try:
                    k=(g['officialDate'],g['teams']['away']['team']['id'],
                       g['teams']['home']['team']['id'],
                       g['teams']['away']['score'],g['teams']['home']['score'])
                    by_key[k].add(int(g['gamePk']))
                except KeyError:
                    continue
    lookup={}
    for row in master.itertuples():
        away=TEAM_IDS[ALIASES.get(row.away_team_code,row.away_team_code)]
        home=TEAM_IDS[ALIASES.get(row.home_team_code,row.home_team_code)]
        key=(row.game_date,away,home,int(row.away_final_runs),int(row.home_final_runs))
        candidates=by_key.get(key,set())
        if len(candidates)==1:
            lookup[row.retro_game_id]=next(iter(candidates))
    if len(set(lookup.values()))!=len(lookup):
        raise ValueError('Non-unique MLB gamePk crosswalk')
    return lookup


def v04_predictions(tar_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path,'r:gz') as archive:
        name='i2_v04_production_validation/v04_oos_predictions.csv'
        rows=list(csv.DictReader(io.TextIOWrapper(archive.extractfile(name))))
    frame=pd.DataFrame(rows)
    frame['game_id']=frame.game_id.astype(int)
    if frame.game_id.duplicated().any():
        raise ValueError('Duplicated v0.4 game ID')
    return frame


def load_master(path: Path) -> pd.DataFrame:
    fields = ('retro_game_id', 'season', 'game_date', 'benchmark_matched',
              'away_team_code','home_team_code','away_final_runs','home_final_runs',
              'dk_total_open_total', 'inning2_total_runs', 'away_inn2', 'home_inn2')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', newline='') as handle:
        rows = [{k: row[k] for k in fields} for row in csv.DictReader(handle)]
    frame = pd.DataFrame(rows)
    if frame.retro_game_id.duplicated().any():
        raise ValueError('Duplicate Retrosheet game ID in master')
    frame = frame.loc[(frame.benchmark_matched == 'True') &
                      (frame.dk_total_open_total != '')].copy()
    frame['season'] = frame.season.astype(int)
    frame['total'] = pd.to_numeric(frame.dk_total_open_total)
    frame['i2_over'] = pd.to_numeric(frame.inning2_total_runs).gt(0).astype(int)
    frame['top_y_master'] = pd.to_numeric(frame.away_inn2).gt(0).astype(int)
    frame['bottom_y_master'] = pd.to_numeric(frame.home_inn2).gt(0).astype(int)
    if not frame.total.between(6, 11).all():
        frame = frame.loc[frame.total.between(6, 11)].copy()
    if not np.array_equal(frame.i2_over.to_numpy(),
                          (frame.top_y_master | frame.bottom_y_master).to_numpy()):
        raise ValueError('Master half/full outcome mismatch')
    return frame


def probability(raw_over: np.ndarray, bucket_over: np.ndarray,
                broad_over: float) -> np.ndarray:
    raw = np.clip(raw_over, 1e-6, 1-1e-6)
    return expit(logit(np.clip(bucket_over, 1e-6, 1-1e-6)) +
                 logit(raw) - logit(broad_over))


def half_prob(top: np.ndarray, bottom: np.ndarray,
              target_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same equal-half-logit-shift construction as production JS."""
    t0, b0 = logit(np.clip(top, 1e-6, 1-1e-6)), logit(np.clip(bottom, 1e-6, 1-1e-6))
    lo, hi = np.full(len(top), -20.), np.full(len(top), 20.)
    for _ in range(100):
        mid = (lo+hi)/2
        full = 1-(1-expit(t0+mid))*(1-expit(b0+mid))
        lo = np.where(full < target_full, mid, lo)
        hi = np.where(full >= target_full, mid, hi)
    shift = (lo+hi)/2
    return expit(t0+shift), expit(b0+shift)


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p,1e-9,1-1e-9)
    return {'observed':float(y.mean()), 'predicted':float(p.mean()),
            'logloss':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            'brier':float(np.mean((p-y)**2))}


def paired_ci(data: pd.DataFrame, p: np.ndarray, draws: int = 5000) -> dict:
    y=1-data.under_y.to_numpy()
    raw=np.clip(1-data.under_p.to_numpy(),1e-9,1-1e-9)
    p=np.clip(p,1e-9,1-1e-9)
    delta=-(y*np.log(p)+(1-y)*np.log1p(-p)) + y*np.log(raw)+(1-y)*np.log1p(-raw)
    byday=pd.DataFrame({'date':data.date,'delta':delta}).groupby('date').agg(
        total=('delta','sum'),games=('delta','size')).to_numpy()
    rng=np.random.default_rng(260930)
    draws_mean=np.empty(draws)
    for i in range(draws):
        sampled=byday[rng.integers(len(byday),size=len(byday))].sum(axis=0)
        draws_mean[i]=sampled[0]/sampled[1]
    return {'logloss_delta_vs_raw':float(delta.mean()),
            'ci95_day_cluster':[float(x) for x in np.quantile(draws_mean,[.025,.975])]}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--master',type=Path,required=True)
    parser.add_argument('--bundle',action='append',nargs=2,required=True)
    parser.add_argument('--schedule',action='append',nargs=2)
    parser.add_argument('--v04-release',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    bundles={int(year):Path(path) for year,path in args.bundle}
    if set(bundles)!=set(TRAIN):
        raise ValueError(f'Requires bundles for {set(TRAIN)}')
    source_digest=hashlib.sha256(args.master.read_bytes()).hexdigest()
    if source_digest!=MASTER_SHA256:
        raise ValueError('Historical master release checksum mismatch')
    control_digest=hashlib.sha256(args.v04_release.read_bytes()).hexdigest() if args.v04_release else None
    if control_digest and control_digest!=V04_RELEASE_SHA256:
        raise ValueError('v0.4 validation release checksum mismatch')
    master=load_master(args.master)
    if bool(args.schedule)!=bool(args.v04_release):
        raise ValueError('Schedule snapshots and v0.4 release must be supplied together')
    v04=v04_predictions(args.v04_release) if args.v04_release else None
    crosswalk=schedule_crosswalk(master,{int(y):Path(p) for y,p in args.schedule}) if v04 is not None else {}
    result={'version':'i2-opening-total-increment-v1',
            'source_sha256':source_digest,
            'source_eligible_rows':len(master),
            'v04_validation_release_sha256':control_digest,
            'crosswalk_matched_games':len(crosswalk),
            'market_inputs_used':'opening full-game total point only',
            'i2_prices_used':False,'method':'Production exact-bucket logit recentering and equal-half shift',
            'rule_eras':{'2022':'2021 pre-rule training',
                          '2024':'2023 post-rule training',
                          '2025':'2023–2024 post-rule training'},
            'excluded_years':{'2023':'No earlier post-rule total prior',
                              '2026':'No canonical game-level DK opening-total archive; previously inspected skew'},
            'seasons':{}}
    for year, train_years in TRAIN.items():
        train=master.loc[master.season.isin(train_years)]
        if not len(train) or train.season.nunique()!=len(train_years):
            raise ValueError('Missing historical training seasons')
        bucket=train.groupby('total').i2_over.agg(['mean','size'])
        broad=float(train.i2_over.mean())
        replay=load_season(bundles[year],year)
        matched=replay.merge(master.loc[master.season==year,
                 ['retro_game_id','game_date','total','i2_over','top_y_master','bottom_y_master']],
                 left_on='gid',right_on='retro_game_id',how='inner',validate='one_to_one')
        matched=matched.loc[matched.total.isin(bucket.index)].copy()
        if matched.empty:
            raise ValueError(f'No exact-bucket games for {year}')
        if not (matched.date.eq(matched.game_date).all() and
                (1-matched.under_y).eq(matched.i2_over).all() and
                matched.top_y.eq(matched.top_y_master).all() and
                matched.bottom_y.eq(matched.bottom_y_master).all()):
            raise ValueError(f'Game/date/outcome mismatch in {year}')
        raw_over=1-matched.under_p.to_numpy()
        corrected=probability(raw_over,matched.total.map(bucket['mean']).to_numpy(),broad)
        top,bottom=half_prob(matched.top_p.to_numpy(),matched.bottom_p.to_numpy(),corrected)
        y=matched.i2_over.to_numpy()
        result['seasons'][str(year)]={
            'training_seasons':train_years,'training_games':len(train),
            'training_broad_over':broad,
            'training_buckets':{str(k):{'n':int(v['size']),'over':float(v['mean'])}
                                for k,v in bucket.iterrows()},
            'replay_games':len(replay),'matched_eligible':len(matched),
            'unmatched_or_outside_training_buckets':len(replay)-len(matched),
            'first_date':matched.date.min(),'last_date':matched.date.max(),
            'full_i2':{'raw':scores(y,raw_over),'total_conditioned':scores(y,corrected),
                       'paired_day_ci':paired_ci(matched,corrected)},
            'top':{'raw':scores(matched.top_y.to_numpy(),matched.top_p.to_numpy()),
                   'total_conditioned':scores(matched.top_y.to_numpy(),top)},
            'bottom':{'raw':scores(matched.bottom_y.to_numpy(),matched.bottom_p.to_numpy()),
                      'total_conditioned':scores(matched.bottom_y.to_numpy(),bottom)},
        }
        if v04 is not None:
            matched['game_id']=matched.gid.map(crosswalk)
            controls=matched.merge(v04[['game_id','p_under_local_cv','actual_under']],
                                   on='game_id',how='inner',validate='one_to_one')
            if len(controls):
                if not (controls.under_y.to_numpy()==pd.to_numeric(controls.actual_under).to_numpy()).all():
                    raise ValueError('v0.4 outcome mismatch')
                # Merge indices are reset; identify the parent row by unique retro game ID.
                pred=dict(zip(matched.gid,corrected))
                cond=controls.gid.map(pred).to_numpy(dtype=float)
                yy=controls.under_y.to_numpy()
                result['seasons'][str(year)]['v04_release_matched']={
                    'status':'Historical OOS validation candidate, not a timestamped frozen production forecast',
                    'games':len(controls),
                    'vnext_raw':scores(yy,controls.under_p.to_numpy()),
                    'vnext_total':scores(yy,1-cond),
                    'v04_local_cv':scores(yy,pd.to_numeric(controls.p_under_local_cv).to_numpy()),
                }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

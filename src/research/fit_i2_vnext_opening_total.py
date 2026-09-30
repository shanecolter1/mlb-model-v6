#!/usr/bin/env python3
"""Fit vNext-specific top/bottom total slopes on prior post-rule seasons.

Research-only post-freeze diagnostic; vNext pre-freeze inputs remain baseball-only.
The total feature and coefficient are learned only from earlier vNext replay games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

try:
    from .audit_i2_date_regime import load_season
    from .validate_i2_opening_total_increment import load_master, MASTER_SHA256
except ImportError:
    from audit_i2_date_regime import load_season
    from validate_i2_opening_total_increment import load_master, MASTER_SHA256

TRAIN = {2024: (2023,), 2025: (2023, 2024)}
RIDGE = 0.01  # Fixed before evaluation, applied only to the total slope.
CENTER = 8.5


def joined_season(master: pd.DataFrame, bundle: Path, year: int) -> pd.DataFrame:
    replay = load_season(bundle, year)
    cols = ['retro_game_id','game_date','total','i2_over','top_y_master','bottom_y_master']
    cohort=master.loc[master.season==year,cols]
    data=replay.merge(cohort,left_on='gid',right_on='retro_game_id',
                      how='inner',validate='one_to_one')
    if data.empty:
        raise ValueError(f'No matched totals for {year}')
    if not (data.date.eq(data.game_date).all() and
            (1-data.under_y).eq(data.i2_over).all() and
            data.top_y.eq(data.top_y_master).all() and
            data.bottom_y.eq(data.bottom_y_master).all()):
        raise ValueError(f'Game/date/outcome mismatch in {year}')
    return data


def fit_half(data: pd.DataFrame, half: str, with_total: bool) -> np.ndarray:
    x = np.column_stack((np.ones(len(data)),
                         data.total.to_numpy(dtype=float)-CENTER))
    if not with_total:
        x=x[:,:1]
    eta0=logit(np.clip(data[f'{half}_p'].to_numpy(),1e-6,1-1e-6))
    y=data[f'{half}_y'].to_numpy()
    def objective(beta):
        eta=eta0+x@beta
        penalty=np.r_[0,beta[1:]]
        return (float(np.mean(np.logaddexp(0,eta)-y*eta)+RIDGE*penalty@penalty),
                x.T@(expit(eta)-y)/len(y)+2*RIDGE*penalty)
    result=minimize(objective,np.zeros(x.shape[1]),jac=True,method='BFGS',
                    options={'gtol':1e-10})
    if not result.success and np.max(np.abs(objective(result.x)[1]))>1e-7:
        raise RuntimeError(result.message)
    return result.x if with_total else np.r_[result.x,0.]


def half_predictions(data: pd.DataFrame, half: str, beta: np.ndarray) -> np.ndarray:
    eta=logit(np.clip(data[f'{half}_p'].to_numpy(),1e-6,1-1e-6))
    return expit(eta+beta[0]+beta[1]*(data.total.to_numpy()-CENTER))


def metric(y: np.ndarray,p: np.ndarray) -> dict:
    p=np.clip(p,1e-9,1-1e-9)
    return {'actual':float(y.mean()),'predicted':float(p.mean()),
            'logloss':float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),
            'brier':float(np.mean((p-y)**2))}


def predictions(data: pd.DataFrame, betas: dict[str,np.ndarray] | None) -> dict:
    if betas is None:
        top=data.top_p.to_numpy()
        bottom=data.bottom_p.to_numpy()
        under=data.under_p.to_numpy()
    else:
        top=half_predictions(data,'top',betas['top'])
        bottom=half_predictions(data,'bottom',betas['bottom'])
        under=(1-top)*(1-bottom)
    return {'top':top,'bottom':bottom,'under':under}


def scores(data: pd.DataFrame, pred: dict) -> dict:
    return {h:metric(data[f'{h}_y'].to_numpy(),pred[h]) for h in ('top','bottom','under')}


def paired_ci(data: pd.DataFrame, target: np.ndarray,
              reference: np.ndarray, draws: int = 5000) -> dict:
    y=data.under_y.to_numpy()
    a=np.clip(target,1e-9,1-1e-9)
    b=np.clip(reference,1e-9,1-1e-9)
    delta=-(y*np.log(a)+(1-y)*np.log1p(-a)) + y*np.log(b)+(1-y)*np.log1p(-b)
    daily=pd.DataFrame({'date':data.date,'delta':delta}).groupby('date').agg(
        total=('delta','sum'),n=('delta','size')).to_numpy()
    rng=np.random.default_rng(260930)
    samples=np.empty(draws)
    for i in range(draws):
        v=daily[rng.integers(len(daily),size=len(daily))].sum(axis=0)
        samples[i]=v[0]/v[1]
    return {'delta_logloss':float(delta.mean()),
            'ci95_day_cluster':[float(x) for x in np.quantile(samples,[.025,.975])]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master',type=Path,required=True)
    p.add_argument('--bundle',action='append',nargs=2,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if hashlib.sha256(args.master.read_bytes()).hexdigest()!=MASTER_SHA256:
        raise ValueError('Historical master checksum mismatch')
    bundles={int(y):Path(path) for y,path in args.bundle}
    if set(bundles)!={2023,2024,2025}:
        raise ValueError('Requires 2023–2025 vNext replays')
    master=load_master(args.master)
    seasons={y:joined_season(master,path,y) for y,path in bundles.items()}
    result={'version':'i2-vnext-trained-total-slope-v1',
            'scope':'Research post-freeze diagnostic; no production model changed',
            'source_sha256':MASTER_SHA256,'ridge_slope':RIDGE,'total_center':CENTER,
            'market_input':'DraftKings pregame full-game opening total point only',
            'i2_price_used':False,'regime':'2023+ post-rule only',
            'excluded_years':{'2022':'No 2021 vNext replay to train pre-rule effect',
                              '2023':'Training foundation; no prior post-rule vNext season',
                              '2026':'No canonical opening-total archive and outcomes previously inspected'},
            'seasons':{}}
    for year,train_years in TRAIN.items():
        train=pd.concat([seasons[y] for y in train_years],ignore_index=True)
        test=seasons[year]
        specs={name:{h:fit_half(train,h,with_total=(name=='vnext_total'))
                     for h in ('top','bottom')}
               for name in ('constant_half','vnext_total')}
        pred={'raw':predictions(test,None),
              **{name:predictions(test,beta) for name,beta in specs.items()}}
        result['seasons'][str(year)]={
            'training_seasons':train_years,'training_games':len(train),
            'matched_test_games':len(test),'first_test_date':test.date.min(),
            'last_test_date':test.date.max(),
            'coefficients':{name:{h:{'intercept':float(b[0]),
                                      'total_slope_per_run':float(b[1])}
                                   for h,b in halves.items()}
                            for name,halves in specs.items()},
            'scores':{name:scores(test,forecast) for name,forecast in pred.items()},
            'total_vs_raw':paired_ci(test,pred['vnext_total']['under'],pred['raw']['under']),
            'total_vs_constant_half':paired_ci(test,pred['vnext_total']['under'],
                                                pred['constant_half']['under']),
        }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

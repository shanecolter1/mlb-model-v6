#!/usr/bin/env python3
"""Joint, half-specific temporal sensitivity fit for market-isolated I2.

The Savant drag series is retrospective; this is descriptive validation only.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from audit_i2_date_regime import load_season

TEAM_IDS = dict(zip(
    'LAA AZ BAL BOS CHC CIN CLE COL DET HOU KC LAD WSH NYM OAK PIT SD SEA SF STL TB TEX TOR MIN PHI ATL CWS MIA NYY MIL'.split(),
    [108,109,110,111,112,113,114,115,116,117,118,119,120,121,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,158]))
ALIASES = {'ARI':'AZ','ANA':'LAA','CHA':'CWS','CHN':'CHC','KCA':'KC',
           'LAN':'LAD','NYA':'NYY','NYN':'NYM','SDN':'SD','SFN':'SF',
           'SLN':'STL','TBA':'TB','WAS':'WSH','ATH':'OAK'}
FEATURES = ('intercept','early_28','late_42','hr_bip_28','walk_pa_28',
            'drag_cd_28','late_offense_contender','late_defense_contender')
RIDGES = (0.001, 0.01, 0.1, 1.0)


def team_id(code: str) -> int:
    return TEAM_IDS[ALIASES.get(code, code)]


def read_year(year: int, bundle: Path, temporal: Path) -> pd.DataFrame:
    data = load_season(bundle, year)
    with zipfile.ZipFile(bundle) as z:
        games = {str(g['gid']): g for g in json.loads(
            z.read(f'replay_{year}_inputs.json'))['games']}
    for side in ('away','home'):
        data[f'{side}_id'] = [int(games[gid][f'{side}_team_id'])
                              if games[gid].get(f'{side}_team_id') else
                              team_id(games[gid][f'{side}_team_retro']) for gid in data.gid]
    league = pd.read_csv(temporal / f'league_{year}_asof.csv')
    drag = pd.read_csv(temporal / f'drag_{year}_asof.csv')
    standings = pd.read_csv(temporal / f'standings_{year}_asof.csv')
    data = data.merge(league[['game_date','prior_28d_hr_per_bip',
                              'prior_28d_walk_rate']], left_on='date',right_on='game_date',
                      validate='many_to_one').drop(columns='game_date')
    data = data.merge(drag[['game_date','drag_cd_28d']],left_on='date',right_on='game_date',
                      validate='many_to_one').drop(columns='game_date')
    for side in ('away','home'):
        s = standings[['game_date','team_id','wins','losses']].rename(
            columns={'game_date':'date','team_id':f'{side}_id',
                     'wins':f'{side}_wins','losses':f'{side}_losses'})
        data = data.merge(s,on=['date',f'{side}_id'],how='left',validate='many_to_one')
    day = pd.to_datetime(data.date)
    data['early'] = np.maximum(0,1-(day-day.min()).dt.days/28)
    data['late'] = np.maximum(0,1-(day.max()-day).dt.days/42)
    for side in ('away','home'):
        n = data[f'{side}_wins'] + data[f'{side}_losses']
        pct = data[f'{side}_wins'] / n.replace(0,np.nan)
        data[f'{side}_contender'] = np.maximum(0,1-np.abs(pct-.5)/.15).fillna(0)
    return data


def matrix(data: pd.DataFrame, half: str) -> np.ndarray:
    offense, defense = ('away','home') if half=='top' else ('home','away')
    return np.column_stack((
        np.ones(len(data)), data.early, data.late,
        (data.prior_28d_hr_per_bip.fillna(.045)-.045)/.01,
        (data.prior_28d_walk_rate.fillna(.08)-.08)/.02,
        (data.drag_cd_28d.fillna(.345)-.345)/.005,
        data.late * data[f'{offense}_contender'],
        data.late * data[f'{defense}_contender']))


def fit(data: pd.DataFrame, half: str, ridge: float, constant: bool = False) -> np.ndarray:
    x = matrix(data,half)
    if constant:
        x = x[:, :1]
    y = data[f'{half}_y'].to_numpy()
    eta0 = logit(np.clip(data[f'{half}_p'].to_numpy(),1e-6,1-1e-6))
    def obj(beta):
        eta = eta0+x@beta
        p=expit(eta)
        penalty=np.r_[0,beta[1:]]
        return (np.mean(np.logaddexp(0,eta)-y*eta)+ridge*(penalty@penalty),
                x.T@(p-y)/len(y)+2*ridge*penalty)
    result=minimize(obj,np.zeros(x.shape[1]),jac=True,method='BFGS',
                    options={'gtol':1e-9})
    if not result.success and np.max(np.abs(obj(result.x)[1]))>1e-6:
        raise RuntimeError(result.message)
    return np.r_[result.x, np.zeros(len(FEATURES)-1)] if constant else result.x


def under_prob(data: pd.DataFrame, top_beta, bottom_beta) -> np.ndarray:
    halves=[]
    for half,beta in (('top',top_beta),('bottom',bottom_beta)):
        p=np.clip(data[f'{half}_p'].to_numpy(),1e-6,1-1e-6)
        halves.append(expit(logit(p)+matrix(data,half)@beta))
    return (1-halves[0])*(1-halves[1])


def paired_ci(data: pd.DataFrame, top_beta, bottom_beta, draws=5000) -> dict:
    y=data.under_y.to_numpy()
    raw=np.clip(data.under_p.to_numpy(),1e-9,1-1e-9)
    candidate=np.clip(under_prob(data,top_beta,bottom_beta),1e-9,1-1e-9)
    delta=-(y*np.log(candidate)+(1-y)*np.log1p(-candidate)) + (
        y*np.log(raw)+(1-y)*np.log1p(-raw))
    daily=pd.DataFrame({'date':data.date,'delta':delta}).groupby('date').agg(
        total=('delta','sum'),games=('delta','size')).to_numpy()
    rng=np.random.default_rng(260929)
    samples=[]
    for _ in range(draws):
        draw=daily[rng.integers(len(daily),size=len(daily))].sum(axis=0)
        samples.append(draw[0]/draw[1])
    return {'logloss_delta_vs_raw':float(np.mean(delta)),
            'ci95_day_cluster':[float(x) for x in np.quantile(samples,[.025,.975])]}


def score(data: pd.DataFrame, top_beta=None, bottom_beta=None) -> dict:
    pred={}
    for half,beta in (('top',top_beta),('bottom',bottom_beta)):
        p=np.clip(data[f'{half}_p'].to_numpy(),1e-6,1-1e-6)
        pred[half]=expit(logit(p)+matrix(data,half)@beta) if beta is not None else p
    under=(1-pred['top'])*(1-pred['bottom'])
    # Raw baseline uses the archived full-inning simulation, not rounded half product.
    if top_beta is None and bottom_beta is None:
        under=data.under_p.to_numpy()
    y=data.under_y.to_numpy()
    def metric(target,p):
        p=np.clip(p,1e-9,1-1e-9)
        return {'observed':float(np.mean(target)), 'predicted':float(np.mean(p)),
                'logloss':float(np.mean(-target*np.log(p)-(1-target)*np.log1p(-p))),
                'brier':float(np.mean((target-p)**2))}
    return {'games':len(data),'under':metric(y,under),
            'top':metric(data.top_y.to_numpy(),pred['top']),
            'bottom':metric(data.bottom_y.to_numpy(),pred['bottom'])}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',action='append',nargs=2,required=True)
    p.add_argument('--temporal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    bundles={int(year):Path(path) for year,path in args.bundle}
    if set(bundles)!={2023,2024,2025,2026}:
        raise ValueError('Requires 2023–2026 bundles')
    years={y:read_year(y,path,args.temporal) for y,path in bundles.items()}
    choices=[]
    for ridge in RIDGES:
        folds={}
        for hold in (2023,2024,2025):
            train=pd.concat([years[y] for y in (2023,2024,2025) if y!=hold])
            betas={h:fit(train,h,ridge) for h in ('top','bottom')}
            folds[str(hold)]=score(years[hold],betas['top'],betas['bottom'])
        loss=sum(v['games']*v['under']['logloss'] for v in folds.values()) / sum(
            v['games'] for v in folds.values())
        choices.append({'ridge':ridge,'loso_under_logloss':loss,'folds':folds})
    best=min(choices,key=lambda c:c['loso_under_logloss'])
    constant_folds={}
    for hold in (2023,2024,2025):
        tr=pd.concat([years[y] for y in (2023,2024,2025) if y!=hold])
        b={h:fit(tr,h,0,constant=True) for h in ('top','bottom')}
        constant_folds[str(hold)]=score(years[hold],b['top'],b['bottom'])
    train=pd.concat([years[y] for y in (2023,2024,2025)])
    betas={h:fit(train,h,best['ridge']) for h in ('top','bottom')}
    constant_betas={h:fit(train,h,0,constant=True) for h in ('top','bottom')}
    result={'version':'joint-temporal-sensitivity-v1','market_inputs_used':False,
            'candidate_selection':'2023–2025 leave-one-season-out full-I2 logloss',
            'holdout_caution':'2026 outcomes previously inspected; descriptive replication',
            'drag_caution':'Retrospective Savant snapshot; historical publication timing unverified',
            'features':FEATURES,'choices':choices,'selected_ridge':best['ridge'],
            'constant_half_loso':constant_folds,
            'coefficients':{h:dict(zip(FEATURES,map(float,b))) for h,b in betas.items()},
            'development_raw':{str(y):score(years[y]) for y in (2023,2024,2025)},
            'replication_2026':{'raw':score(years[2026]),
                                'recombined_identity':score(years[2026],np.zeros(len(FEATURES)),np.zeros(len(FEATURES))),
                                'constant_half':score(years[2026],constant_betas['top'],constant_betas['bottom']),
                                'joint':score(years[2026],betas['top'],betas['bottom']),
                                'paired_day_ci_joint_vs_raw':paired_ci(years[2026],betas['top'],betas['bottom'])}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

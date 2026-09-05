#!/usr/bin/env python3
"""Locate the source of full-inning zero-mass anomalies at the half-inning level.

For each DraftKings opening total x inning x half, compare empirical P(0) with Poisson
and moment-fit negative-binomial expectations using the half-inning mean/variance.
Historical research only; no inning-market prices. I1-I8 are clean ranking cells.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True,exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True,exist_ok=True)
MIN_N=300
TARGETS={(8.5,8),(8.5,6),(9.0,6),(8.5,3),(8.0,3)}

def nb_p0(mu,var):
    if not np.isfinite(mu) or mu<0:return np.nan
    if mu==0:return 1.0
    if not np.isfinite(var) or var<=mu:return math.exp(-mu)
    r=mu*mu/(var-mu)
    return (r/(r+mu))**r

def one(x):
    n=len(x); mu=x.mean(); var=x.var(ddof=1); p0=(x==0).mean(); pscore=1-p0
    return {'n':n,'mean_runs':mu,'variance':var,'p0_pct':100*p0,'p1_pct':100*(x==1).mean(),'p2plus_pct':100*(x>=2).mean(),
            'mean_runs_given_score':x[x>0].mean() if (x>0).any() else np.nan,
            'multi_run_given_score_pct':100*(x[x>0]>=2).mean() if (x>0).any() else np.nan,
            'poisson_p0_pct':100*math.exp(-mu),'poisson_residual_pp':100*(p0-math.exp(-mu)),
            'nb_p0_pct':100*nb_p0(mu,var),'nb_residual_pp':100*(p0-nb_p0(mu,var))}

df=pd.read_csv(URL,compression='gzip',low_memory=False)
if 'benchmark_matched' in df.columns:df=df[df.benchmark_matched==True].copy()
df['pregame_total']=pd.to_numeric(df.dk_total_open_total,errors='coerce');df['season']=pd.to_numeric(df.season,errors='coerce')
rows=[]; seasons=[]
for inn in range(1,10):
  for half,col in [('top',f'away_inn{inn}'),('bottom',f'home_inn{inn}')]:
    if col not in df.columns:continue
    z=df[['season','pregame_total',col]].copy();z['runs']=pd.to_numeric(z[col],errors='coerce');z=z[z.runs.notna() & z.pregame_total.notna()]
    for t,g in z.groupby('pregame_total'):
      x=g.runs.to_numpy(float)
      if len(x)<MIN_N:continue
      rec=one(x);rec.update({'pregame_total':t,'inning':inn,'half':half,'clean_i1_i8':inn<=8,'target_cell':(float(t),inn) in TARGETS});rows.append(rec)
      for s,sg in g.groupby('season'):
        xx=sg.runs.to_numpy(float)
        if len(xx)>=30:
          rr=one(xx);rr.update({'pregame_total':t,'inning':inn,'half':half,'season':int(s)});seasons.append(rr)
r=pd.DataFrame(rows);s=pd.DataFrame(seasons)
if len(s):
  st=s.groupby(['pregame_total','inning','half']).agg(seasons=('season','nunique'),season_p0_sd_pp=('p0_pct','std'),season_nb_residual_mean_pp=('nb_residual_pp','mean'),season_nb_residual_sd_pp=('nb_residual_pp','std')).reset_index()
  r=r.merge(st,on=['pregame_total','inning','half'],how='left')
r.to_csv(OUT/'half_inning_zero_mass_shape_2021_2025.csv',index=False);s.to_csv(OUT/'half_inning_zero_mass_shape_by_season_2021_2025.csv',index=False)
# Pair top/bottom for target interpretation.
p=r[r.target_cell].pivot(index=['pregame_total','inning'],columns='half',values=['p0_pct','mean_runs','nb_residual_pp','season_p0_sd_pp']).reset_index();p.columns=['_'.join([str(x) for x in c if str(x)]) for c in p.columns];p.to_csv(OUT/'half_inning_target_top_bottom_comparison_2021_2025.csv',index=False)
clean=r[(r.clean_i1_i8)&(r.n>=500)].copy();clean['persistent_half_shape']=clean.nb_residual_pp.abs()/(1+clean.season_nb_residual_sd_pp)
top=clean.sort_values(['persistent_half_shape','n'],ascending=[False,False]).head(25)
cols=['pregame_total','inning','half','n','mean_runs','p0_pct','nb_p0_pct','nb_residual_pp','multi_run_given_score_pct','season_p0_sd_pp','season_nb_residual_sd_pp','persistent_half_shape']
md=['# Half-Inning Zero-Mass Shape Analysis (2021–2025)','',
'## Finding sought','The preceding full-inning dependence test showed that most priority cells have essentially independent top/bottom scoring. This analysis therefore asks which **half-inning marginal distributions** create the excess full-inning zero probability.','',
'## Priority target top/bottom decomposition','',p.to_markdown(index=False,floatfmt='.4f'),'','## Strongest persistent half-inning distribution-shape departures — I1–I8, N >= 500','',top[cols].to_markdown(index=False,floatfmt='.4f'),'','## Interpretation','- A residual present in both halves suggests a general inning/total structure rather than home/away asymmetry.','- A residual concentrated in only one half suggests team-role/home-away/run-allocation structure.','- This is a mechanism diagnostic, not evidence of sportsbook EV. Actual DK/FD inning prices remain the final test.']
(AN/'HALF_INNING_ZERO_MASS_SHAPE_2021_2025.md').write_text('\n'.join(md),encoding='utf-8');print('\n'.join(md))

#!/usr/bin/env python3
"""Analyze full-inning U0.5 through half-inning dependence and favorite-strength asymmetry.

Historical research only. Uses DraftKings opening full-game total and opening moneyline as
conditioning variables; no inning-market prices are used. I1-I8 are the clean ranking universe.
I9 is diagnostic only because bottom-nine non-play creates structural censoring/settlement issues.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd

URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True,exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True,exist_ok=True)
MIN_N=100
TARGETS={(8.5,8),(8.5,6),(9.0,6),(8.5,3),(8.0,3)}


def implied(o):
    if pd.isna(o): return np.nan
    o=float(o)
    return (-o)/((-o)+100.0) if o<0 else 100.0/(o+100.0)


def novig_fav_prob(row, home_col, away_col):
    hp=implied(row[home_col]); ap=implied(row[away_col])
    if not np.isfinite(hp) or not np.isfinite(ap) or hp+ap<=0: return np.nan
    h=hp/(hp+ap)
    return max(h,1-h)


def fav_bucket(p):
    if not np.isfinite(p): return None
    if p < .55: return '50-55%'
    if p < .60: return '55-60%'
    if p < .65: return '60-65%'
    if p < .70: return '65-70%'
    return '70%+'


def season_sd(g):
    q=g.groupby('season').under.mean()
    return 100*q.std(ddof=1) if len(q)>=2 else np.nan


def summarize(g):
    n=len(g)
    p_top0=(g.top_runs==0).mean(); p_bot0=(g.bottom_runs==0).mean(); p_full0=g.under.mean()
    indep=p_top0*p_bot0
    top_sc=(g.top_runs>0).astype(int); bot_sc=(g.bottom_runs>0).astype(int)
    corr=top_sc.corr(bot_sc) if top_sc.nunique()>1 and bot_sc.nunique()>1 else np.nan
    scored=g[g.full_runs>0]
    return {
      'n':n,'under_pct':100*p_full0,'top_zero_pct':100*p_top0,'bottom_zero_pct':100*p_bot0,
      'independence_p0_pct':100*indep,'joint_zero_dependence_pp':100*(p_full0-indep),
      'top_bottom_scoring_phi':corr,'mean_runs':g.full_runs.mean(),
      'multi_run_given_score_pct':100*(scored.full_runs>=2).mean() if len(scored) else np.nan,
      'threeplus_given_score_pct':100*(scored.full_runs>=3).mean() if len(scored) else np.nan,
      'season_under_sd_pp':season_sd(g)
    }


df=pd.read_csv(URL,compression='gzip',low_memory=False)
if 'benchmark_matched' in df.columns: df=df[df.benchmark_matched==True].copy()
df['season']=pd.to_numeric(df['season'],errors='coerce')
total_col='dk_total_open_total'; home_ml='dk_moneyline_open_homeOdds'; away_ml='dk_moneyline_open_awayOdds'
for c in [total_col,home_ml,away_ml]:
    if c not in df.columns: raise RuntimeError(f'missing required {c}')
df['pregame_total']=pd.to_numeric(df[total_col],errors='coerce')
df['favorite_novig_prob']=df.apply(lambda r: novig_fav_prob(r,home_ml,away_ml),axis=1)
df['favorite_strength_bucket']=df.favorite_novig_prob.map(fav_bucket)

rows=[]; bucket_rows=[]; season_rows=[]
for inn in range(1,10):
    a=f'away_inn{inn}'; h=f'home_inn{inn}'
    if a not in df.columns or h not in df.columns: continue
    x=df[['season','pregame_total','favorite_novig_prob','favorite_strength_bucket',a,h]].copy()
    x['top_runs']=pd.to_numeric(x[a],errors='coerce'); x['bottom_runs']=pd.to_numeric(x[h],errors='coerce')
    # Clean full-inning semantics: require both halves played. This excludes skipped B9 from joint dependence.
    x=x[x.top_runs.notna() & x.bottom_runs.notna() & x.pregame_total.notna()].copy()
    x['full_runs']=x.top_runs+x.bottom_runs; x['under']=(x.full_runs==0).astype(int)
    for t,g in x.groupby('pregame_total'):
        if len(g)<MIN_N: continue
        z=summarize(g); rows.append({'pregame_total':t,'inning':inn,**z,'clean_i1_i8':inn<=8,'target_cell':(float(t),inn) in TARGETS})
        for s,sg in g.groupby('season'):
            if len(sg)>=30:
                ss=summarize(sg); season_rows.append({'pregame_total':t,'inning':inn,'season':int(s),**ss})
        for b,bg in g[g.favorite_strength_bucket.notna()].groupby('favorite_strength_bucket'):
            if len(bg)<MIN_N: continue
            zz=summarize(bg); bucket_rows.append({'pregame_total':t,'inning':inn,'favorite_strength_bucket':b,'mean_favorite_novig_prob':bg.favorite_novig_prob.mean(),**zz})

r=pd.DataFrame(rows); b=pd.DataFrame(bucket_rows); s=pd.DataFrame(season_rows)
r.to_csv(OUT/'inning_half_dependence_overall_2021_2025.csv',index=False)
b.to_csv(OUT/'inning_favorite_asymmetry_buckets_2021_2025.csv',index=False)
s.to_csv(OUT/'inning_half_dependence_by_season_2021_2025.csv',index=False)

# Within-target favorite spread: max minus min U rate across eligible strength buckets.
spread=[]
for (t,i),g in b.groupby(['pregame_total','inning']):
    if len(g)>=2:
        spread.append({'pregame_total':t,'inning':i,'eligible_buckets':len(g),'min_under_pct':g.under_pct.min(),'max_under_pct':g.under_pct.max(),'under_spread_pp':g.under_pct.max()-g.under_pct.min(),'dependence_spread_pp':g.joint_zero_dependence_pp.max()-g.joint_zero_dependence_pp.min()})
sp=pd.DataFrame(spread); sp.to_csv(OUT/'inning_favorite_asymmetry_spread_2021_2025.csv',index=False)

target=r[r.target_cell].sort_values(['pregame_total','inning'])
tb=b.merge(target[['pregame_total','inning']].drop_duplicates(),on=['pregame_total','inning'],how='inner').sort_values(['pregame_total','inning','mean_favorite_novig_prob'])
cols=['pregame_total','inning','n','under_pct','top_zero_pct','bottom_zero_pct','independence_p0_pct','joint_zero_dependence_pp','top_bottom_scoring_phi','mean_runs','multi_run_given_score_pct','season_under_sd_pp']
md=['# Inning Half-Dependence & Favorite-Asymmetry Analysis (2021–2025)','',
'## Question','Does full-inning U0.5 contain structure that a mean-run model misses because (a) the two half-innings are dependent or (b) the same full-game total hides materially different favorite/underdog run-allocation shapes?','',
'Full-inning zero probability is decomposed as observed P(top=0,bottom=0) versus the independence benchmark P(top=0)×P(bottom=0). Positive `joint_zero_dependence_pp` means scoreless halves cluster within the same games. Favorite strength is the no-vig probability of the stronger side from DraftKings opening moneylines.','',
'**I1–I8 are clean. I9 remains diagnostic only. No inning-market prices are inputs.**','',
'## Priority target cells','',target[cols].to_markdown(index=False,floatfmt='.4f'),'','## Favorite-strength decomposition of priority cells','',tb[['pregame_total','inning','favorite_strength_bucket','n','mean_favorite_novig_prob','under_pct','joint_zero_dependence_pp','multi_run_given_score_pct','season_under_sd_pp']].to_markdown(index=False,floatfmt='.4f'),'','## Interpretation','- If joint-zero dependence is near zero, the full-inning anomaly is largely explained by the two half-inning marginals rather than an extra game-level coupling effect.','- If U0.5 changes materially with favorite strength at the same game total, the full-game total is hiding run-allocation asymmetry; moneyline/team-strength information may improve inning pricing.','- If neither effect is meaningful and stable, the empirical total×inning base rate should remain the preferred prior rather than adding complexity.']
(AN/'INNING_HALF_DEPENDENCE_FAVORITE_ASYMMETRY_2021_2025.md').write_text('\n'.join(md),encoding='utf-8')
print('\n'.join(md))

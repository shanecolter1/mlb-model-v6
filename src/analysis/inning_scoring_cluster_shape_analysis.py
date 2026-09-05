import math
from pathlib import Path
import numpy as np
import pandas as pd

URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True,exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True,exist_ok=True)
MIN_N=300

def nb_p0(mu,var):
    if not np.isfinite(mu) or mu<0: return np.nan
    if mu==0: return 1.0
    if not np.isfinite(var) or var<=mu: return math.exp(-mu)
    r=mu*mu/(var-mu)
    return (r/(r+mu))**r

def rowstats(x):
    n=len(x); mu=x.mean(); var=x.var(ddof=1) if n>1 else np.nan
    p0=(x==0).mean(); pos=x[x>0]
    return {
      'n':n,'mean_runs':mu,'run_variance':var,'p0_under_pct':100*p0,
      'p1_pct':100*(x==1).mean(),'p2_pct':100*(x==2).mean(),'p3plus_pct':100*(x>=3).mean(),
      'score_pct':100*(1-p0),
      'mean_runs_given_score':pos.mean() if len(pos) else np.nan,
      'multi_run_given_score_pct':100*(pos>=2).mean() if len(pos) else np.nan,
      'threeplus_given_score_pct':100*(pos>=3).mean() if len(pos) else np.nan,
      'runs_from_multi_run_innings_pct':100*x[x>=2].sum()/x.sum() if x.sum()>0 else np.nan,
      'nb_p0_pct':100*nb_p0(mu,var),
      'nb_zero_residual_pp':100*(p0-nb_p0(mu,var)),
    }

df=pd.read_csv(URL,compression='gzip',low_memory=False)
if 'benchmark_matched' in df.columns: df=df[df.benchmark_matched==True].copy()
tot=pd.to_numeric(df.dk_total_open_total,errors='coerce')
rows=[]; seasons=[]
for t in sorted(tot.dropna().unique()):
    b=df[tot==t]
    for inn in range(1,10):
        c=f'inning{inn}_total_runs'
        if c not in b: continue
        s=pd.to_numeric(b[c],errors='coerce').dropna(); x=s.to_numpy(float)
        if len(x)<MIN_N: continue
        st=rowstats(x); rows.append({'pregame_total':t,'inning':inn,'clean_i1_i8':inn<=8,**st})
        if 'season' in b:
            tmp=b.loc[s.index,['season',c]].copy(); tmp[c]=pd.to_numeric(tmp[c],errors='coerce')
            for yr,g in tmp.groupby('season'):
                gx=g[c].dropna().to_numpy(float)
                if len(gx)<30: continue
                seasons.append({'pregame_total':t,'inning':inn,'season':yr,**rowstats(gx)})
r=pd.DataFrame(rows); sy=pd.DataFrame(seasons)
if len(sy):
    stab=sy.groupby(['pregame_total','inning']).agg(
      seasons=('season','nunique'),
      season_p0_sd_pp=('p0_under_pct','std'),
      season_nb_zero_resid_sd_pp=('nb_zero_residual_pp','std'),
      season_mean_runs_given_score_sd=('mean_runs_given_score','std'),
      season_multi_run_given_score_sd_pp=('multi_run_given_score_pct','std'),
      seasons_nb_resid_positive=('nb_zero_residual_pp',lambda z:int((z>0).sum())),
      seasons_nb_resid_negative=('nb_zero_residual_pp',lambda z:int((z<0).sum())),
    ).reset_index()
    r=r.merge(stab,on=['pregame_total','inning'],how='left')
r.to_csv(OUT/'inning_scoring_cluster_shape_2021_2025.csv',index=False)
sy.to_csv(OUT/'inning_scoring_cluster_shape_by_season_2021_2025.csv',index=False)
clean=r[(r.clean_i1_i8)&(r.n>=500)].copy()
resid=clean.sort_values(['nb_zero_residual_pp','season_nb_zero_resid_sd_pp'],ascending=[False,True]).head(20)
cluster=clean.sort_values(['multi_run_given_score_pct','season_multi_run_given_score_sd_pp'],ascending=[False,True]).head(20)
cols=['pregame_total','inning','n','p0_under_pct','mean_runs','mean_runs_given_score','multi_run_given_score_pct','threeplus_given_score_pct','runs_from_multi_run_innings_pct','nb_p0_pct','nb_zero_residual_pp','season_p0_sd_pp','season_nb_zero_resid_sd_pp','season_multi_run_given_score_sd_pp']
md=['# Inning Scoring Cluster-Shape Analysis (2021–2025)','',
'## Purpose','This decomposes U/O 0.5 into two separate baseball processes: how often an inning scores at all, and how many runs arrive once scoring begins. The goal is to identify cells where run production is unusually clustered rather than merely high or low on average. I9 is retained only diagnostically and excluded from clean rankings.','',
'## Largest positive zero-mass residual after negative-binomial overdispersion — I1–I8, N >= 500','',resid[cols].to_markdown(index=False,floatfmt='.4f'),'',
'## Highest multi-run clustering conditional on scoring — I1–I8, N >= 500','',cluster[cols].to_markdown(index=False,floatfmt='.4f'),'',
'## Interpretation','- `mean_runs_given_score` measures severity once the inning breaks scoreless.','- `multi_run_given_score_pct` measures how often a scoring inning becomes 2+ runs.','- `runs_from_multi_run_innings_pct` shows how much of total inning scoring is concentrated in 2+ run innings.','- `nb_zero_residual_pp` asks whether zero-run mass remains unusual even after a negative-binomial model absorbs ordinary overdispersion.','- Season SD fields are reported separately; no composite ranking is used.','',
'## Outputs','- `data/derived/all_inning/inning_scoring_cluster_shape_2021_2025.csv`','- `data/derived/all_inning/inning_scoring_cluster_shape_by_season_2021_2025.csv`']
(AN/'INNING_SCORING_CLUSTER_SHAPE_2021_2025.md').write_text('\n'.join(md),encoding='utf-8')
print('\n'.join(md))

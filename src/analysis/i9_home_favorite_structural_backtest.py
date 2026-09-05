import pandas as pd
import numpy as np
from pathlib import Path
URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True,exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(URL,compression='gzip',low_memory=False)
g=df[(df.benchmark_matched==True)&(pd.to_numeric(df.dk_total_open_total,errors='coerce')==8.0)].copy()
g['date']=pd.to_datetime(g.game_date); g=g.sort_values(['date','retro_game_id'])
g['u']=pd.to_numeric(g.inning3_total_runs,errors='coerce').eq(0).astype(int)
g=g[pd.to_numeric(g.inning3_total_runs,errors='coerce').notna()].copy()
p=g.u.mean(); n=len(g)
# sequential non-overlapping block tests
rows=[]
for bs in [25,50,100,250,500]:
 x=g.u.to_numpy(); rates=np.array([x[i:i+bs].mean() for i in range(0,n,bs) if len(x[i:i+bs])==bs])
 expected=np.sqrt(p*(1-p)/bs); observed=rates.std(ddof=1); ratio=(observed/expected)**2
 rows.append([bs,len(rates),rates.mean(),observed,expected,ratio,rates.min(),rates.max()])
# rolling ranges
roll=[]
for w in [25,50,100,250,500]:
 r=g.u.rolling(w).mean().dropna(); roll.append([w,len(r),r.min(),r.max(),r.std(ddof=1)])
# linear probability time trend, slope per 1000 games and z
x=np.arange(n,dtype=float); y=g.u.to_numpy(dtype=float); slope=np.cov(x,y,ddof=0)[0,1]/np.var(x); intercept=y.mean()-slope*x.mean(); resid=y-(intercept+slope*x); se=np.sqrt((resid@resid)/(n-2)/((x-x.mean())@(x-x.mean()))); z=slope/se
# autocorrelation lags
acs=[]
for lag in [1,2,5,10,25,50]: acs.append([lag,pd.Series(y).autocorr(lag=lag)])
# longest runs
best0=best1=cur0=cur1=0
for v in y:
 if v==0: cur0+=1;cur1=0;best0=max(best0,cur0)
 else: cur1+=1;cur0=0;best1=max(best1,cur1)
blocks=pd.DataFrame(rows,columns=['block_games','blocks','mean_under','observed_sd','binomial_expected_sd','dispersion_ratio','min_rate','max_rate'])
blocks.to_csv(OUT/'i3_total8_game_stability_blocks.csv',index=False)
pd.DataFrame(roll,columns=['window','windows','min_rate','max_rate','rolling_sd']).to_csv(OUT/'i3_total8_game_stability_rolling.csv',index=False)
pd.DataFrame(acs,columns=['lag','autocorrelation']).to_csv(OUT/'i3_total8_game_stability_autocorrelation.csv',index=False)
md=['# I3 Under 0.5 / Pregame Total 8.0 — Game-to-Game Stability','',f'N: {n}',f'Under rate: {100*p:.3f}%',f'Fair odds: {-100*p/(1-p):.1f}',f'Linear time trend: {100*slope*1000:.3f} percentage points per 1,000 games (z={z:.2f})',f'Longest Under streak: {best1}',f'Longest Over streak: {best0}','','## Non-overlapping sequential blocks','','| Games/block | Blocks | Observed SD | Binomial expected SD | Dispersion ratio | Min | Max |','|---:|---:|---:|---:|---:|---:|---:|']
for _,r in blocks.iterrows(): md.append(f"| {int(r.block_games)} | {int(r.blocks)} | {100*r.observed_sd:.2f} pp | {100*r.binomial_expected_sd:.2f} pp | {r.dispersion_ratio:.3f} | {100*r.min_rate:.1f}% | {100*r.max_rate:.1f}% |")
md+=['','## Rolling windows','','| Window | Min | Max | Rolling SD |','|---:|---:|---:|---:|']
for w,c,mn,mx,sd in roll: md.append(f'| {w} | {100*mn:.1f}% | {100*mx:.1f}% | {100*sd:.2f} pp |')
md+=['','## Serial dependence','','| Lag | Autocorrelation |','|---:|---:|']+[f'| {lag} | {ac:.4f} |' for lag,ac in acs]
(AN/'I9_HOME_FAVORITE_STRUCTURAL_BACKTEST_2021_2025.md').write_text('\n'.join(md))
# keep workflow-required legacy csv names intact with the new stability summaries
blocks.to_csv(OUT/'i9_home_favorite_structural_backtest_2021_2025.csv',index=False)
pd.DataFrame(roll,columns=['window','windows','min_rate','max_rate','rolling_sd']).to_csv(OUT/'i9_home_favorite_disjoint_tiers_2021_2025.csv',index=False)
print('\n'.join(md))
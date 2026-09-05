import pandas as pd, numpy as np
from pathlib import Path
URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True,exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(URL,compression='gzip',low_memory=False)
df=df[df['benchmark_matched']==True].copy()
datecol=next((c for c in ['game_date','gameDate','date'] if c in df.columns),None)
if datecol: df['_date']=pd.to_datetime(df[datecol],errors='coerce'); df=df.sort_values('_date',kind='stable')
tot=pd.to_numeric(df['dk_total_open_total'],errors='coerce')
# test every observed half-point/integer total with >=100 valid games, innings 1-9
rows=[]
for total in sorted(tot.dropna().unique()):
 base=df[tot==total]
 for inn in range(1,10):
  col=f'inning{inn}_total_runs'
  if col not in base: continue
  z=pd.to_numeric(base[col],errors='coerce').dropna(); y=(z==0).astype(int).to_numpy(); n=len(y)
  if n<100: continue
  p=y.mean(); x=np.arange(n,dtype=float)
  slope=np.polyfit(x,y,1)[0] if n>1 else np.nan
  ac1=pd.Series(y).autocorr(1); ac5=pd.Series(y).autocorr(5)
  # block overdispersion at multiple scales; weighted score emphasizes 100/250-game practical stability
  od={}; sds={}; ranges={}
  for k in [25,50,100,250,500]:
   m=n//k
   if m>=2:
    a=y[:m*k].reshape(m,k).mean(axis=1); exp=p*(1-p)/k
    od[k]=a.var(ddof=1)/exp if exp else np.nan; sds[k]=100*a.std(ddof=1); ranges[k]=100*(a.max()-a.min())
   else: od[k]=sds[k]=ranges[k]=np.nan
  # season SD retained as independent long-horizon check
  tmp=base.loc[z.index].copy(); tmp['_u']=(pd.to_numeric(tmp[col],errors='coerce')==0).astype(int)
  season_sd=100*tmp.groupby('season')['_u'].mean().std(ddof=1) if 'season' in tmp and tmp['season'].nunique()>1 else np.nan
  # stationarity score: deviations of overdispersion from 1 + serial dependence + scaled drift; lower is better
  ods=[abs(od[k]-1) for k in [50,100,250] if np.isfinite(od[k])]
  score=(np.mean(ods) if ods else np.nan)+5*abs(ac1 if np.isfinite(ac1) else 0)+3*abs(ac5 if np.isfinite(ac5) else 0)+abs(100*slope*1000)/10
  rows.append({'pregame_total':total,'inning':inn,'n':n,'under_pct':100*p,'season_sd_pp':season_sd,'autocorr_lag1':ac1,'autocorr_lag5':ac5,'drift_pp_per_1000':100*slope*1000,'od25':od[25],'od50':od[50],'od100':od[100],'od250':od[250],'od500':od[500],'sd100_pp':sds[100],'range100_pp':ranges[100],'sd250_pp':sds[250],'range250_pp':ranges[250],'stationarity_score':score})
r=pd.DataFrame(rows)
# Primary ranking requires enough observations for >=8 independent 250-game blocks (N>=2000); secondary N>=500.
r['core_n2000']=r.n>=2000; r['usable_n500']=r.n>=500
r=r.sort_values(['core_n2000','stationarity_score','n'],ascending=[False,True,False])
r.to_csv(OUT/'all_inning_total_sequence_stability_2021_2025.csv',index=False)
core=r[r.core_n2000].head(20); usable=r[r.usable_n500].head(30)
cols=['pregame_total','inning','n','under_pct','season_sd_pp','od50','od100','od250','autocorr_lag1','autocorr_lag5','drift_pp_per_1000','sd100_pp','range100_pp','stationarity_score']
md=['# All Inning × Pregame Total Game-Sequence Stability (2021–2025)','', 'Every observed pregame total with >=100 valid games is tested for innings I1–I9. Ranking is based on game-sequence stationarity, not merely season SD. I9 remains structurally censored when bottom nine is skipped.','', '## Primary ranking — N >= 2,000','',core[cols].to_markdown(index=False,floatfmt='.4f'),'','## Broader ranking — N >= 500','',usable[cols].to_markdown(index=False,floatfmt='.4f'),'','## Score definition','Lower is better. Composite uses absolute deviation of 50/100/250-game overdispersion from the Bernoulli expectation of 1.0, lag-1 and lag-5 autocorrelation, and absolute linear drift. Season SD is reported but is not used in the game-sequence score.']
(AN/'ALL_INNING_TOTAL_GAME_SEQUENCE_STABILITY_2021_2025.md').write_text('\n'.join(md),encoding='utf-8')
print('\n'.join(md))
import pandas as pd
import numpy as np
from pathlib import Path

URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True,exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(URL,compression='gzip',low_memory=False)
g=df[(df['benchmark_matched']==True)&(pd.to_numeric(df['dk_total_open_total'],errors='coerce')==8.0)].copy()
# chronological ordering; fall back to original order within date
datecol=next((c for c in ['game_date','gameDate','date'] if c in g.columns),None)
if datecol: g['_date']=pd.to_datetime(g[datecol],errors='coerce'); g=g.sort_values(['_date'],kind='stable')
r=pd.to_numeric(g['inning3_total_runs'],errors='coerce')
g=g[r.notna()].copy(); g['under']=(pd.to_numeric(g['inning3_total_runs'],errors='coerce')==0).astype(int)
y=g['under'].to_numpy(); n=len(y); p=y.mean()

def blocks(k):
 m=n//k
 a=y[:m*k].reshape(m,k).mean(axis=1) if m else np.array([])
 obs=a.var(ddof=1) if len(a)>1 else np.nan
 exp=p*(1-p)/k
 return {'block_size':k,'blocks':m,'mean':a.mean() if len(a) else np.nan,'min':a.min() if len(a) else np.nan,'max':a.max() if len(a) else np.nan,'sd_pp':100*a.std(ddof=1) if len(a)>1 else np.nan,'expected_sd_pp':100*np.sqrt(exp),'overdispersion_ratio':obs/exp if exp and len(a)>1 else np.nan}
bs=pd.DataFrame([blocks(k) for k in [25,50,100,250,500]])
roll=[]
for k in [25,50,100,250,500]:
 s=pd.Series(y).rolling(k).mean().dropna()
 roll.append({'window':k,'windows':len(s),'mean':s.mean(),'min':s.min(),'max':s.max(),'sd_pp':100*s.std(ddof=1)})
rs=pd.DataFrame(roll)
# autocorrelation and linear time drift
acs=[]
for lag in [1,2,5,10,25,50]:
 ac=pd.Series(y).autocorr(lag=lag)
 acs.append({'lag':lag,'autocorr':ac})
ac=pd.DataFrame(acs)
x=np.arange(n,dtype=float); slope=np.polyfit(x,y,1)[0]; drift_pp_1000=100*slope*1000
# streaks
best1=best0=cur1=cur0=0
for v in y:
 if v: cur1+=1; cur0=0; best1=max(best1,cur1)
 else: cur0+=1; cur1=0; best0=max(best0,cur0)
# halves/quarters as simple structural-break diagnostics
segments=[]
for parts in [2,4,8]:
 for i,a in enumerate(np.array_split(y,parts),1): segments.append({'parts':parts,'segment':i,'n':len(a),'under_pct':100*a.mean()})
seg=pd.DataFrame(segments)
bs.to_csv(OUT/'i3_total8_sequence_blocks.csv',index=False); rs.to_csv(OUT/'i3_total8_sequence_rolling.csv',index=False); ac.to_csv(OUT/'i3_total8_sequence_autocorr.csv',index=False); seg.to_csv(OUT/'i3_total8_sequence_segments.csv',index=False)
md=['# I3 Under 0.5 — Total 8.0 Game-Sequence Stability (2021–2025)','',f'Games: **{n}**',f'Overall I3 Under 0.5: **{100*p:.3f}%**',f'Linear drift: **{drift_pp_1000:.3f} percentage points per 1,000 games**',f'Longest Under streak: **{best1}**',f'Longest Over streak: **{best0}**','','## Non-overlapping blocks','',bs.to_markdown(index=False,floatfmt='.4f'),'','## Rolling windows','',rs.to_markdown(index=False,floatfmt='.4f'),'','## Serial correlation','',ac.to_markdown(index=False,floatfmt='.5f'),'','## Sequential segments','',seg.to_markdown(index=False,floatfmt='.3f'),'','## Interpretation rule','Overdispersion near 1.0 means block-to-block variation is consistent with ordinary Bernoulli sampling noise. Materially above 1.0 indicates clustering/regime variation; materially below 1.0 indicates unusually even dispersion. Near-zero serial correlation and small time drift support game-sequence stationarity.']
(AN/'I3_TOTAL8_GAME_SEQUENCE_STABILITY_2021_2025.md').write_text('\n'.join(md),encoding='utf-8')
print('\n'.join(md))
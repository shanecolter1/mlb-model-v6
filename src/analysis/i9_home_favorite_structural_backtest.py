import pandas as pd
import numpy as np
from pathlib import Path

URL='https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT=Path('data/derived/all_inning'); OUT.mkdir(parents=True, exist_ok=True)
AN=Path('analysis'); AN.mkdir(parents=True, exist_ok=True)

df=pd.read_csv(URL, compression='gzip', low_memory=False)
use=df[(df['benchmark_matched']==True) & (df['dk_total_open_total'].isin([7.0,7.5,8.0,8.5,9.0]))].copy()
use=use[pd.to_numeric(use['dk_moneyline_open_homeOdds'], errors='coerce').notna()]
use['home_ml']=pd.to_numeric(use['dk_moneyline_open_homeOdds'], errors='coerce')
use['i9_runs']=pd.to_numeric(use['inning9_total_runs'], errors='coerce')
use['away9']=pd.to_numeric(use['away_inn9'], errors='coerce')
use['home9']=pd.to_numeric(use['home_inn9'], errors='coerce')
# I9 reached if top 9 was played. Blank home9 with reached top9 means bottom 9 not played.
use=use[use['away9'].notna()].copy()
use['bottom9_skipped']=use['home9'].isna()
use['i9_under05']=use['i9_runs'].fillna(use['away9']).eq(0)
use['bottom9_played']=~use['bottom9_skipped']

def fair_american(p):
    if p<=0 or p>=1: return np.nan
    return -100*p/(1-p) if p>=0.5 else 100*(1-p)/p

# nested thresholds show the actual strategy effect as favorite strength increases
thresholds=[-150,-175,-200,-225,-250,-300]
rows=[]
for total in [7.0,7.5,8.0,8.5,9.0,'ALL']:
    base=use if total=='ALL' else use[use['dk_total_open_total']==total]
    for th in thresholds:
        g=base[base['home_ml']<=th]
        if len(g)==0: continue
        played=g[g['bottom9_played']]
        p_under=g['i9_under05'].mean()
        rows.append({
            'pregame_total':total,'home_favorite_threshold':th,'n':len(g),
            'bottom9_skipped_pct':100*g['bottom9_skipped'].mean(),
            'i9_under05_pct':100*p_under,
            'i9_fair_under_american':fair_american(p_under),
            'bottom9_played_n':len(played),
            'under05_if_bottom9_played_pct':100*played['i9_under05'].mean() if len(played) else np.nan,
            'season_sd_under_pp':100*g.groupby('season')['i9_under05'].mean().std(ddof=1) if g['season'].nunique()>1 else np.nan,
        })
res=pd.DataFrame(rows)
res.to_csv(OUT/'i9_home_favorite_structural_backtest_2021_2025.csv',index=False)

# disjoint favorite tiers for easier interpretation
bins=[-10000,-300,-250,-225,-200,-175,-150,0]
labels=['-300 or stronger','-250 to -299','-225 to -249','-200 to -224','-175 to -199','-150 to -174','weaker than -150']
use['fav_tier']=pd.cut(use['home_ml'], bins=bins, labels=labels, right=False)
tiers=[]
for total in [7.0,7.5,8.0,8.5,9.0]:
  for tier,g in use[use['dk_total_open_total']==total].groupby('fav_tier', observed=True):
    if len(g)==0: continue
    p=g['i9_under05'].mean(); played=g[g['bottom9_played']]
    tiers.append({'pregame_total':total,'home_favorite_tier':str(tier),'n':len(g),'bottom9_skipped_pct':100*g['bottom9_skipped'].mean(),'i9_under05_pct':100*p,'i9_fair_under_american':fair_american(p),'under05_if_bottom9_played_pct':100*played['i9_under05'].mean() if len(played) else np.nan})
pd.DataFrame(tiers).to_csv(OUT/'i9_home_favorite_disjoint_tiers_2021_2025.csv',index=False)

core=res[(res['n']>=100)].sort_values(['i9_under05_pct','n'],ascending=[False,False]).head(15)
md=['# I9 Under 0.5 — Home Favorite Structural Backtest (2021–2025)','',
'## Purpose','Test whether strong pregame home favorites create an exploitable structural I9 Under 0.5 signal because the bottom of the ninth is more likely to be skipped.','',
'## Important limitation','The canonical historical master contains DraftKings full-game opening totals and moneylines, but **does not contain historical I9 market prices**. Therefore this report establishes historical hit rates, structural decomposition, and fair prices; it does not claim realized sportsbook ROI. A true ROI test requires archived I9 Under 0.5 prices.','',
'## Method','Sample: matched 2021–2025 games with DraftKings opening totals 7.0–9.0 and a posted opening home moneyline. I9 is counted only when the top of the ninth was reached. A blank home ninth with a reached top ninth is classified as bottom-nine skipped.','',
'## Strongest threshold cells (N >= 100)','', '| Total | Home ML threshold | N | B9 skipped | I9 U0.5 | Fair U | U0.5 if B9 played | Annual SD |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for _,r in core.iterrows():
    md.append(f"| {r['pregame_total']} | {int(r['home_favorite_threshold'])} or stronger | {int(r['n'])} | {r['bottom9_skipped_pct']:.1f}% | {r['i9_under05_pct']:.1f}% | {r['i9_fair_under_american']:.0f} | {r['under05_if_bottom9_played_pct']:.1f}% | {r['season_sd_under_pp']:.2f} pp |")
md += ['', '## Interpretation','The difference between unconditional I9 Under 0.5 and Under 0.5 conditional on the bottom ninth being played quantifies the structural value created by a skipped home half. Increasing Under probability as the home favorite becomes stronger would support using home-favorite strength as an I9-specific market-state variable.','', '## Pricing test status','Historical I9 prices are not present in the 2021–2025 canonical archive. The fair-price column is the break-even American price implied by the observed hit rate; it is the correct benchmark for evaluating any archived or live I9 Under quote.']
(AN/'I9_HOME_FAVORITE_STRUCTURAL_BACKTEST_2021_2025.md').write_text('\n'.join(md),encoding='utf-8')
print(res.to_string(index=False))

#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

from i2_v04_production_validation import load_panel, local_cv, wilson

DEV=[2022,2023,2024]
HOLD=2025

def fair_american(p):
    if not np.isfinite(p) or p<=0 or p>=1: return np.nan
    return -100*p/(1-p) if p>=0.5 else 100*(1-p)/p

def sweep_side(panel,p_under,side):
    y_under=panel.actual_under.to_numpy(int)
    p=np.asarray(p_under,float) if side=='UNDER' else 1-np.asarray(p_under,float)
    y=y_under if side=='UNDER' else 1-y_under
    rows=[]
    seasons=sorted(panel.season.unique())
    for t in np.arange(.40,.751,.0025):
        for split,mask in [
            ('POOLED_2022_2025',np.ones(len(panel),bool)),
            ('DEV_2022_2024',panel.season.isin(DEV).to_numpy()),
            ('HOLDOUT_2025',panel.season.eq(HOLD).to_numpy())]:
            m=mask&(p>=t); n=int(m.sum()); w=int(y[m].sum()) if n else 0
            lo,hi=wilson(w,n); hit=w/n if n else np.nan
            sr=[]
            for yr in seasons:
                q=m&(panel.season.to_numpy()==yr)
                if q.sum(): sr.append(float(y[q].mean()))
            rows.append({
                'side':side,'threshold':round(float(t),4),'split':split,'n':n,'wins':w,'losses':n-w,
                'hit_rate':hit,'realized_fair_american':fair_american(hit) if n else np.nan,
                'mean_model_p':float(p[m].mean()) if n else np.nan,
                'calibration_gap_pp':float((hit-p[m].mean())*100) if n else np.nan,
                'wilson_low':lo,'wilson_high':hi,
                'season_hit_rate_sd_pp':float(np.std(sr,ddof=1)*100) if len(sr)>1 else np.nan,
                'min_season_n':min([int((m&(panel.season.to_numpy()==yr)).sum()) for yr in seasons if (m&(panel.season.to_numpy()==yr)).sum()>0],default=0)
            })
    return pd.DataFrame(rows)

def season_table(panel,p_under,side,threshold):
    y_under=panel.actual_under.to_numpy(int); p=np.asarray(p_under,float) if side=='UNDER' else 1-np.asarray(p_under,float); y=y_under if side=='UNDER' else 1-y_under
    rows=[]
    for yr in sorted(panel.season.unique()):
        m=(panel.season.to_numpy()==yr)&(p>=threshold); n=int(m.sum()); w=int(y[m].sum()) if n else 0; lo,hi=wilson(w,n)
        rows.append({'side':side,'threshold':threshold,'season':int(yr),'n':n,'wins':w,'losses':n-w,'hit_rate':w/n if n else np.nan,'mean_model_p':float(p[m].mean()) if n else np.nan,'wilson_low':lo,'wilson_high':hi})
    return pd.DataFrame(rows)

def choose(df,side):
    x=df[(df.side==side)&(df.split=='DEV_2022_2024')&(df.n>=150)].copy()
    if x.empty: return None
    # Rank price-blind by conservative realized reliability: Wilson lower bound, then lower seasonal SD, then N.
    x['sd_sort']=x.season_hit_rate_sd_pp.fillna(999.)
    x=x.sort_values(['wilson_low','sd_sort','n','threshold'],ascending=[False,True,False,True])
    return x.iloc[0].to_dict()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variants-dir',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',required=True)
    a=ap.parse_args(); meta=json.loads(Path(a.manifest).read_text()); panel=load_panel(Path(a.variants_dir),meta)
    edges,bins,lam,path,p_under,cv=local_cv(panel)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    u=sweep_side(panel,p_under,'UNDER'); o=sweep_side(panel,p_under,'OVER'); full=pd.concat([u,o],ignore_index=True); full.to_csv(out/'full_gate_sweep.csv',index=False)
    picks={}
    for side in ['UNDER','OVER']:
        cand=choose(full,side); picks[side]=cand
        if cand:
            season_table(panel,p_under,side,float(cand['threshold'])).to_csv(out/f'{side.lower()}_selected_gate_by_season.csv',index=False)
    # Also save support points because probability discreteness materially affects threshold interpretation.
    support=pd.DataFrame({'p_under_local_cv':np.round(p_under,8)}).value_counts().reset_index(name='n').sort_values('p_under_local_cv')
    support.to_csv(out/'probability_support.csv',index=False)
    manifest={
        'model':'I2 v0.4 Local-CV Production','price_used':False,'oos_n':int(len(panel)),
        'development_seasons':DEV,'validation_season':HOLD,'selected_lambda':lam,
        'selected_bin_shrinkage':[float(x) for x in path],'fixed_decile_edges':[float(x) for x in edges],
        'under_candidate':picks['UNDER'],'over_candidate':picks['OVER'],
        'selection_rule':'Development N>=150; maximize Wilson lower bound, then minimize season hit-rate SD, then maximize N; price blind.',
        'holdout_note':'2025 was inspected in prior research and is validation, not pristine untouched holdout.'
    }
    for side,cand in picks.items():
        if cand:
            h=full[(full.side==side)&(full.split=='HOLDOUT_2025')&(full.threshold==cand['threshold'])]
            manifest[f'{side.lower()}_holdout_at_candidate']=h.iloc[0].to_dict() if len(h) else None
            p=full[(full.side==side)&(full.split=='POOLED_2022_2025')&(full.threshold==cand['threshold'])]
            manifest[f'{side.lower()}_pooled_at_candidate']=p.iloc[0].to_dict() if len(p) else None
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,default=float)+'\n')
    print(json.dumps(manifest,indent=2,default=float))

if __name__=='__main__': main()

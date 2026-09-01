#!/usr/bin/env python3
"""Build strictly prior-date Statcast batter/pitcher feature candidates for M1.

No shrinkage or production weights are imposed here. 30d/90d/365d/season are
candidate history windows to be selected chronologically downstream. Features
are evaluated at each development PA game date using only pitches from earlier
calendar dates, so doubleheaders/same-day games cannot leak.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS={'30d':30,'90d':90,'365d':365,'season':None}
FAMILIES={
 'fb':{'FF','SI','FC','FA'},
 'breaking':{'SL','CU','KC','ST','SV','CS'},
 'offspeed':{'CH','FS','FO','SC','KN','EP'},
}
SWING={'swinging_strike','swinging_strike_blocked','foul','foul_tip','hit_into_play','foul_bunt','missed_bunt'}
WHIFF={'swinging_strike','swinging_strike_blocked','missed_bunt'}

ADD_COLS=[
 'pitch_n','swing_n','whiff_n','outside_n','chase_n','contact_n','zone_n',
 'fb_n','breaking_n','offspeed_n',
 'fb_velo_sum','fb_velo_n','fb_spin_sum','fb_spin_n','extension_sum','extension_n',
 'bip_n','ev_sum','hardhit_n','barrel_n','la_sum','xwoba_contact_sum','xwoba_contact_n','xba_contact_sum','xba_contact_n',
 'fb_swing_n','fb_whiff_n','fb_bip_n','fb_xwoba_sum','fb_xwoba_n',
 'breaking_swing_n','breaking_whiff_n','breaking_bip_n','breaking_xwoba_sum','breaking_xwoba_n',
 'offspeed_swing_n','offspeed_whiff_n','offspeed_bip_n','offspeed_xwoba_sum','offspeed_xwoba_n'
]

def n(s): return pd.to_numeric(s,errors='coerce')
def ratio(num,den): return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=np.asarray(den)>0)

def primitives(x):
    z=x.copy(); desc=z.description.astype(str); pt=z.pitch_type.astype(str)
    zone=n(z.zone); ev=n(z.launch_speed); la=n(z.launch_angle); lsa=n(z.launch_speed_angle); xw=n(z.estimated_woba_using_speedangle); xba=n(z.estimated_ba_using_speedangle)
    swing=desc.isin(SWING); whiff=desc.isin(WHIFF); inz=zone.between(1,9); bip=ev.notna()
    z['pitch_n']=1.0; z['swing_n']=swing.astype(float); z['whiff_n']=whiff.astype(float); z['outside_n']=(~inz & zone.notna()).astype(float); z['chase_n']=(swing & ~inz & zone.notna()).astype(float); z['contact_n']=(swing & ~whiff).astype(float); z['zone_n']=inz.astype(float)
    for fam,types in FAMILIES.items():
        f=pt.isin(types); z[f'{fam}_n']=f.astype(float)
        z[f'{fam}_swing_n']=(f&swing).astype(float); z[f'{fam}_whiff_n']=(f&whiff).astype(float); z[f'{fam}_bip_n']=(f&bip).astype(float)
        z[f'{fam}_xwoba_sum']=xw.where(f&xw.notna(),0.0).fillna(0.0); z[f'{fam}_xwoba_n']=(f&xw.notna()).astype(float)
    fb=pt.isin(FAMILIES['fb']); velo=n(z.release_speed); spin=n(z.release_spin_rate); ext=n(z.release_extension)
    z['fb_velo_sum']=velo.where(fb&velo.notna(),0.0).fillna(0.0); z['fb_velo_n']=(fb&velo.notna()).astype(float)
    z['fb_spin_sum']=spin.where(fb&spin.notna(),0.0).fillna(0.0); z['fb_spin_n']=(fb&spin.notna()).astype(float)
    z['extension_sum']=ext.fillna(0.0); z['extension_n']=ext.notna().astype(float)
    z['bip_n']=bip.astype(float); z['ev_sum']=ev.fillna(0.0); z['hardhit_n']=(bip&(ev>=95.0)).astype(float)
    z['barrel_n']=(bip&(lsa==6)).astype(float)
    z['la_sum']=la.fillna(0.0); z['xwoba_contact_sum']=xw.fillna(0.0); z['xwoba_contact_n']=xw.notna().astype(float); z['xba_contact_sum']=xba.fillna(0.0); z['xba_contact_n']=xba.notna().astype(float)
    return z

def make_daily(z,idcol):
    q=z[[idcol,'game_date']+ADD_COLS].copy(); q[idcol]=pd.to_numeric(q[idcol],errors='coerce'); q=q.dropna(subset=[idcol,'game_date']); q[idcol]=q[idcol].astype('int64')
    return q.groupby([idcol,'game_date'],as_index=False)[ADD_COLS].sum()

def summarize(S,prefix):
    out={}
    out[prefix+'pitch_count']=S['pitch_n']
    out[prefix+'swing_rate']=ratio(S['swing_n'],S['pitch_n']); out[prefix+'whiff_rate']=ratio(S['whiff_n'],S['swing_n']); out[prefix+'chase_rate']=ratio(S['chase_n'],S['outside_n']); out[prefix+'contact_rate']=ratio(S['contact_n'],S['swing_n']); out[prefix+'zone_rate']=ratio(S['zone_n'],S['pitch_n'])
    out[prefix+'mix_fb']=ratio(S['fb_n'],S['pitch_n']); out[prefix+'mix_breaking']=ratio(S['breaking_n'],S['pitch_n']); out[prefix+'mix_offspeed']=ratio(S['offspeed_n'],S['pitch_n'])
    out[prefix+'fb_velo']=ratio(S['fb_velo_sum'],S['fb_velo_n']); out[prefix+'fb_spin']=ratio(S['fb_spin_sum'],S['fb_spin_n']); out[prefix+'extension']=ratio(S['extension_sum'],S['extension_n'])
    out[prefix+'avg_ev']=ratio(S['ev_sum'],S['bip_n']); out[prefix+'hardhit_rate']=ratio(S['hardhit_n'],S['bip_n']); out[prefix+'barrel_rate']=ratio(S['barrel_n'],S['bip_n']); out[prefix+'avg_launch_angle']=ratio(S['la_sum'],S['bip_n']); out[prefix+'xwoba_contact']=ratio(S['xwoba_contact_sum'],S['xwoba_contact_n']); out[prefix+'xba_contact']=ratio(S['xba_contact_sum'],S['xba_contact_n'])
    for fam in FAMILIES:
        out[prefix+f'{fam}_whiff_rate']=ratio(S[f'{fam}_whiff_n'],S[f'{fam}_swing_n']); out[prefix+f'{fam}_xwoba_contact']=ratio(S[f'{fam}_xwoba_sum'],S[f'{fam}_xwoba_n'])
    return out

def asof_for_targets(daily,targets,idcol,prefix):
    rows=[]
    t=targets.copy(); t[idcol]=pd.to_numeric(t[idcol],errors='coerce'); t=t.dropna(subset=[idcol,'game_date']); t[idcol]=t[idcol].astype('int64')
    dgroups={int(k):g.sort_values('game_date') for k,g in daily.groupby(idcol,sort=False)}
    for ent,tg in t.groupby(idcol,sort=False):
        tg=tg.sort_values('game_date'); g=dgroups.get(int(ent))
        base=pd.DataFrame({idcol:tg[idcol].to_numpy(),'game_date':tg.game_date.to_numpy()})
        if g is None or g.empty:
            rows.append(base); continue
        dates=g.game_date.to_numpy(dtype='datetime64[D]'); vals=g[ADD_COLS].to_numpy(float); cs=np.vstack([np.zeros((1,vals.shape[1])),np.cumsum(vals,axis=0)])
        td=tg.game_date.to_numpy(dtype='datetime64[D]'); hi=np.searchsorted(dates,td,side='left')
        for w,days in WINDOWS.items():
            if days is None:
                starts=pd.to_datetime(tg.game_date).dt.to_period('Y').dt.start_time.to_numpy(dtype='datetime64[D]')
                lo=np.searchsorted(dates,starts,side='left')
            else:
                lo=np.searchsorted(dates,td-np.timedelta64(days,'D'),side='left')
            sums=cs[hi]-cs[lo]
            S={c:sums[:,j] for j,c in enumerate(ADD_COLS)}
            feats=summarize(S,prefix+f'{w}_')
            for c,v in feats.items(): base[c]=v
        rows.append(base)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=[idcol,'game_date'])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--statcast-root',type=Path,required=True); ap.add_argument('--m1-matrix',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    files=sorted(a.statcast_root.rglob('statcast_pitches_20*.parquet')); years=sorted(int(p.stem.split('_')[-1]) for p in files)
    if years!=[2021,2022,2023,2024]: raise RuntimeError(f'Statcast development seasons must be exactly 2021-2024; found {years}')
    parts=[pd.read_parquet(p) for p in files]; x=pd.concat(parts,ignore_index=True); x['game_date']=pd.to_datetime(x.game_date,errors='coerce').dt.normalize(); x=x[x.game_type.astype(str).isin(['R','F','D','L','W'])].copy()
    if (x.game_date.dt.year>=2025).any(): raise RuntimeError('2025 holdout leakage')
    z=primitives(x).rename(columns={'batter':'batter_id','pitcher':'pitcher_id'})
    m=pd.read_parquet(a.m1_matrix); m['game_date']=pd.to_datetime(m.game_date,errors='coerce').dt.normalize(); m['season']=pd.to_numeric(m.season,errors='coerce').astype(int)
    if (m.season>=2025).any(): raise RuntimeError('2025 holdout leakage in M1 targets')
    bt=m[['batter_id','game_date']].drop_duplicates(); pt=m[['pitcher_id','game_date']].drop_duplicates()
    bd=make_daily(z,'batter_id'); pdaily=make_daily(z,'pitcher_id')
    bf=asof_for_targets(bd,bt,'batter_id','batter_'); pf=asof_for_targets(pdaily,pt,'pitcher_id','pitcher_')
    bf.to_parquet(a.output_dir/'statcast_batter_asof.parquet',index=False); pf.to_parquet(a.output_dir/'statcast_pitcher_asof.parquet',index=False)
    manifest={
      'status':'PASS','architecture':'strict_prior_date_statcast_candidate_feature_store','development_seasons':[2021,2022,2023,2024],
      'holdout_season':2025,'holdout_opened':False,'market_data_used':False,'windows':list(WINDOWS),
      'candidate_families':['discipline_contact','pitch_mix','velocity_spin_extension','contact_quality_expected_stats'],
      'hardhit_definition':'Statcast >=95 mph','barrel_definition':'Statcast launch_speed_angle category 6',
      'same_day_history_included':False,'shrinkage_applied':False,'batter_rows':int(len(bf)),'pitcher_rows':int(len(pf)),
      'batter_columns':list(bf.columns),'pitcher_columns':list(pf.columns)
    }
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('_columns')},indent=2))
if __name__=='__main__': main()

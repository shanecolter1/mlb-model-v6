#!/usr/bin/env python3
"""Build canonical 2021-2024 all-inning half-inning research matrix.

Development-only target spine for the unified I1-I9 engine. 2025 target outcomes
are excluded before any inning target is extracted. Existing I2 production code
is not touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

DEV_MAX_SEASON = 2024
INNINGS = range(1, 10)
TEAM_ID_TO_CODE = {
    108:'LAA',109:'ARI',110:'BAL',111:'BOS',112:'CHC',113:'CIN',114:'CLE',115:'COL',
    116:'DET',117:'HOU',118:'KC',119:'LAD',120:'WSH',121:'NYM',133:'OAK',134:'PIT',
    135:'SD',136:'SEA',137:'SF',138:'STL',139:'TB',140:'TEX',141:'TOR',142:'MIN',
    143:'PHI',144:'ATL',145:'CHW',146:'MIA',147:'NYY',158:'MIL'
}
ALIASES = {'ANA':'LAA','CHA':'CHW','CHN':'CHC','LAN':'LAD','NYA':'NYY','NYN':'NYM',
           'SDN':'SD','SFN':'SF','SLN':'STL','TBA':'TB','KCA':'KC','WAS':'WSH'}


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, low_memory=False)


def find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f'{name} expected exactly once under {root}; found {[str(x) for x in hits]}')
    return hits[0]


def norm_code(v):
    if pd.isna(v):
        return None
    s = str(v).upper().strip()
    return ALIASES.get(s, s)


def master_dev(master_path: Path) -> pd.DataFrame:
    m = read(master_path).copy()
    need = ['game_date','away_team_code','home_team_code','game_number','dk_total_open_total']
    miss = [c for c in need if c not in m.columns]
    if miss:
        raise RuntimeError(f'historical master missing {miss}')
    m['game_date'] = pd.to_datetime(m['game_date'], errors='coerce').dt.normalize()
    m['season'] = m['game_date'].dt.year
    # Holdout guard: remove 2025 before extracting or evaluating any inning target.
    m = m[m['season'].between(2021, DEV_MAX_SEASON)].copy()
    m['away_team_code'] = m['away_team_code'].map(norm_code)
    m['home_team_code'] = m['home_team_code'].map(norm_code)
    m['game_number'] = pd.to_numeric(m['game_number'], errors='coerce').astype('Int64')
    m['dk_total_open_total'] = pd.to_numeric(m['dk_total_open_total'], errors='coerce')
    m = m[m['dk_total_open_total'].notna()].copy()
    return m


def add_game_number(gi: pd.DataFrame) -> pd.DataFrame:
    """Recover Retrosheet singleton=0 / DH=1,2 using chronology only."""
    x = gi.copy()
    keys = ['game_date','away_team_code','home_team_code']
    if 'game_number' in x.columns:
        x['game_number'] = pd.to_numeric(x['game_number'], errors='coerce').astype('Int64')
        return x
    group_size = x.groupby(keys, dropna=False)['game_id'].transform('size')
    if 'game_datetime' not in x.columns:
        if (group_size > 1).any():
            raise RuntimeError('same-date matchup duplicates require game_datetime/game_number')
        x['game_number'] = 0
        return x
    x['_sort_dt'] = pd.to_datetime(x['game_datetime'], errors='coerce', utc=True)
    x = x.sort_values(keys + ['_sort_dt','game_id'], kind='mergesort').copy()
    group_size = x.groupby(keys, dropna=False)['game_id'].transform('size')
    seq = x.groupby(keys, dropna=False).cumcount() + 1
    x['game_number'] = np.where(group_size.eq(1), 0, seq).astype(int)
    return x.drop(columns=['_sort_dt'])


def prepare_game_index(game_index: pd.DataFrame) -> pd.DataFrame:
    g = game_index.copy()
    g['game_date'] = pd.to_datetime(g['game_date'], errors='coerce').dt.normalize()
    g['season'] = g['game_date'].dt.year
    g = g[g['season'].between(2021, DEV_MAX_SEASON)].copy()
    if 'away_team_id' not in g.columns or 'home_team_id' not in g.columns:
        raise RuntimeError('game_index missing away_team_id/home_team_id')
    g['away_team_code'] = pd.to_numeric(g['away_team_id'], errors='coerce').map(TEAM_ID_TO_CODE)
    g['home_team_code'] = pd.to_numeric(g['home_team_id'], errors='coerce').map(TEAM_ID_TO_CODE)
    if g['away_team_code'].isna().any() or g['home_team_code'].isna().any():
        bad = g[g['away_team_code'].isna() | g['home_team_code'].isna()][['game_id','away_team_id','home_team_id']].head(20)
        raise RuntimeError(f'unmapped MLB team ids: {bad.to_dict("records")}')
    return add_game_number(g)


def build_half_spine(master: pd.DataFrame, game_index: pd.DataFrame, inning_outcomes: pd.DataFrame) -> pd.DataFrame:
    g = prepare_game_index(game_index)
    join = ['game_date','away_team_code','home_team_code','game_number']
    idx_cols = ['game_id','game_date','game_number','away_team_id','home_team_id','away_team_code','home_team_code','season']
    if 'game_datetime' in g.columns:
        idx_cols.append('game_datetime')
    idx = g[idx_cols].copy()
    if idx.duplicated(join).any():
        raise RuntimeError('game_index non-unique after game-number reconstruction')
    if master.duplicated(join).any():
        raise RuntimeError('historical master non-unique on canonical join keys')
    gm = master[join + ['dk_total_open_total']].merge(idx, on=join, how='inner', validate='one_to_one')
    if len(gm) != len(master):
        miss = master.merge(idx[join], on=join, how='left', indicator=True)
        miss = miss[miss['_merge']=='left_only'][join]
        raise RuntimeError(f'historical/game-index join incomplete: matched {len(gm)} of {len(master)}; first missing={miss.head(20).to_dict("records")}')

    io = inning_outcomes.copy()
    io['inning'] = pd.to_numeric(io['inning'], errors='coerce').astype('Int64')
    io = io[io['inning'].between(1,9) & io['game_id'].isin(gm['game_id'])].copy()
    required = ['game_id','inning','away_runs','home_runs','away_half_played','home_half_played']
    miss = [c for c in required if c not in io.columns]
    if miss:
        raise RuntimeError(f'inning_outcomes missing {miss}')

    base = gm[['game_id','game_date','season','away_team_id','home_team_id','away_team_code','home_team_code','game_number','dk_total_open_total']].copy()
    z = io[required].merge(base, on='game_id', how='inner', validate='many_to_one')
    rows = []
    for r in z.itertuples(index=False):
        common = {
            'game_id': r.game_id,
            'game_date': r.game_date,
            'season': int(r.season),
            'inning': int(r.inning),
            'dk_total_open_total': float(r.dk_total_open_total),
            'away_team_id': r.away_team_id,
            'home_team_id': r.home_team_id,
            'away_team_code': r.away_team_code,
            'home_team_code': r.home_team_code,
            'game_number': int(r.game_number),
        }
        for half, runs_name, played_name, batting, pitching in [
            ('top','away_runs','away_half_played','away_team_id','home_team_id'),
            ('bottom','home_runs','home_half_played','home_team_id','away_team_id')]:
            played = bool(getattr(r, played_name))
            val = pd.to_numeric(pd.Series([getattr(r, runs_name)]), errors='coerce').iloc[0]
            rows.append({**common,
                         'half': half,
                         'batting_team_id': getattr(r, batting),
                         'pitching_team_id': getattr(r, pitching),
                         'half_played': played,
                         'half_runs': float(val) if played and pd.notna(val) else np.nan})

    out = pd.DataFrame(rows)
    out['half_scored'] = np.where(out['half_played'] & out['half_runs'].notna(), (out['half_runs'] >= 1).astype(float), np.nan)
    out['target_class'] = np.where(out['half_played'], 'PLAYED_TARGET', 'UNPLAYED_HALF')
    out['holdout_target_loaded'] = False
    return out.sort_values(['game_date','game_id','inning','half'], kind='stable').reset_index(drop=True)


def audit(out: pd.DataFrame) -> dict:
    dup = int(out.duplicated(['game_id','inning','half']).sum())
    by_inning = {}
    for inn in INNINGS:
        g = out[out['inning']==inn]
        by_inning[str(inn)] = {
            'rows': int(len(g)),
            'played_halves': int(g['half_played'].sum()),
            'unplayed_halves': int((~g['half_played']).sum()),
            'scoring_halves': int(g['half_scored'].fillna(0).sum()),
        }
    b9 = out[(out['inning']==9)&(out['half']=='bottom')]
    return {
        'status': 'PASS' if dup==0 and len(out) else 'FAIL',
        'development_seasons': sorted(int(x) for x in out['season'].dropna().unique()),
        'max_season': int(out['season'].max()) if len(out) else None,
        'rows': int(len(out)),
        'games': int(out['game_id'].nunique()),
        'duplicate_game_inning_half_rows': dup,
        'holdout_target_loaded': False,
        'bottom9': {'rows':int(len(b9)),'played':int(b9['half_played'].sum()),'unplayed':int((~b9['half_played']).sum())},
        'by_inning': by_inning,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master', type=Path, required=True)
    ap.add_argument('--reusable-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    master = master_dev(a.master)
    game_index = read(find_one(a.reusable_root,'game_index.parquet'))
    inning_outcomes = read(find_one(a.reusable_root,'inning_outcomes.parquet'))
    out = build_half_spine(master, game_index, inning_outcomes)
    if len(out)==0:
        raise RuntimeError('all-inning development matrix is empty')
    if out['season'].max() > DEV_MAX_SEASON:
        raise RuntimeError('2025 holdout target leaked into development matrix')
    if out.duplicated(['game_id','inning','half']).any():
        raise RuntimeError('non-unique game x inning x half rows')
    out.to_parquet(a.output, index=False)
    manifest = audit(out)
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

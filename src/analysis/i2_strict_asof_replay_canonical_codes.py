#!/usr/bin/env python3
"""Strict I2 replay wrapper using canonical historical-master team codes.

The canonical joined master uses ARI/OAK/WAS, while the normalized StatsAPI layer
was being normalized to AZ/ATH/WSH. This wrapper aligns StatsAPI identities to the
canonical-master convention and preserves the fixed index-safe fallback join.
"""
import pandas as pd
import i2_strict_asof_replay as replay_mod

CANONICAL = {
    'AZ':'ARI','ARI':'ARI','ARIZONA DIAMONDBACKS':'ARI',
    'ATH':'OAK','OAK':'OAK','OAKLAND ATHLETICS':'OAK','ATHLETICS':'OAK',
    'WSH':'WAS','WSN':'WAS','WAS':'WAS','WASHINGTON NATIONALS':'WAS',
}


def normalize_code_canonical(x):
    s = str(x).upper().strip()
    if s in CANONICAL:
        return CANONICAL[s]
    # Reuse all other existing mappings, then convert any legacy target values.
    v = replay_mod.team_code_map().get(s, s)
    return {'AZ':'ARI','ATH':'OAK','WSH':'WAS'}.get(v, v)


def join_master_to_games_fixed(master, games):
    m = master.copy().reset_index(drop=True)
    if 'game_number' in m.columns:
        m['game_number_join'] = pd.to_numeric(m['game_number'], errors='coerce').fillna(0).astype(int)
    else:
        m = m.sort_values(['game_date','away_team_code','home_team_code']).reset_index(drop=True)
        m['game_number_join'] = m.groupby(['game_date','away_team_code','home_team_code']).cumcount()

    x = m.merge(
        games,
        left_on=['game_date','away_team_code','home_team_code','game_number_join'],
        right_on=['game_date','away_code','home_code','game_number_join'],
        how='left', validate='one_to_one', sort=False,
    ).reset_index(drop=True)

    miss = x['game_id'].isna()
    if miss.any():
        counts = games.groupby(['game_date','away_code','home_code']).size().rename('n').reset_index()
        unique = games.merge(counts[counts['n'] == 1], on=['game_date','away_code','home_code'], how='inner', validate='many_to_one')
        fallback_left = x.loc[miss, ['game_date','away_team_code','home_team_code']].copy()
        fallback_left['_x_index'] = fallback_left.index
        fb = fallback_left.merge(
            unique,
            left_on=['game_date','away_team_code','home_team_code'],
            right_on=['game_date','away_code','home_code'],
            how='left', validate='one_to_one', sort=False,
        ).set_index('_x_index')
        for c in ['game_id','away_team_id','home_team_id']:
            x.loc[fb.index, c] = fb[c]
    return x


replay_mod.normalize_code = normalize_code_canonical
replay_mod.join_master_to_games = join_master_to_games_fixed

if __name__ == '__main__':
    replay_mod.main()

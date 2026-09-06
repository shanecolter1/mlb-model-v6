#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import i2_strict_asof_replay as r


def main():
    master_path = Path(sys.argv[1])
    games_path = Path(sys.argv[2])
    outdir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('data/derived/i2/join_diagnostics')
    outdir.mkdir(parents=True, exist_ok=True)

    m = r.canonical_master(master_path).reset_index(drop=True)
    g = r.build_game_key_from_normalized(games_path).reset_index(drop=True)

    # Raw code coverage.
    master_codes = set(m['away_team_code']).union(set(m['home_team_code']))
    game_codes = set(g['away_code']).union(set(g['home_code']))
    code_report = {
        'master_only_codes': sorted(master_codes - game_codes),
        'games_only_codes': sorted(game_codes - master_codes),
        'master_codes': sorted(master_codes),
        'games_codes': sorted(game_codes),
    }

    # Exact date/team matchup candidate counts, independent of game number.
    matchup_counts = g.groupby(['game_date','away_code','home_code']).size().rename('statsapi_matchup_n').reset_index()
    z = m.merge(matchup_counts,
        left_on=['game_date','away_team_code','home_team_code'],
        right_on=['game_date','away_code','home_code'], how='left')
    z['statsapi_matchup_n'] = z['statsapi_matchup_n'].fillna(0).astype(int)

    # Compare likely game-number conventions.
    if 'game_number' in m.columns:
        raw_num = pd.to_numeric(m['game_number'], errors='coerce')
    else:
        raw_num = pd.Series([float('nan')]*len(m))
    m['_rowid'] = range(len(m))
    tests = {}
    candidates = {
        'raw_fill0': raw_num.fillna(0).astype(int),
        'raw_minus1_floor0': (raw_num.fillna(1).astype(int)-1).clip(lower=0),
        'raw_plus1': raw_num.fillna(0).astype(int)+1,
    }
    # Canonical occurrence ordinal by same date/teams.
    occ = m.sort_values(['game_date','away_team_code','home_team_code','_rowid']).groupby(
        ['game_date','away_team_code','home_team_code']).cumcount()
    candidates['canonical_occurrence_0based'] = occ.reindex(m.index).fillna(0).astype(int)

    detail = m.copy()
    for name, nums in candidates.items():
        left = m.copy()
        left['_join_num'] = nums.to_numpy()
        gg = g.rename(columns={'game_number_join':'_join_num'})
        x = left.merge(gg,
            left_on=['game_date','away_team_code','home_team_code','_join_num'],
            right_on=['game_date','away_code','home_code','_join_num'],
            how='left', sort=False)
        matched = x['game_id'].notna()
        tests[name] = {'matched': int(matched.sum()), 'total': int(len(x)), 'rate': float(matched.mean())}
        detail[f'match_{name}'] = matched.to_numpy()

    # Unique-matchup fallback ceiling using date+teams only.
    unique_keys = g.groupby(['game_date','away_code','home_code']).filter(lambda q: len(q)==1)
    u = m.merge(unique_keys[['game_date','away_code','home_code','game_id']],
        left_on=['game_date','away_team_code','home_team_code'],
        right_on=['game_date','away_code','home_code'], how='left')
    tests['unique_date_team_fallback'] = {
        'matched': int(u['game_id'].notna().sum()), 'total': int(len(u)), 'rate': float(u['game_id'].notna().mean())
    }

    # Any exact date/team candidate at all.
    any_match = z['statsapi_matchup_n'] > 0
    tests['any_exact_date_team_candidate'] = {
        'matched': int(any_match.sum()), 'total': int(len(z)), 'rate': float(any_match.mean())
    }

    # Unmatched diagnostics for zero candidate date/team match.
    no_candidate = z[z['statsapi_matchup_n']==0].copy()
    by_season = no_candidate.groupby('season').size().rename('unmatched').reset_index()
    by_pair = no_candidate.groupby(['away_team_code','home_team_code']).size().rename('unmatched').reset_index().sort_values('unmatched', ascending=False)

    # Check date +/-1 with same teams to identify timezone/date drift.
    gkeys = g[['game_date','away_code','home_code']].drop_duplicates()
    shifts = {}
    for d in [-1,1]:
        left = m[['game_date','away_team_code','home_team_code']].copy()
        left['probe_date'] = left['game_date'] + pd.to_timedelta(d, unit='D')
        q = left.merge(gkeys,
            left_on=['probe_date','away_team_code','home_team_code'],
            right_on=['game_date','away_code','home_code'], how='left')
        shifts[str(d)] = int(q['away_code'].notna().sum())

    summary = {
        'master_rows': int(len(m)),
        'statsapi_rows': int(len(g)),
        'game_number_value_counts': {str(k): int(v) for k,v in raw_num.value_counts(dropna=False).to_dict().items()},
        'code_report': code_report,
        'join_tests': tests,
        'zero_exact_date_team_candidates': int(len(no_candidate)),
        'date_shift_same_team_matches': shifts,
    }
    (outdir/'summary.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    by_season.to_csv(outdir/'unmatched_by_season.csv', index=False)
    by_pair.head(100).to_csv(outdir/'top_unmatched_team_pairs.csv', index=False)
    no_candidate.head(500).to_csv(outdir/'unmatched_examples.csv', index=False)
    detail.to_csv(outdir/'game_number_match_matrix.csv.gz', index=False, compression='gzip')
    print(json.dumps(summary, indent=2, default=str))

if __name__ == '__main__':
    main()

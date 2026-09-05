import math
from pathlib import Path
import numpy as np
import pandas as pd

URL = 'https://github.com/shanecolter1/mlb-model-v6/releases/download/historical-mlb-2021-2025-v1/MLB_Game_Stats_Joined_2021_2025.csv.gz'
OUT = Path('data/derived/all_inning'); OUT.mkdir(parents=True, exist_ok=True)
AN = Path('analysis'); AN.mkdir(parents=True, exist_ok=True)

MIN_N = 300
BOOT = 2000
RNG = np.random.default_rng(20260905)

def fair_american(p):
    if not np.isfinite(p) or p <= 0 or p >= 1:
        return np.nan
    return -100*p/(1-p) if p >= .5 else 100*(1-p)/p

def bootstrap_p0_ci(y):
    n = len(y)
    if n < 2:
        return (np.nan, np.nan)
    # bootstrap Bernoulli zero indicator, vectorized in chunks to keep memory bounded
    z = (y == 0).astype(float)
    vals = []
    chunk = 250
    for _ in range((BOOT + chunk - 1)//chunk):
        b = min(chunk, BOOT - len(vals))
        if b <= 0: break
        idx = RNG.integers(0, n, size=(b, n))
        vals.extend(z[idx].mean(axis=1).tolist())
    return tuple(np.quantile(vals, [.025, .975]))

def nb_p0(mu, var):
    # Negative-binomial moment fit. If not overdispersed, Poisson is the natural limit.
    if not np.isfinite(mu) or mu < 0:
        return np.nan
    if mu == 0:
        return 1.0
    if not np.isfinite(var) or var <= mu:
        return math.exp(-mu)
    r = mu*mu/(var-mu)
    return (r/(r+mu))**r

# Canonical historical master
df = pd.read_csv(URL, compression='gzip', low_memory=False)
if 'benchmark_matched' in df.columns:
    df = df[df['benchmark_matched'] == True].copy()

# Restrict to standard half-run/integer pregame total grid and observed innings.
total = pd.to_numeric(df['dk_total_open_total'], errors='coerce')
rows = []
season_rows = []

for t in sorted(total.dropna().unique()):
    base = df[total == t]
    for inn in range(1, 10):
        col = f'inning{inn}_total_runs'
        if col not in base.columns:
            continue
        x = pd.to_numeric(base[col], errors='coerce').dropna().to_numpy(dtype=float)
        n = len(x)
        if n < MIN_N:
            continue
        mu = x.mean(); var = x.var(ddof=1) if n > 1 else np.nan
        p0 = (x == 0).mean(); p1 = (x == 1).mean(); p2 = (x == 2).mean(); p3p = (x >= 3).mean()
        pois0 = math.exp(-mu)
        nb0 = nb_p0(mu, var)
        ci_lo, ci_hi = bootstrap_p0_ci(x)
        # Residuals in percentage points. Positive => more zero-run innings than mean-run model implies.
        pois_resid = 100*(p0-pois0)
        nb_resid = 100*(p0-nb0)
        rows.append({
            'pregame_total': t, 'inning': inn, 'n': n,
            'mean_runs': mu, 'run_variance': var, 'variance_to_mean': var/mu if mu > 0 else np.nan,
            'p0_under_pct': 100*p0, 'p1_pct': 100*p1, 'p2_pct': 100*p2, 'p3plus_pct': 100*p3p,
            'fair_under_american': fair_american(p0),
            'poisson_p0_pct': 100*pois0, 'poisson_zero_residual_pp': pois_resid,
            'nb_p0_pct': 100*nb0, 'nb_zero_residual_pp': nb_resid,
            'p0_ci95_lo_pct': 100*ci_lo, 'p0_ci95_hi_pct': 100*ci_hi,
            'clean_i1_i8': inn <= 8,
        })

        if 'season' in base.columns:
            tmp = base.loc[pd.to_numeric(base[col], errors='coerce').dropna().index, ['season', col]].copy()
            tmp[col] = pd.to_numeric(tmp[col], errors='coerce')
            for s, g in tmp.groupby('season'):
                if len(g) < 30: continue
                gx = g[col].to_numpy(dtype=float)
                gmu = gx.mean(); gp0 = (gx == 0).mean(); gvar = gx.var(ddof=1)
                season_rows.append({
                    'pregame_total': t, 'inning': inn, 'season': s, 'n': len(gx),
                    'mean_runs': gmu, 'p0_under_pct': 100*gp0,
                    'poisson_zero_residual_pp': 100*(gp0-math.exp(-gmu)),
                    'nb_zero_residual_pp': 100*(gp0-nb_p0(gmu,gvar)),
                })

r = pd.DataFrame(rows)
s = pd.DataFrame(season_rows)

if not s.empty:
    stab = s.groupby(['pregame_total','inning']).agg(
        seasons=('season','nunique'),
        season_p0_sd_pp=('p0_under_pct','std'),
        season_poisson_resid_mean_pp=('poisson_zero_residual_pp','mean'),
        season_poisson_resid_sd_pp=('poisson_zero_residual_pp','std'),
        season_nb_resid_mean_pp=('nb_zero_residual_pp','mean'),
        season_nb_resid_sd_pp=('nb_zero_residual_pp','std'),
        seasons_poisson_resid_positive=('poisson_zero_residual_pp', lambda q: int((q>0).sum())),
        seasons_poisson_resid_negative=('poisson_zero_residual_pp', lambda q: int((q<0).sum())),
    ).reset_index()
    r = r.merge(stab, on=['pregame_total','inning'], how='left')

# Persistent structural zero-mass score: magnitude of residual penalized by year-to-year residual volatility.
# This is descriptive/ranking only; betting decisions must use actual sportsbook price and walk-forward validation.
r['persistent_zero_score'] = np.where(
    r.get('season_poisson_resid_sd_pp', np.nan).notna(),
    r['poisson_zero_residual_pp'].abs() / (1 + r['season_poisson_resid_sd_pp']),
    np.nan,
)

r.to_csv(OUT/'inning_zero_mass_residuals_2021_2025.csv', index=False)
s.to_csv(OUT/'inning_zero_mass_residuals_by_season_2021_2025.csv', index=False)

clean = r[(r.clean_i1_i8) & (r.n >= 500)].copy()
zi = clean.sort_values(['persistent_zero_score','n'], ascending=[False,False]).head(20)
zd = clean.assign(abs_resid=clean.poisson_zero_residual_pp.abs()).sort_values(['persistent_zero_score','n'], ascending=[False,False]).head(20)

cols = ['pregame_total','inning','n','mean_runs','p0_under_pct','fair_under_american','poisson_p0_pct','poisson_zero_residual_pp','nb_p0_pct','nb_zero_residual_pp','season_p0_sd_pp','season_poisson_resid_sd_pp','persistent_zero_score']
md = [
'# Inning U/O 0.5 Zero-Mass Residual Analysis (2021–2025)',
'',
'## Question',
'Does an inning have a systematically different probability of scoring **zero runs** than a sportsbook-style mean-runs model would imply? This targets distribution shape, not merely expected runs.',
'',
'For each pregame DraftKings game-total × inning cell, the analysis compares the empirical P(0 runs) with (1) a Poisson model using the cell\'s observed mean runs and (2) a moment-fit negative-binomial model that allows overdispersion. Positive residual = more scoreless innings than the mean-run model implies; negative = fewer.',
'',
'**I9 is retained for diagnostics but excluded from the clean target ranking because bottom-nine censoring creates a structural settlement/game-state effect.**',
'',
'## Highest persistent zero-mass departures — clean I1–I8, N >= 500',
'',
zi[cols].to_markdown(index=False, floatfmt='.4f'),
'',
'## Interpretation',
'- `poisson_zero_residual_pp` is the primary shape diagnostic.',
'- `nb_zero_residual_pp` asks whether the effect survives after allowing ordinary run-count overdispersion.',
'- `season_poisson_resid_sd_pp` tests whether the residual itself is stable across seasons.',
'- `persistent_zero_score` is only a discovery ranking: |Poisson residual| / (1 + season residual SD). It is **not** a betting score.',
'- Any candidate must next survive strict walk-forward validation and comparison with actual DraftKings/FanDuel inning prices.',
'',
'## Outputs',
'- `data/derived/all_inning/inning_zero_mass_residuals_2021_2025.csv`',
'- `data/derived/all_inning/inning_zero_mass_residuals_by_season_2021_2025.csv`',
]
(AN/'INNING_ZERO_MASS_RESIDUAL_ANALYSIS_2021_2025.md').write_text('\n'.join(md), encoding='utf-8')
print('\n'.join(md))

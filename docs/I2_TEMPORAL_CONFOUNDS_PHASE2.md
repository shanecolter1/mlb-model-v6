# I2 temporal confounds: point-in-time input build

The existing I2 replay ZIPs contain pregame predictions and observed I2
outcomes. They cannot by themselves distinguish a recurring seasonal pattern
from a 2026-specific ball or league environment shift. The existing Statcast
research dataset contains inning-two terminal plate appearances only.

This phase builds one missing candidate input from MLB Stats API final feeds:
every recognized regular-season terminal plate appearance in **all innings**. For each
regular-season game date, `build_i2_league_asof_features.py` computes league
event counts and rates over the preceding 14, 28 and 56 calendar days. The
rates include hits, home runs, walks, strikeouts, outs on balls in play, and
home runs per ball in play. All same-day games are excluded, including games
that happened earlier that day. The output is a dated research covariate,
not a calibration adjustment or a production forecast.

The GitHub Actions workflow `i2_temporal_asof_features.yml` built annual
2022–2026 CSV artifacts independently in
[run 36545276521](https://github.com/shanecolter1/mlb-model-v6/actions/runs/36545276521).
The compact derived CSVs and checksummed `FEATURE_MANIFEST.json` are stored
in `data/derived/i2_vnext/temporal/`. Coverage is 2,430 games in 2022,
2,430 in 2023, 2,429 in 2024, 2,430 in 2025, and 2,429 in 2026. Duplicate
schedule records are counted once; records with conflicting game dates and
final feeds without plays are excluded. The 2022 series is retained to
diagnose the pre-2023 rule environment but must not be pooled as an
interchangeable present-day season. Each job checks the complete regular
season game universe and the prior-day cutoff. The raw feeds are ephemeral.

The pregame trailing 28-day home-run rate per ball in play was:

| Date | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: |
| May 24 | 4.61% | 4.17% | 4.32% | 4.00% |
| July 15 | 4.70% | 4.83% | 4.78% | 5.01% |
| August 15 | 5.06% | 4.87% | 4.74% | 4.31% |
| September 15 | 5.06% | 4.38% | 4.82% | 4.39% |

The within-2026 increase through July is visible in all-innings outcomes,
but that rate fell again by August while the raw I2 model's largest Over
residuals arrived in late August and September. Ball-flight changes may be
one contributor, but the home-run event rate by itself does not explain the
later I2 pattern. These rows are trailing windows with substantial overlap,
not independent game samples or a fitted ball effect.

**Limitations:** Stats API event outcomes are not independent physical
measurements of the baseball. They can reflect changes in personnel, weather,
parks, and schedule. Official [Savant drag/carry measurements](https://baseballsavant.mlb.com/drag-dashboard) must be joined
as a separate as-of series before describing a 2026 ball-specific component.
Team playoff context also needs its own pregame historical series. None of
those effects is estimated or promoted here. The next comparison must join
these event features to frozen forecasts by game date, fit jointly on prior
seasons, and measure full-I2 log loss and Brier in chronological validation.

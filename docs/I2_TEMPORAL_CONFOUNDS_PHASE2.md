# I2 temporal confounds: point-in-time input build

The existing I2 replay ZIPs contain pregame predictions and observed I2
outcomes. They cannot by themselves distinguish a recurring seasonal pattern
from a 2026-specific ball or league environment shift. The existing Statcast
research dataset contains inning-two terminal plate appearances only.

This phase builds one missing candidate input from MLB Stats API final feeds:
every regular-season terminal plate appearance in **all innings**. For each
regular-season game date, `build_i2_league_asof_features.py` computes league
event counts and rates over the preceding 14, 28 and 56 calendar days. The
rates include hits, home runs, walks, strikeouts, outs on balls in play, and
home runs per ball in play. All same-day games are excluded, including games
that happened earlier that day. The output is a dated research covariate,
not a calibration adjustment or a production forecast.

The GitHub Actions workflow `i2_temporal_asof_features.yml` builds annual
2022–2026 CSV artifacts independently. The 2022 series is retained to
diagnose the pre-2023 rule environment but must not be pooled as an
interchangeable present-day season. Each job checks the complete regular
season game universe and the prior-day cutoff. The raw feeds are ephemeral;
the compact daily feature CSV is the retained artifact for the next phase.

**Limitations:** Stats API event outcomes are not independent physical
measurements of the baseball. They can reflect changes in personnel, weather,
parks, and schedule. Official Savant drag/carry measurements must be joined
as a separate as-of series before describing a 2026 ball-specific component.
Team playoff context also needs its own pregame historical series. None of
those effects is estimated or promoted here. The next comparison must join
these event features to frozen forecasts by game date, fit jointly on prior
seasons, and measure full-I2 log loss and Brier in chronological validation.

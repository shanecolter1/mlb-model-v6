# I2 temporal confounds: pregame context inputs

Research checkpoint, 2026-09-29. No live forecast or betting threshold changed.

## Sources and time boundaries

* MLB Stats API regular-season standings were requested for the calendar day preceding each game date. The archived team rows contain wins, losses, division and wild-card games back, ranks and clinch fields. There are 178–184 complete 30-team dates per season in 2022–26; opening dates lacking a complete response are left missing. Historical spot checks show the Yankees' wins change from early to late season, confirming the date parameter does not silently return final standings.
* Baseball Savant's public drag dashboard embeds daily mean drag coefficient (`mean_cd`) for four-seam fastballs, with pitch and game counts. The snapshot has 177–178 observed days per full 2022–26 season. A rolling 14/28/56-day mean is weighted by the number of pitches and ends **two calendar days before** the game. It includes all available four-seam fastballs, not just balls put in play or second-inning events.
* The dashboard is a **retrospective snapshot** fetched on 2026-09-29. We cannot verify that its historical values, environmental correction, or coverage were published at the two-day lag during each season. This input is suitable for descriptive sensitivity analysis but cannot by itself establish a clean point-in-time walk-forward result. The manifest records the source HTML SHA and this limitation.
* The earlier all-innings, all-PA event-rate series is prior-day only. Standings, drag and event rates must be joined by exact game date, with missing values explicit. All are independent of betting prices.

## Initial physical-series inspection

Two-day-lagged, pitch-weighted 28-day drag coefficient on comparable game dates:

| Season | May 24 | Aug 15 | Sep 15 |
|---|---:|---:|---:|
| 2022 | .3477 | .3474 | .3474 |
| 2023 | .3410 | .3402 | .3439 |
| 2024 | .3443 | .3428 | .3457 |
| 2025 | .3516 | .3505 | .3517 |
| 2026 | .3468 | .3421 | .3471 |

The 2026 physical series is lower in mid-August than late May, then rebounds by mid-September. Lower drag is directionally consistent with greater carry, but this pattern alone cannot attribute I2 scoring changes to the baseball. Temperature, weather, roster, pitching, schedule and scoring environment remain entangled. Cross-year levels also differ substantially.

## Joint forecast structure and gate

The next fitted candidate uses the archived market-isolated top and bottom scoring probabilities as offsets, with shared rule-era and league-environment effects and half-specific late-season/team-context terms. It must include **simultaneous** season position, prior league events, drag and team standings context in one regularized fit. It should recombine the two half probabilities for the full I2 Under, rather than directly fitting a full-inning correction that conflicts with the halves. The 2022 rule era needs its own effect; data from 2023 onward support the post-rule era. Fit and tuning must be walk-forward by season and date, compared against the unmodified forecast and a constant-calibration baseline on log loss, Brier and date-cluster uncertainty. Stratify diagnostics by early, middle and late season and by 2026 spring/summer/fall. No term is attributed causally merely because it improves a backtest.

The 2026 outcome pattern was already inspected. Therefore any candidate chosen with knowledge of it is a **descriptive replication**, not an untouched holdout. Production promotion needs a later genuinely prospective evaluation and a timestamped ball metric source.

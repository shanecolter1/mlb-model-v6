# Inning U/O 0.5 Zero-Mass Residual Analysis (2021–2025)

## Question
Does an inning have a systematically different probability of scoring **zero runs** than a sportsbook-style mean-runs model would imply? This targets distribution shape, not merely expected runs.

For each pregame DraftKings game-total × inning cell, the analysis compares the empirical P(0 runs) with (1) a Poisson model using the cell's observed mean runs and (2) a moment-fit negative-binomial model that allows overdispersion. Positive residual = more scoreless innings than the mean-run model implies; negative = fewer.

**I9 is retained for diagnostics but excluded from the clean target ranking because bottom-nine censoring creates a structural settlement/game-state effect.**

## Highest persistent zero-mass departures — clean I1–I8, N >= 500

|   pregame_total |   inning |         n |   mean_runs |   p0_under_pct |   fair_under_american |   poisson_p0_pct |   poisson_zero_residual_pp |   nb_p0_pct |   nb_zero_residual_pp |   season_p0_sd_pp |   season_poisson_resid_sd_pp |   persistent_zero_score |
|----------------:|---------:|----------:|------------:|---------------:|----------------------:|-----------------:|---------------------------:|------------:|----------------------:|------------------:|-----------------------------:|------------------------:|
|          9.0000 |   3.0000 | 1839.0000 |      1.0919 |        50.2447 |             -100.9836 |          33.5579 |                    16.6868 |     48.2842 |                1.9605 |            2.5665 |                       0.6626 |                 10.0367 |
|          8.0000 |   1.0000 | 2146.0000 |      0.9730 |        53.1221 |             -113.3201 |          37.7958 |                    15.3263 |     51.7461 |                1.3759 |            3.7263 |                       0.7507 |                  8.7546 |
|          8.5000 |   4.0000 | 2871.0000 |      0.9958 |        52.3859 |             -110.0219 |          36.9420 |                    15.4439 |     50.5191 |                1.8668 |            1.7328 |                       1.0195 |                  7.6473 |
|          7.5000 |   4.0000 | 1612.0000 |      0.9181 |        55.9553 |             -127.0423 |          39.9271 |                    16.0282 |     53.9492 |                2.0061 |            2.5115 |                       1.1915 |                  7.3137 |
|          8.5000 |   6.0000 | 2869.0000 |      1.0401 |        52.2133 |             -109.2633 |          35.3425 |                    16.8708 |     49.9843 |                2.2290 |            1.9367 |                       1.3785 |                  7.0930 |
|          8.5000 |   5.0000 | 2871.0000 |      1.0362 |        52.1073 |             -108.8000 |          35.4792 |                    16.6281 |     50.1959 |                1.9114 |            3.6913 |                       1.3918 |                  6.9522 |
|          9.0000 |   7.0000 | 1836.0000 |      1.0822 |        51.6885 |             -106.9899 |          33.8834 |                    17.8050 |     49.4316 |                2.2569 |            3.0679 |                       1.5908 |                  6.8724 |
|          8.5000 |   3.0000 | 2871.0000 |      1.0794 |        51.4803 |             -106.1019 |          33.9794 |                    17.5009 |     49.4052 |                2.0751 |            2.2690 |                       1.6376 |                  6.6351 |
|          8.5000 |   2.0000 | 2871.0000 |      0.8966 |        56.1477 |             -128.0381 |          40.7974 |                    15.3503 |     54.4533 |                1.6944 |            2.5376 |                       1.3669 |                  6.4855 |
|          8.0000 |   8.0000 | 2144.0000 |      0.9883 |        54.2444 |             -118.5525 |          37.2194 |                    17.0250 |     52.1752 |                2.0692 |            1.5235 |                       1.6425 |                  6.4429 |
|          7.5000 |   8.0000 | 1609.0000 |      0.8987 |        56.8676 |             -131.8444 |          40.7101 |                    16.1576 |     55.7470 |                1.1207 |            3.8005 |                       1.6109 |                  6.1886 |
|          9.5000 |   3.0000 |  875.0000 |      1.0789 |        51.5429 |             -106.3679 |          33.9984 |                    17.5445 |     48.6136 |                2.9293 |            5.7012 |                       1.9072 |                  6.0349 |
|          8.0000 |   7.0000 | 2144.0000 |      0.9128 |        56.8097 |             -131.5335 |          40.1407 |                    16.6690 |     54.4101 |                2.3996 |            1.7168 |                       1.8613 |                  5.8257 |
|          9.0000 |   1.0000 | 1839.0000 |      1.1343 |        48.2327 |              107.3281 |          32.1643 |                    16.0684 |     45.8786 |                2.3541 |            2.8961 |                       1.7692 |                  5.8025 |
|          8.5000 |   1.0000 | 2871.0000 |      1.0498 |        51.2365 |             -105.0714 |          35.0005 |                    16.2360 |     48.4198 |                2.8167 |            1.3332 |                       1.8026 |                  5.7931 |
|          8.0000 |   4.0000 | 2146.0000 |      0.9613 |        54.0075 |             -117.4265 |          38.2387 |                    15.7688 |     52.1359 |                1.8716 |            1.9195 |                       1.7888 |                  5.6543 |
|          9.5000 |   6.0000 |  874.0000 |      1.1224 |        47.3684 |              111.1111 |          32.5489 |                    14.8195 |     45.3773 |                1.9911 |            5.4429 |                       1.8304 |                  5.2358 |
|          8.0000 |   6.0000 | 2145.0000 |      0.9506 |        54.0793 |             -117.7665 |          38.6516 |                    15.4277 |     52.1419 |                1.9374 |            3.2169 |                       1.9600 |                  5.2120 |
|          8.5000 |   7.0000 | 2867.0000 |      0.9299 |        55.5633 |             -125.0392 |          39.4596 |                    16.1037 |     53.2609 |                2.3024 |            1.5141 |                       2.0994 |                  5.1958 |
|          7.0000 |   6.0000 |  655.0000 |      0.8427 |        57.2519 |             -133.9286 |          43.0526 |                    14.1993 |     56.6320 |                0.6200 |            3.7661 |                       1.7683 |                  5.1292 |

## Interpretation
- `poisson_zero_residual_pp` is the primary shape diagnostic.
- `nb_zero_residual_pp` asks whether the effect survives after allowing ordinary run-count overdispersion.
- `season_poisson_resid_sd_pp` tests whether the residual itself is stable across seasons.
- `persistent_zero_score` is only a discovery ranking: |Poisson residual| / (1 + season residual SD). It is **not** a betting score.
- Any candidate must next survive strict walk-forward validation and comparison with actual DraftKings/FanDuel inning prices.

## Outputs
- `data/derived/all_inning/inning_zero_mass_residuals_2021_2025.csv`
- `data/derived/all_inning/inning_zero_mass_residuals_by_season_2021_2025.csv`
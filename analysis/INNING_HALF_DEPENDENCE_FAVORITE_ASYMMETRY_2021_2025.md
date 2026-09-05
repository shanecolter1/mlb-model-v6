# Inning Half-Dependence & Favorite-Asymmetry Analysis (2021–2025)

## Question
Does full-inning U0.5 contain structure that a mean-run model misses because (a) the two half-innings are dependent or (b) the same full-game total hides materially different favorite/underdog run-allocation shapes?

Full-inning zero probability is decomposed as observed P(top=0,bottom=0) versus the independence benchmark P(top=0)×P(bottom=0). Positive `joint_zero_dependence_pp` means scoreless halves cluster within the same games. Favorite strength is the no-vig probability of the stronger side from DraftKings opening moneylines.

**I1–I8 are clean. I9 remains diagnostic only. No inning-market prices are inputs.**

## Priority target cells

|   pregame_total |   inning |         n |   under_pct |   top_zero_pct |   bottom_zero_pct |   independence_p0_pct |   joint_zero_dependence_pp |   top_bottom_scoring_phi |   mean_runs |   multi_run_given_score_pct |   season_under_sd_pp |
|----------------:|---------:|----------:|------------:|---------------:|------------------:|----------------------:|---------------------------:|-------------------------:|------------:|----------------------------:|---------------------:|
|          8.0000 |   3.0000 | 2146.0000 |     54.1473 |        73.5788 |           73.5322 |               54.1040 |                     0.0432 |                   0.0022 |      0.9520 |                     53.7602 |               0.8604 |
|          8.5000 |   3.0000 | 2871.0000 |     51.4803 |        73.8070 |           69.6621 |               51.4156 |                     0.0648 |                   0.0032 |      1.0794 |                     58.3632 |               2.2690 |
|          8.5000 |   6.0000 | 2869.0000 |     52.2133 |        72.8825 |           71.5232 |               52.1279 |                     0.0854 |                   0.0043 |      1.0401 |                     56.7469 |               1.9367 |
|          8.5000 |   8.0000 | 2862.0000 |     55.5206 |        75.6115 |           73.8994 |               55.8764 |                    -0.3558 |                  -0.0189 |      0.9479 |                     56.4022 |               1.9683 |
|          9.0000 |   6.0000 | 1837.0000 |     52.1502 |        71.9107 |           71.2030 |               51.2026 |                     0.9476 |                   0.0466 |      1.0289 |                     56.4278 |               2.9005 |

## Favorite-strength decomposition of priority cells

|   pregame_total |   inning | favorite_strength_bucket   |    n |   mean_favorite_novig_prob |   under_pct |   joint_zero_dependence_pp |   multi_run_given_score_pct |   season_under_sd_pp |
|----------------:|---------:|:---------------------------|-----:|---------------------------:|------------:|---------------------------:|----------------------------:|---------------------:|
|          8.0000 |        3 | 50-55%                     |  792 |                     0.5228 |     52.9040 |                     0.2956 |                     53.8874 |               3.0796 |
|          8.0000 |        3 | 55-60%                     |  694 |                     0.5726 |     55.1873 |                     0.0191 |                     55.9486 |               2.1799 |
|          8.0000 |        3 | 60-65%                     |  401 |                     0.6233 |     55.3616 |                     0.3657 |                     53.6313 |               7.7488 |
|          8.0000 |        3 | 65-70%                     |  186 |                     0.6716 |     53.7634 |                    -1.2574 |                     50.0000 |               7.8725 |
|          8.5000 |        3 | 50-55%                     | 1052 |                     0.5224 |     52.0913 |                    -0.7684 |                     57.3413 |               2.4411 |
|          8.5000 |        3 | 55-60%                     |  915 |                     0.5744 |     50.7104 |                     0.3344 |                     61.4191 |               2.1413 |
|          8.5000 |        3 | 60-65%                     |  541 |                     0.6231 |     53.2348 |                     1.4719 |                     58.4980 |               5.9827 |
|          8.5000 |        3 | 65-70%                     |  259 |                     0.6706 |     49.8069 |                     0.4278 |                     56.1538 |              13.9040 |
|          8.5000 |        6 | 50-55%                     | 1052 |                     0.5224 |     51.4259 |                     0.1951 |                     57.5342 |               2.4748 |
|          8.5000 |        6 | 55-60%                     |  914 |                     0.5745 |     53.2823 |                     0.5113 |                     58.3138 |               1.9134 |
|          8.5000 |        6 | 60-65%                     |  541 |                     0.6231 |     49.3530 |                    -0.7657 |                     57.2993 |               3.5927 |
|          8.5000 |        6 | 65-70%                     |  258 |                     0.6705 |     56.5891 |                    -0.0781 |                     50.8929 |               3.8513 |
|          8.5000 |        8 | 50-55%                     | 1051 |                     0.5225 |     57.7545 |                     0.1044 |                     56.5315 |               3.2356 |
|          8.5000 |        8 | 55-60%                     |  913 |                     0.5744 |     53.9978 |                    -0.8208 |                     54.2857 |               1.0454 |
|          8.5000 |        8 | 60-65%                     |  538 |                     0.6231 |     53.3457 |                    -0.8312 |                     55.7769 |               2.5394 |
|          8.5000 |        8 | 65-70%                     |  256 |                     0.6705 |     55.4688 |                     1.1475 |                     63.1579 |               8.5698 |
|          9.0000 |        6 | 50-55%                     |  664 |                     0.5224 |     51.0542 |                     1.0882 |                     54.4615 |               5.0748 |
|          9.0000 |        6 | 55-60%                     |  582 |                     0.5727 |     51.0309 |                    -0.0357 |                     53.3333 |               4.9435 |
|          9.0000 |        6 | 60-65%                     |  353 |                     0.6226 |     54.1076 |                     1.5175 |                     61.1111 |               7.4635 |
|          9.0000 |        6 | 65-70%                     |  182 |                     0.6701 |     54.3956 |                     1.0687 |                     59.0361 |               6.9140 |

## Interpretation
- If joint-zero dependence is near zero, the full-inning anomaly is largely explained by the two half-inning marginals rather than an extra game-level coupling effect.
- If U0.5 changes materially with favorite strength at the same game total, the full-game total is hiding run-allocation asymmetry; moneyline/team-strength information may improve inning pricing.
- If neither effect is meaningful and stable, the empirical total×inning base rate should remain the preferred prior rather than adding complexity.
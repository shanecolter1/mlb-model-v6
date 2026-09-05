# MLB Full-Inning Under/Over 0.5 Consistency Analysis — 2021–2025

## Objective

Identify which **inning × DraftKings pregame opening full-game total** combinations have the most stable historical full-inning Under/Over 0.5 probability.

The empirical probability already exists in the historical benchmark. This analysis therefore does **not** create a replacement probability score. It measures how consistently that probability repeated by season.

## Metric

For each inning × pregame-total cell:

- **Over 0.5 probability** = P(full inning scores 1+ runs).
- **Under 0.5 probability** = 1 − P(1+).
- **Consistency metric** = sample standard deviation of the season-specific P(1+) values, expressed in percentage points.
- Because Under 0.5 is the complement of Over 0.5, **Under and Over have exactly the same year-to-year standard deviation**.
- Lower annual SD = more consistent.
- Annual range is included as an intuitive secondary dispersion measure.
- Pooled N and the minimum season N are retained to prevent small-sample cells from being mistaken for stable signals.

The practical ranking uses **CORE cells with pooled N ≥ 500**. Cells below 500 observations remain in the companion CSV and are tagged `THIN`, but are excluded from the primary rankings.

The benchmark covers matched 2021–2025 MLB regular-season games. The 2025 sportsbook archive is partial through 2025-08-16.

## Main finding

The **most stable CORE cell overall is pregame total 8.0 × I3**:

- Under 0.5: **54.15%**
- Over 0.5: **45.85%**
- Pooled N: **2,146**
- Year-to-year SD: **0.86 pp**
- Five-season range: **2.18 pp**

However, the strongest combination of **high Under probability and high stability** is **pregame total 7.5 × I9**:

- Under 0.5: **65.65%**
- Empirical fair Under odds: **-191**
- Pooled N: **1,607**
- Year-to-year SD: **0.98 pp**
- Five-season range: **1.95 pp**

The next especially strong I9 cell is **8.5 × I9** at **66.79% Under**, fair **-201**, with **1.22 pp** annual SD over **2,861** observations.

## Most consistent CORE cells

| Pregame total | Inning | N | Under 0.5 | Over 0.5 | Annual SD (pp) | Annual range (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | I3 | 2,146 | 54.15% | 45.85% | 0.86 | 2.18 |
| 7.5 | I9 | 1,607 | 65.65% | 34.35% | 0.98 | 1.95 |
| 8.5 | I9 | 2,861 | 66.79% | 33.21% | 1.22 | 2.77 |
| 8.5 | I1 | 2,871 | 51.24% | 48.76% | 1.33 | 2.96 |
| 8.5 | I7 | 2,867 | 55.56% | 44.44% | 1.51 | 3.67 |
| 8 | I8 | 2,144 | 54.24% | 45.76% | 1.52 | 3.98 |
| 9 | I5 | 1,839 | 51.22% | 48.78% | 1.71 | 4.31 |
| 8 | I7 | 2,144 | 56.81% | 43.19% | 1.72 | 4.06 |
| 8.5 | I4 | 2,871 | 52.39% | 47.61% | 1.73 | 4.39 |
| 9 | I4 | 1,839 | 49.37% | 50.63% | 1.75 | 3.72 |
| 7 | I5 | 655 | 55.73% | 44.27% | 1.75 | 4.37 |
| 9 | I9 | 1,830 | 63.83% | 36.17% | 1.82 | 4.08 |
| 7.5 | I2 | 1,612 | 58.06% | 41.94% | 1.87 | 4.53 |
| 7 | I2 | 655 | 63.82% | 36.18% | 1.89 | 3.88 |
| 8 | I4 | 2,146 | 54.01% | 45.99% | 1.92 | 5.10 |

## High-probability Under 0.5 cells

The following CORE cells have pooled Under 0.5 probability of at least 60%, ranked by consistency.

| Pregame total | Inning | N | Under 0.5 | Fair Under | Annual SD (pp) | Annual range (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 7.5 | I9 | 1,607 | 65.65% | -191 | 0.98 | 1.95 |
| 8.5 | I9 | 2,861 | 66.79% | -201 | 1.22 | 2.77 |
| 9 | I9 | 1,830 | 63.83% | -176 | 1.82 | 4.08 |
| 7 | I2 | 655 | 63.82% | -176 | 1.89 | 3.88 |
| 8 | I9 | 2,143 | 65.24% | -188 | 2.49 | 5.95 |
| 9.5 | I9 | 871 | 62.00% | -163 | 3.62 | 8.87 |
| 7 | I9 | 651 | 68.97% | -222 | 4.49 | 11.36 |

No CORE inning × total cell has an empirical **Over 0.5 probability ≥55%** in this benchmark. The repeatable directional signal is therefore concentrated on the Under side, especially I9 and selected low-total I2 cells.

## Most stable pregame-total bucket for each inning

| Inning | Most stable total | N | Under 0.5 | Annual SD (pp) |
| --- | --- | --- | --- | --- |
| I1 | 8.5 | 2,871 | 51.24% | 1.33 |
| I2 | 7.5 | 1,612 | 58.06% | 1.87 |
| I3 | 8 | 2,146 | 54.15% | 0.86 |
| I4 | 8.5 | 2,871 | 52.39% | 1.73 |
| I5 | 9 | 1,839 | 51.22% | 1.71 |
| I6 | 8.5 | 2,869 | 52.21% | 1.94 |
| I7 | 8.5 | 2,867 | 55.56% | 1.51 |
| I8 | 8 | 2,144 | 54.24% | 1.52 |
| I9 | 7.5 | 1,607 | 65.65% | 0.98 |

## Interpretation

Pure consistency and betting usefulness are different concepts. The 8.0 × I3 cell is the most consistent overall, but its 54.15% Under rate is only moderately directional. The 7.5 × I9 and 8.5 × I9 cells are more compelling empirical benchmarks because they combine a materially higher Under probability with unusually low year-to-year dispersion.

For model conditioning, keep **probability** and **consistency** as separate fields. Do not blend them into a new probability. The empirical probability answers *how often* the outcome occurred; annual SD answers *how reliably that rate repeated*.

## Data notes

- Full inning = visiting-team runs + home-team runs in that inning.
- An inning enters the denominator when its top half was reached; if the home ninth was not played, its contribution is 0 under the existing benchmark rule.
- Pregame conditioner = DraftKings opening full-game total.
- Core ranking range in this report is totals 7.0–9.5 because those inning cells exceed the N ≥ 500 practical threshold.
- Totals 6.0, 6.5, 10.0, 10.5, and 11.0 are retained in the companion CSV as `THIN` for transparency.
- The companion CSV contains season-specific rates, pooled probabilities, fair odds, annual SD, annual range, pooled standard error, sample tier, and core stability rank.

## Source assets

- `data/external/MLB_Game_Stats_Joined_2021_2025.manifest.json`
- Historical benchmark workbook: `MLB_Inning_Benchmark_Tables_2021_2025.xlsx`
- Canonical joined master release: `historical-mlb-2021-2025-v1`

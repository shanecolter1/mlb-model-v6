# I9 Under 0.5 — Home Favorite Structural Backtest (2021–2025)

## Purpose
Test whether strong pregame home favorites create an exploitable structural I9 Under 0.5 signal because the bottom of the ninth is more likely to be skipped.

## Important limitation
The canonical historical master contains DraftKings full-game opening totals and moneylines, but **does not contain historical I9 market prices**. Therefore this report establishes historical hit rates, structural decomposition, and fair prices; it does not claim realized sportsbook ROI. A true ROI test requires archived I9 Under 0.5 prices.

## Method
Sample: matched 2021–2025 games with DraftKings opening totals 7.0–9.0 and a posted opening home moneyline. I9 is counted only when the top of the ninth was reached. A blank home ninth with a reached top ninth is classified as bottom-nine skipped.

## Strongest threshold cells (N >= 100)

| Total | Home ML threshold | N | B9 skipped | I9 U0.5 | Fair U | U0.5 if B9 played | Annual SD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.5 | -250 or stronger | 130 | 70.8% | 80.8% | -420 | 60.5% | 4.16 pp |
| 8.5 | -225 or stronger | 242 | 72.7% | 78.5% | -365 | 60.6% | 7.26 pp |
| 8.5 | -200 or stronger | 374 | 66.8% | 75.9% | -316 | 59.7% | 6.96 pp |
| 7.0 | -175 or stronger | 146 | 58.2% | 75.3% | -306 | 60.7% | 9.08 pp |
| ALL | -300 or stronger | 146 | 67.1% | 73.3% | -274 | 39.6% | 9.37 pp |
| 7.0 | -150 or stronger | 231 | 56.3% | 73.2% | -273 | 58.4% | 4.36 pp |
| ALL | -250 or stronger | 432 | 65.5% | 72.5% | -263 | 49.0% | 5.60 pp |
| 8.0 | -200 or stronger | 265 | 59.2% | 72.5% | -263 | 56.5% | 8.06 pp |
| ALL | -225 or stronger | 762 | 65.0% | 72.4% | -263 | 52.1% | 6.29 pp |
| ALL | -200 or stronger | 1193 | 60.9% | 71.7% | -253 | 53.3% | 4.62 pp |
| 8.5 | -150 or stronger | 969 | 57.3% | 71.5% | -251 | 55.3% | 2.92 pp |
| 8.5 | -175 or stronger | 609 | 61.2% | 71.4% | -250 | 54.2% | 3.12 pp |
| 8.0 | -225 or stronger | 174 | 61.5% | 70.7% | -241 | 49.3% | 12.21 pp |
| 9.0 | -225 or stronger | 143 | 60.8% | 70.6% | -240 | 46.4% | 13.35 pp |
| ALL | -175 or stronger | 1982 | 58.1% | 70.4% | -238 | 53.3% | 2.14 pp |

## Interpretation
The difference between unconditional I9 Under 0.5 and Under 0.5 conditional on the bottom ninth being played quantifies the structural value created by a skipped home half. Increasing Under probability as the home favorite becomes stronger would support using home-favorite strength as an I9-specific market-state variable.

## Pricing test status
Historical I9 prices are not present in the 2021–2025 canonical archive. The fair-price column is the break-even American price implied by the observed hit rate; it is the correct benchmark for evaluating any archived or live I9 Under quote.
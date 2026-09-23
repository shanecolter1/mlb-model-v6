# I2 bankroll and staking (downstream v1)

This module sizes I2 Under 0.5 and Over 0.5 after the existing prediction and v0.4 Local-CV calibration have completed. It does not run, modify, or fetch data for the prediction engine or its calibration. The frozen v0.4 artifact must have `predictionFrozenBeforeDerivativeMarketRetrieval: true`, `derivativeMarketDataUsed: false`, `i2PriceDataUsed: false`, `localCvProduction: true`, an eligible game, and a row with `v04Calibration.status: APPLIED`. Missing calibration returns **CALIBRATION INPUT MISSING** with zero stake.

## Order of operations

1. Run the current I2 pipeline and freeze its baseball and total-conditioned probability.
2. Run the existing v0.4 Local-CV calibration and save its output. The calibrated `under05Pct` or `over05Pct` becomes the staking probability. The parent game's total-conditioned `under05` or `over05` is displayed as the raw model probability before Local-CV.
3. Create a selection file with the bankroll and the game/market identifiers. It must contain no prices.
4. The staking command verifies and freezes every selected calibrated probability in memory, then reads a separate quote file.
5. It converts American odds, computes EV and full Kelly, computes an exact finite-horizon *first-passage* operational floor probability, and chooses the lower of the configured Kelly target and risk ceiling. A negative expected log-growth proposal is rejected.

The calibrated probability is taken from the validated artifact; a supplied `calibrated_win_probability` must match it. Missing calibration is never replaced by an empirical, raw, implied, or unvalidated number. No second confidence multiplier is applied. A separate price must be specified for each game and I2 side. The module does not retrieve sportsbook odds itself.

## CLI

Save a local selection file, for example:

```json
{
  "bankroll": {"current_bankroll": 10000, "session_start_bankroll": 10000},
  "risk": {"kelly_multiplier": 1, "risk_horizon_bets": 100, "max_ruin_probability": 0.1},
  "wagers": [{"game_identifier": 823738, "market_identifier": "I2_UNDER_0.5"}],
  "open_wagers": [{"game_identifier": 823738, "market_identifier": "I2_OVER_0.5", "stake_dollars": 50}]
}
```

Save a separate quote file **after** the probability artifact exists:

```json
[{"game_identifier": 823738, "market_identifier": "I2_UNDER_0.5", "american_odds": -165, "sportsbook": "example", "timestamp": "2026-09-08T22:00:00Z"}]
```

Then run:

```sh
node src/pipeline/size_i2_bankroll.mjs \
  --predictions data/runtime/i2/YYYY-MM-DD_v04_predictions.json \
  --selections /private/path/selections.json \
  --quotes /private/path/quotes.json \
  --output /private/path/staking-report.json
```

`--output` is optional; the full human-readable report always prints to stdout. Keep bankroll/quote files and output outside the public repository. The optional JSON report is written through a temporary file with private permissions. If any selected probability is not frozen and calibrated, no quote file is opened and every selected stake is zero.

For direct integration, import `freezeI2StakingProbability`, `sizeI2Wager`, `finiteHorizonFirstPassage`, `settleI2Wager`, `summarizeI2Portfolio`, or `sizeI2Batch` from `src/staking/i2_bankroll.js`. `sizeI2Batch` accepts a lazy `loadQuotes` callback and calls it only after probability checks.

## Mathematical and operational definitions

For probability `p`, net payout `b`, and stake fraction `f`, EV per dollar risked is `p*b - (1-p)`, Kelly is `max(0, EV/b)`, and expected log growth is `p*log(1+f*b) + (1-p)*log(1-f)`. The default multiplier is 1, so full Kelly is the growth target. Stake dollars are rounded **down** to cents, then log growth and actual risk are recomputed using that executable stake.

The default operational floor is 10% of **session-start bankroll**, held fixed as current bankroll changes. The default ceiling is a 10% probability of touching or crossing that floor **at any point** during the next 100 sequential bets. The exact binomial dynamic program absorbs paths on their first crossing; it does not use ending-bankroll probability or Monte Carlo. A binary search finds the greatest permitted constant fraction for a hypothetical future sequence of identical probabilities and payouts. The final stake never rises to that ceiling merely because risk capacity is available.

The risk projection assumes the same calibrated `p`, price `b`, and fraction `f` for each future sequential wager. Its horizon is a sizing scenario, not a forecast that 100 identical I2 opportunities will occur. It does not jointly model overlapping bets or changing probabilities/prices. A maximum exact horizon of 1,000 bets bounds computation; invalid/unavailable risk calculations return **zero stake**, never an uncapped Kelly stake.

Literal $0 ruin is distinct from the operational floor. With fractions strictly between 0 and 1, ordinary loss multipliers remain positive, so literal $0 is not reached in finitely many bets. The output reports both labels separately. At a 100% fraction, one loss can reach literal $0; the module reports that case separately.

Individual Kelly sizes are preserved in the portfolio summary. Total exposure includes already open wagers and new recommendations. Multiple markets in the same game or explicit `correlation_group` are reported together. Cross-game correlations and simultaneous portfolio first passage are not modeled; the report states **PORTFOLIO CORRELATION NOT MODELED**. No arbitrary portfolio multiplier is applied.

The pre-existing I2 research/validation gates still govern whether a projection is eligible for live wagering. This module only sizes wagers from a marked frozen calibrated artifact; it does not certify model accuracy, quote freshness, or the independence of future wagers.

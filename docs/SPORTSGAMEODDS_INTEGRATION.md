# SportsGameOdds MLB inning-market integration

## Purpose

This integration is a read-only sportsbook market adapter. It does **not** change I2 or all-inning prediction methodology, calibration, upstream data retrieval, or model inputs.

The prediction artifact must exist first. Price retrieval then runs in a separate post-freeze phase for market enumeration, fair-price comparison, EV, and staking only.

## Supported market targets

The adapter requests the SportsGameOdds documented MLB inning identifiers:

- Full-inning Over/Under: innings 1-8 are documented and requested; 9th-inning full markets are enabled automatically if `/markets` reports support for the configured books.
- Full-inning 3-way moneyline (away / draw / home): innings 1-8 are documented and requested; 9th-inning 3-way is enabled automatically if supported.
- Away-team inning Over/Under: innings 1-9. In MLB this maps to the top half-inning.
- Home-team inning Over/Under: innings 1-9. In MLB this maps to the bottom half-inning.

The provider's `/markets` endpoint is queried on every run for the requested market IDs and bookmakers so actual support can be audited rather than assumed. Ninth-inning full-total/3-way IDs are treated as candidates and are sent to `/events` only when `/markets` confirms support.

`includeAltLines=true` is enabled by default. Main and alternate bookmaker lines are flattened into the normalized price board, so a modeled 0.5 line is retained even when a sportsbook's displayed main inning total is 1.5 or another number.

## Secret

Create one GitHub Actions repository secret:

`SPORTSGAMEODDS_API_KEY`

Do not commit the key to source control. The adapter authenticates with the `x-api-key` request header.

The existing `ODDS_API_KEY` remains untouched and continues to belong to The Odds API. This separation prevents accidental provider/key crossover.

## Default Rookie-plan books

The initial price board requests:

- DraftKings (`draftkings`)
- FanDuel (`fanduel`)
- BetMGM (`betmgm`)
- Caesars (`caesars`)

The list is configurable with `SGO_BOOKMAKERS`.

## Runtime files

For date `YYYY-MM-DD` the post-freeze pipeline writes:

- `data/runtime/i2/YYYY-MM-DD_sportsgameodds_inning_markets.json`
- `data/runtime/i2/YYYY-MM-DD_sportsgameodds_market_support.json`
- `docs/inning_markets/YYYY-MM-DD_sportsgameodds_inning_markets.csv`

## Market isolation enforcement

`src/market/sportsgameodds_data_source.mjs` refuses price retrieval unless the caller supplies a valid post-freeze context with `projectionFrozen: true` and a valid `frozenAt` timestamp.

`src/pipeline/fetch_sgo_mlb_inning_markets.mjs` obtains that timestamp from the existing frozen prediction artifact. If the artifact does not exist or lacks a valid timestamp, the odds request fails before contacting SportsGameOdds.

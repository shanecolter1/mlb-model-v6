# Sportsbook Data Governance

## Canonical source

**The Odds API is the single canonical external source for sportsbook data used by model workflows in this repository.**

Do not scrape sportsbook websites, use search-engine snippets, betting aggregators, media pages, consensus pages, or substitute public odds sources when The Odds API is configured and available.

The API credential must be supplied through the environment variable `ODDS_API_KEY`. Never commit the key to the repository.

## Market isolation

Baseball prediction engines must remain isolated from betting-market information until predictions are frozen.

### Pre-freeze

The only market-derived input permitted for the MLB I2 prediction engine is:

- Bookmaker: DraftKings
- Market: full-game total
- Field retained: total **point only**
- Allowed metadata: event id, teams, commence time, bookmaker identity, capture/update timestamps

The following are prohibited before prediction freeze:

- prices/juice
- moneylines
- run lines
- alternate lines
- I2/inning derivative prices
- implied probabilities
- consensus prices
- line movement
- betting percentages
- betting commentary or market-derived ratings

The pre-freeze artifact scope must remain `FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES`.

### Post-freeze

After model probabilities are written to the frozen prediction artifact, The Odds API may be used for sportsbook market enumeration and pricing. This includes FanDuel, DraftKings, and other supported books and derivative markets. Post-freeze sportsbook data may be used only to:

- enumerate available wagers
- retrieve prices
- calculate break-even probability
- compare price with frozen fair probability/fair odds
- calculate EV
- apply qualification and staking rules

Post-freeze market data must never be fed back into the frozen baseball probability engine.

## Source hierarchy

For sportsbook data:

1. **The Odds API** — canonical live source.
2. Validated repository artifact previously captured from The Odds API — operational fallback only when the live API is temporarily unavailable and the workflow explicitly permits using a locked historical point.
3. No other sportsbook-data source is authorized unless the user explicitly changes this governance.

Baseball/fundamental data retain their separate source hierarchy and are not governed by this document.

## Implementation

The central adapter is `src/market/sportsbook_data_source.mjs`.

All new sportsbook retrieval code should import from that adapter rather than calling sportsbook websites or constructing independent odds-provider integrations.

The I2 daily workflow injects `ODDS_API_KEY` from GitHub Actions secrets and records `THE_ODDS_API` as its sportsbook source.

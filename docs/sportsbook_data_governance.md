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

After model probabilities are written to the frozen prediction artifact, The Odds API may be used for sportsbook market enumeration and pricing. Post-freeze sportsbook data may be used only to enumerate available wagers, retrieve prices, calculate break-even probability, compare price with frozen fair probability/fair odds, calculate EV, and apply qualification/staking rules. Post-freeze market data must never be fed back into the frozen baseball probability engine.

For every model output, retrieve and display an availability/price field for each of these requested books/markets:

1. FanDuel (`fanduel`, US)
2. DraftKings (`draftkings`, US)
3. Hard Rock Bet (`hardrockbet`, US2)
4. bet365
5. Fanatics (`fanatics`, US)
6. Caesars (`williamhill_us`, US)
7. BetMGM (`betmgm`, US)
8. Kalshi (`kalshi`, US exchange)
9. Polymarket (`polymarket`, US exchange)
10. Pinnacle (`pinnacle`, EU)

If a requested source does not return the requested event/market, the output must explicitly show `NOT_RETURNED_FOR_EVENT_MARKET`. If the canonical provider does not support the requested source for the relevant sport/region, show `UNSUPPORTED_BY_CANONICAL_SOURCE`. Never silently omit a requested source and never replace it with another book.

Current canonical-provider notes:
- The Odds API presently lists Kalshi and Polymarket as US exchanges.
- Pinnacle is listed under the EU region; The Odds API notes that its Pinnacle data is sourced from the public website and may therefore be delayed.
- The Odds API does not presently list a US MLB bet365 bookmaker key. Until that changes, bet365 must remain visible in output as `UNSUPPORTED_BY_CANONICAL_SOURCE`, rather than being scraped or substituted.

For each available source, output the exact market name/point, Under price, Over price, provider update timestamp, model probability, break-even probability, fair odds, and EV for each side when mathematically applicable.

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

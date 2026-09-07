# I2 Historical Market Source Research — Phase 3 Checkpoint

Date: 2026-09-07
Branch: analysis/i2-weekday-edge-20260906

## Objective
Identify a self-serve historical source that reaches the 2025 MLB season and explicitly supports standalone 2nd-inning game totals with sportsbook-specific prices.

## Best candidate: SportsGameOdds

### Exact MLB I2 market confirmed
SportsGameOdds documents the oddID structure as:

`{statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}`

For the MLB game-wide 2nd-inning run total:

- Over: `points-all-2i-ou-over`
- Under: `points-all-2i-ou-under`

Relevant definitions:
- `points` = the sport's primary scoring statistic (runs for baseball)
- `all` = game-wide total rather than home/away team
- `2i` = 2nd Inning
- `ou` = Over/Under
- `over` / `under` = wager side

SportsGameOdds' MLB market guide explicitly states that individual innings 1st through 8th support per-inning Moneyline, Spread, Total, and 3-Way markets, and gives the pattern `points-all-1i-ou-over` for individual inning totals. The documented oddID grammar makes the second-inning pair above deterministic.

### Sportsbook support
SportsGameOdds states its MLB individual-inning markets are supported by books including Pinnacle, bet365, DraftKings, BetMGM, Caesars and others, with availability varying by market.

Pro plan bookmaker coverage includes Pinnacle, DraftKings, FanDuel, BetMGM, Caesars, Circa and 80+ total books.

### Historical depth
SportsGameOdds FAQ states historical odds data is available on Pro and above, with coverage generally beginning in February 2024. This encompasses the full 2025 MLB season.

### Critical 2025 caveat
SportsGameOdds states explicit bookmaker-level fields:
- `openOdds`
- `closeOdds`
- `openOverUnder`
- `closeOverUnder`

are available since January 2026.

Therefore 2025 historical event data may retain sportsbook-specific I2 prices, but cannot be assumed to contain explicit opening/closing markers. One actual 2025 event must be tested before the source is promoted.

### Access and pricing
- Amateur: $0, 2,500 objects/month, 10 requests/minute; real current data but no historical-data entitlement.
- Rookie: $99/month; no historical-data entitlement listed.
- Pro: $299/month; historical data included.
- Pro offers a self-serve 7-day free trial.
- API key is delivered immediately after self-serve signup.
- No sales contact is required for Pro.

The public FAQ states the free Amateur plan can be started without a credit card; paid-plan trial billing requirements should be verified at checkout before starting the Pro trial.

## Exact Phase 4 proof query
Once a Pro-trial API key is available, query one known 2025 MLB date/event using only the two I2 oddIDs and sharp/major books.

Conceptual request:

`GET https://api.sportsgameodds.com/v2/events`

Parameters:
- `leagueID=MLB`
- `startsAfter=<2025 date start>`
- `startsBefore=<2025 date end>`
- `finalized=true`
- `oddIDs=points-all-2i-ou-over,points-all-2i-ou-under`
- `bookmakerID=pinnacle,draftkings,fanduel`
- `includeAltLines=true` if needed to expose the 0.5 line

Success criteria:
1. 2025 finalized MLB event is returned.
2. `points-all-2i-ou-over` / `under` are present.
3. O/U line includes 0.5 (primary I2 market).
4. Book-specific prices exist for at least Pinnacle; DK/FD are secondary.
5. Determine what timestamp or historical-price semantics survive for 2025.
6. If explicit opener is unavailable, determine whether returned price is a pregame snapshot suitable for market-efficiency validation.

## Source decision
SportsGameOdds is **approved for a one-game 2025 proof test**, but **not yet approved for bulk extraction** until the Phase 4 API test confirms actual 2025 I2 rows and clarifies the price timestamp/opening limitation.

## Rejected/secondary sources retained from prior phases
- OddsSafari: exact I2 API exists currently but archived 2025 market payloads return HTTP 500; rejected.
- SportsDataIO: exact I2 market exists but historical access is sales/entitlement gated; rejected by project requirements.
- The Odds API: reaches 2025 but official MLB period-market catalog does not contain standalone 2nd inning totals; rejected for I2.
- OddsPapi: I2 capable but history starts January 2026; rejected for 2025.
- PropLine: I2 capable but historical line movement begins April 2026; rejected for 2025.
- TheRundown: deep archive but 2025 access is expensive and individual-inning MLB support has not been proven; lower priority.

## Next phase
Phase 4 = one-game SportsGameOdds 2025 API proof. Do not build bulk extraction until this passes.

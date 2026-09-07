# I2 Historical Market Source Research — Phase 2 Checkpoint

Date: 2026-09-07
Branch: analysis/i2-weekday-edge-20260906

## Objective
Determine whether OddsSafari's archived 2025 MLB pages can still return exact standalone 2nd-inning Over/Under market data through the site's active API.

## Exact OddsSafari API contract recovered
Current match-page JavaScript uses:

`/api/match?EventID={event}&MarketTypeID={market_type}&ScopeID={scope}&Lang=en`

Bookmaker line-history UI uses:

`/api/get_match_page_bookmaker_history?OddsID={odds_id}&BookmakerID={bookmaker_id}&Lang=en`

## Exact market and scope IDs established from a current MLB event
Using current MLB EventID 2317393 (PHI Phillies vs ATL Braves):

- `MarketTypeID = 905` = `Over/Under Runs`
- `ScopeID = 905` = `2nd Inning`
- Current API response is HTTP 200.
- Current API response contains one I2 bet and explicit `Over` / `Under` outcomes.
- Current response returns `ParamID = 7529` for this I2 market.

This proves OddsSafari's application still supports standalone 2nd-inning total markets today.

## Archived 2025 test
Archived 2025 target:
- PHI Phillies vs TOR Blue Jays
- EventID = 1944521
- Date = 2025-06-13

Exact request tested:

`/api/match?EventID=1944521&MarketTypeID=905&ScopeID=905&Lang=en`

Result:
- HTTP 500 Internal Server Error

Control tests against other candidate market types at the same archived event/scope also returned HTTP 500.

The archived match page itself currently renders the game/event metadata but reports `No Markets Available`.

## Conclusion
OddsSafari is **rejected as the production historical 2025 I2 source**.

Reason:
- Search-indexed/cached 2025 pages prove the site previously exposed inning markets.
- The current application/API proves the exact 2nd-inning market definition and request parameters.
- However, the active API no longer serves the archived 2025 market payload, even when called with the exact correct market and scope IDs.

This is a data-retention problem, not a selector or scraping problem. Further UI reverse engineering is not warranted.

## Durable Phase 2 artifacts
- `analysis_outputs/oddssafari_phase2_probe.txt`
- `analysis_outputs/oddssafari_phase2_network.json`
- `analysis_outputs/oddssafari_phase2_page.html`
- `analysis_outputs/oddssafari_phase2.png`
- `analysis_outputs/oddssafari_phase2_endpoint_summary.txt`
- `analysis_outputs/oddssafari_phase2_id_test.txt`

## Phase 3 target
Find a self-serve source that satisfies ALL of the following:
1. Historical coverage reaches the 2025 MLB season.
2. Standalone `2nd inning` / `I2` total market is explicitly supported.
3. Over and Under prices are available, preferably by sportsbook.
4. Opening price or timestamped line history is available, or at minimum a defensible pregame snapshot.
5. Access is self-serve; no sales call/manual entitlement request.

Priority lead for Phase 3: `odds-api.net`, followed by other historical sportsbook/odds-comparison archives only if the 2025 history requirement is independently verified.

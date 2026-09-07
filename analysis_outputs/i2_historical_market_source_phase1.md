# I2 Historical Market Source Research — Phase 1 Checkpoint

Date: 2026-09-07
Branch: analysis/i2-weekday-edge-20260906

## Objective
Validate a self-serve historical source for standalone MLB 2nd-inning total markets, with enough archived pricing detail to benchmark the I2 model without a vendor sales-gate.

## OddsSafari validation completed

Confirmed public archived 2025 MLB game pages exist, including:
- PHI Phillies vs TOR Blue Jays — 2025-06-13: https://www.oddssafari.com/matches/baseball/usa/mlb/phi-phillies-vs-tor-blue-jays/1944521
- KC Royals vs NY Yankees — 2025-06-11: https://www.oddssafari.com/matches/baseball/usa/mlb/kc-royals-vs-ny-yankees/1943327
- MIN Twins vs TEX Rangers — 2025-06-12: https://www.oddssafari.com/matches/baseball/usa/mlb/min-twins-vs-tex-rangers/1943963
- BAL Orioles vs DET Tigers — 2025-06-12: https://www.oddssafari.com/matches/baseball/usa/mlb/bal-orioles-vs-det-tigers/1943961

Each archived page visibly exposes inning-period selectors including:
- Full Time+EI
- First 5 Innings
- 1st Inning
- 2nd Inning
- 3rd through 8th inning on 2025 archived pages

Each page also exposes an `Over/Under Runs` market family selector and bookmaker comparison tables. Pinnacle appears on the archived 2025 examples above.

OddsSafari also states on the game pages that hovering/clicking the odds-change icon allows users to browse historical odds changes.

## What is proven
1. 2025 MLB events are still publicly addressable by stable match URLs.
2. 2nd Inning is a distinct period available on archived 2025 MLB pages.
3. Over/Under Runs is a distinct market family available on those pages.
4. Pinnacle is represented on archived 2025 MLB pages.
5. Historical odds changes are exposed in the page UI.
6. No paid account or sales contact has been shown as necessary to view these archived pages.

## What is NOT yet proven
The static HTML/search extraction available in the current environment does not preserve the dynamic UI state needed to force the combination:
`Over/Under Runs -> 2nd Inning -> O/U 0.5`.

Therefore the bookmaker values shown by a plain page fetch cannot yet be safely labeled as I2 totals. They may correspond to whichever market/period the page rendered by default when indexed.

## Extraction blocker
Current web extraction can read the archived page and its available buttons, but cannot click the JavaScript period/market controls. Direct Python/container network access to OddsSafari is also unavailable in this environment because outbound DNS is disabled.

## Phase 2 target
Identify the dynamic request used by OddsSafari when selecting:
`Over/Under Runs -> 2nd Inning`.

Success criteria for Phase 2:
- obtain one archived 2025 game's actual I2 O/U line (expected primary total 0.5),
- obtain Pinnacle Over and Under prices,
- determine whether opening/history points are directly retrievable,
- document the request parameters/endpoint so it can be automated across many games.

## Source decision at Phase 1
OddsSafari remains the best no-sales-gate historical lead, but it is **not yet approved as the production source** until Phase 2 proves exact I2 extraction rather than only UI availability.

# I2 Historical Market Source Research — Phase 4 Checkpoint

Date: 2026-09-07
Branch: analysis/i2-weekday-edge-20260906

## Objective
Run one exact 2025 MLB standalone 2nd-inning total query against SportsGameOdds before building any bulk extraction.

## Proof harness completed
Workflow:
`.github/workflows/i2-sportsgameodds-phase4-proof.yml`

Saved output:
`analysis_outputs/sportsgameodds_phase4_proof.json`

The workflow is guarded and never prints or commits an API key.

## Exact proof request preserved
- leagueID: MLB
- startsAfter: 2025-06-13T00:00:00Z
- startsBefore: 2025-06-14T00:00:00Z
- finalized: true
- oddID: `points-all-2i-ou-over,points-all-2i-ou-under`
- bookmakerID: `pinnacle,draftkings,fanduel`
- includeAltLines: true
- limit: 25

## Current result
`BLOCKED_NO_API_KEY`

The repository does not currently have a `SPORTSGAMEODDS_API_KEY` secret. No API call was attempted and no quota was consumed.

## What is already proven before the API call
1. SportsGameOdds explicitly supports individual MLB inning totals.
2. Exact I2 oddIDs are `points-all-2i-ou-over` and `points-all-2i-ou-under`.
3. Historical data generally reaches February 2024 and therefore can potentially cover 2025.
4. Pro is self-serve and includes historical data.
5. Pro offers a 7-day free trial.
6. Major/sharp bookmaker coverage includes Pinnacle and DraftKings; actual 2025 per-market availability must still be proven.
7. Bookmaker-level explicit opening/closing fields are documented only from January 2026 onward, so 2025 price semantics must be inspected empirically.

## Next action
Obtain a self-serve SportsGameOdds Pro-trial API key and store it as the repository secret:
`SPORTSGAMEODDS_API_KEY`

Then re-run the existing Phase 4 proof workflow. Do not build bulk extraction until the one-game response proves 2025 I2 coverage and clarifies what historical price/timestamp is retained.

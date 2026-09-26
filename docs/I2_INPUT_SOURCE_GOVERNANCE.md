# I2 production baseball input sourcing

This is an input-only layer shared by the existing production entry points. No model,
empirical, calibration, EV, or Kelly formula changed. Confidence and audit weights
never enter simulation. The existing full-game-total conditioning exception remains
unchanged; derivative prices remain post-freeze. Source modules do not import market code.

## Production flow

`run_i2_today.mjs` collects sources, resolves each game, resolves exact MLB identities,
runs the unchanged simulator, and checks sources again immediately before freeze.
`run_i2_today_upstream_wrapper.mjs`, `run_i2_full_slate_override.mjs`, and
`run_i2_total_conditioned.mjs` all use that runner. The legacy preliminary builder
now uses the same resolver. The official-lineup builder remains an MLB-only diagnostic;
it does not override the production resolver. Old unverified date-specific override
files are no longer injected into MLB feed responses.

The user-owned MLB upstream remains read-only. Historical schedule range requests
fall through to MLB because the upstream's single-date schedule interface cannot
represent those queries. MLB identity resolution is team-scoped: exact full-name
matches win first; if no exact match exists, a second pass removes accents/punctuation
and only recognized generational suffixes (Jr., Sr., II, III, IV). That fallback is
accepted only when exactly one player on the game's MLB team roster/boxscore matches.
There is no fuzzy, surname-only, or cross-team matching. RotoWire IDs are never
interpreted as MLB IDs. Zero or multiple matches block that game without crashing the
slate. Doubleheaders require both-team and start-time matching within 30 minutes;
ambiguous matches are rejected.

Lineups: posted TEAM / BEAT / confirmed RotoWire; MLB final-system confirmation;
otherwise expected RotoWire, RosterResource, previous completed MLB game. Explicit
newer verified scratch/rest news can replace a named player at a specified position.
Without an explicit replacement, the lineup is unresolved: no invented substitute.
A partial news substitution is not called a fully confirmed lineup.

Starters: confirmed TEAM / BEAT, RotoWire Projected Starters, RosterResource,
other verified reporting, MLB probable pitcher, explicitly labeled fallback.
Any fresh disagreement is `STARTER_CONFLICT`, even if one source has higher priority.
A preliminary simulation can be shown but cannot produce an actionable recommendation.
There is no automatic age-based conflict dismissal within the 24-hour freshness window.
Correct or refresh the source records to resolve a conflict.

`inputAudit` retains sources, timestamps, provider failures, lineups, ordered-slot
changes, added/removed players, top-four differences, projected-to-confirmed accuracy,
news and SRM review requests. `inputSourceAudit` is the compact per-game output.
Weighted slot accuracy gives slots 1–4 twice the audit weight of other slots; it does
not change production batting-order weights. RotoWire/RosterResource exact agreement
in at least 8 slots with the same top four is HIGH; smaller lower-order disagreement
is MEDIUM; top-four disagreement, multiple substitutions, missing validation, and
unresolved news are LOW. LOW alone does not change probabilities or block a bet.

A fresh simulation satisfies invalidation of the previous run. A change detected after
that simulation invalidates this freeze and requires a clean rerun. No old probability
is adjusted to approximate a new starter or order. The market range report revalidates
baseball inputs before recommendations; a failed check or missing input audit blocks
recommendations and Kelly output. Invalidated probabilities remain for audit only.

## Source access and configuration

### RotoWire

The default transport is the public daily lineup page, fetched once per collection
pass for both lineups and starters, without login or API credentials. Only game date,
ET start time, team codes, full names, batting order/handedness, explicit confirmation
status and starter/bulk roles enter normalized input objects. The full HTML, scripts,
prices and market blocks are discarded. Page date must equal the requested date;
unknown markup or gated pages fall through safely. Source publication timestamp is
null when the page provides none; retrieval time is recorded separately. PRIM means
bulk pitcher and is never promoted to actual starter. An unidentified opener is flagged
for starter review and blocks recommendations. Public access can change or fail.

`ROTOWIRE_API_KEY` is optional: the API is attempted only when public retrieval or
usable data for that feed is unavailable. Do not purchase API access merely to use the
public-page path. If API fallback is desired, set GitHub Actions secret `ROTOWIRE_API_KEY`. The reusable daily, preliminary, and
v0.4 workflows pass it to the baseball adapter. No key is stored or logged.

Verified official documentation:
- https://rotowire.readme.io/docs/quick-start
- https://rotowire.readme.io/reference/get_baseball-mlb-projectedlineups-php
- https://rotowire.readme.io/reference/get_baseball-mlb-projectedstarters-php

Authentication is the `key` query parameter, with `date` and `format=json`.
The probable-starter feed is separately priced/entitled. An ordinary website
subscription has not been verified to include these feeds. Both endpoints use
`Date`, `Games`, `DateTime`, and `Teams`; batting order is `Players[].BattingSpot`.
Only explicit `LineupStatus` Confirmed/CONFIRMED/C is treated as confirmed; missing
or unknown status stays projected. API failure, no credential, wrong date, unsupported
schema or unknown identity fails safely. Live authenticated responses have not been
validated in this implementation environment. Provider errors are redacted.

### RosterResource — access dependency still outstanding

Public FanGraphs retrieval returned HTTP 403 in implementation testing; no supported
public JSON API contract was verified. **There is no automatic live RosterResource
scraper in this change.** The resolver accepts a reviewed, timestamped, same-date
snapshot via `I2_ROSTERRESOURCE_SNAPSHOT` (workflows use
`config/i2_rosterresource_snapshot.json`). Without it, the run explicitly reports
`ROSTERRESOURCE_UNAVAILABLE` and reduces validation confidence. A permitted feed or
export integration is needed to make this source fully automatic.

Snapshot contract (all player names must be exact MLB names):

```json
{
  "date": "YYYY-MM-DD",
  "generatedAt": "ISO timestamp",
  "games": {
    "MLB_GAME_PK": {
      "away": {
        "lineup": {
          "source": "https://www.fangraphs.com/roster-resource/depth-charts/TEAM",
          "timestamp": "ISO source observation timestamp",
          "retrievedAt": "ISO retrieval timestamp",
          "players": ["Exactly nine full names in batting order"],
          "vsHand": "R"
        },
        "starter": {
          "source": "https://www.fangraphs.com/roster-resource/probables-grid",
          "timestamp": "ISO source observation timestamp",
          "retrievedAt": "ISO retrieval timestamp",
          "name": "Full MLB pitcher name"
        }
      },
      "home": {}
    }
  }
}
```

`vsHand` is optional; when present, it must match the MLB-identified opposing
pitcher's handedness, otherwise the validator is unavailable. Snapshots expire after
24 hours and must match the target date and game ID. They are always projected,
regardless of an accidental confirmed flag.

### Team/beat reports and news

MLB's official RSS (`https://www.mlb.com/feeds/news/rss.xml`) is checked before freeze.
Only recent items mentioning the team's name or an exact player name are associated
with a game. Keywords identify review requests only; narrative never modifies an input
or probability. Coverage is limited to that feed, not every team or beat reporter.
RSS metadata is not proof of an official posted batting order.

Explicit team/beat facts use `I2_BASEBALL_REPORTS` (default
`config/i2_baseball_reports.json`), with the same date/generatedAt/games/away/home
structure. Each side supports `lineups`, `starters`, and `news` arrays. Each record
requires `provider` (TEAM, BEAT, REPORTING), `source` URL, `timestamp`,
`retrievedAt`, `verified: true`, and `explicit: true`. `lineups` requires
`confirmed: true` and nine `players`; `starters` requires `name` and confirmation
status. These fields represent reviewed evidence, not inferred confidence.

News records include `player`, `action`, `reason`, `confidence`, and
`recommendedAction`. Supported factual exclusions: REST, SCRATCH, INACTIVE;
optional `replacement` and 1-based `position` must identify the actual substitution.
Uncertainty actions: PLATOON_UNCERTAINTY, ROSTER_UNRESOLVED, INJURY_UNCERTAINTY.
Recommended actions: NO_CHANGE, LINEUP_UPDATE_RECOMMENDED, STARTER_UPDATE_REQUIRED,
SRM_REVIEW_RECOMMENDED, PROJECTION_INVALIDATED. Activations/call-ups alone do not
prove a starting slot; supply a posted order or explicit substitution.

Automated beat-reporter ingestion is not installed. Operators must supply verified
reports through this contract until an authorized feed is connected.

### SRM limitation

No executable approved Starter Reliability Module exists on the inspected main
branch (`a7b6271`). Workload findings generate structured approval requests with the
current input, evidence, proposed change, affected starter quality/duration component,
and unknown direction/magnitude where the approved module would be needed.
Nothing is applied; before/after deltas are null, not fabricated. Integration and
numerical delta validation require the actual approved SRM implementation. The
finite-horizon staking module is also absent on main; existing Kelly formulas are
untouched.

## Validation

`npm test`, plus every existing `tests/i2*.test.mjs`.
New tests cover the requested hierarchy, confidence, source timestamps, failure paths,
conflicts/invalidation, news facts, market rejection, SRM non-application, recommendation
blocking, and real production-runner numerical parity against the pre-change runner.
The deterministic parity fixture was captured from main `a7b6271` with identical
players, statistics, seed, calibration and 1,000 trials. It is an integration regression,
not a newly fitted model. Live paid RotoWire and live RosterResource feeds remain
unverified as noted above.

Public transport validation: parser fixtures cover confirmation, full names, date/ET DST, doubleheaders, bulk roles, market-field isolation, credential-free retrieval and optional API fallback. The existing I2 unit workflow also runs a live public-access check without API credentials (non-blocking when the third party is unavailable).

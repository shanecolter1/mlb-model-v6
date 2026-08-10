MLB Live Command Center V11.12 — Complete Netlify / 1Tap Deployment

This package preserves the working V11.8 live-feed / Netlify-function base and
incorporates the later dashboard changes requested for the current production UI.

Included:
- index.html
- netlify.toml
- netlify/functions/mlb.js
- netlify/functions/savant.js

Current V11.12 dashboard changes:
- Final half-inning probability display uses internal 0,1,2,3,4,5,6+ buckets.
- Displayed rows are 0,1,2,3,4 with:
  Exact Probability
  Exact Fair American Odds
  Aggregate (At Least) Probability
  Aggregate Fair American Odds
- Runs already scored are incorporated into FINAL half-inning probabilities.
- Pregame probable-pitcher season-stat cards.
- Full batting lineups rather than only the first three hitters.
- Box-score section.
- Pregame data readiness moved to the bottom of the pregame output.
- Scheduled game start time remains displayed from the MLB gameDate feed.
- Reliever performance-rating TTO calculation uses pitcher-specific batters faced,
  preventing a fresh late-inning reliever from inheriting the game's global batting-order count.

Netlify configuration:
- Publish directory: .
- Functions directory: netlify/functions
- Node bundler: esbuild

Required existing environment variables:
- ODDS_API_KEY if sportsbook-data features are enabled by the existing dashboard workflow.

1Tap target:
- Owner: shanecolter1
- Repository: mlb-model-v6
- Branch: main
- Netlify enabled: true
- Publish directory: .
- Functions directory: netlify/functions

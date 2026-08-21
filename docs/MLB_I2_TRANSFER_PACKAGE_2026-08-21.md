# MLB I2 UNDER/OVER MODEL — TRANSFER PACKAGE

**Transfer date:** 2026-08-21  
**Repository:** `shanecolter1/mlb-model-v6`  
**Branch:** `main`  
**Repository head at package creation:** `1acb32d12d2a58bf2d9916b032e2182ca85cd715`  
**Current model state:** **v0.3 Total-Conditioned Research Build**  
**Primary governing spec:** `docs/I2_MODEL.md`

---

## 0. INSTRUCTIONS TO THE RECEIVING CHATGPT / AGENT

This package is intended to let another conversation continue the MLB second-inning project without re-inventing, simplifying, or silently changing the workflow.

Before doing any new analysis:

1. Treat `docs/I2_MODEL.md` in the GitHub repository as the current authoritative model specification.
2. Treat this transfer package as the authoritative project-history/handoff document.
3. Inspect the current repository head before modifying code because GitHub may have advanced after this package was generated.
4. Use the user's own MLB data first. Do not start with public web search.
5. Do not silently change the model architecture, source priority, market-isolation rules, lineup rules, calibration rules, or output format.
6. Do not use a single day's results to tune production coefficients directly. Changes must be tested as challengers and promoted only with out-of-sample evidence.
7. When showing probabilities, **always include fair American odds**.
8. A ranking is not an edge. Always distinguish relative rank from probability lift versus the correct historical prior and from actual sportsbook EV.

---

# 1. PROJECT OBJECTIVE

Build a pregame model for the **full MLB second inning (I2)** that estimates a discrete distribution of total runs scored by both teams combined:

- `P(I2 = 0)`
- `P(I2 = 1)`
- `P(I2 = 2)`
- `P(I2 = 3)`
- `P(I2 >= 4)`

and derives:

- `P(I2 Over 0.5)` = `P(1+)`
- `P(I2 Under 0.5)` = `P(0)`
- `P(2+)`, `P(3+)`, `P(4+)`
- fair American odds for Under and Over
- Top 2 scoring probability
- Bottom 2 scoring probability
- likely I2 starting batting-order slot for each team

The model is intended to become a serious betting research model, but its current status remains **research/shadow**, not production-betting approved.

---

# 2. NON-NEGOTIABLE GOVERNANCE RULES

## 2.1 User-data-first source priority

For every I2 run, lineup check, starter check, audit, postmortem, or model update, use this order:

1. **Shared/read-only upstream MLB model data layer** — the user's machine-readable MLB system and existing MLB model outputs.
2. **GitHub model/runtime artifacts** — current frozen snapshots, model outputs, code, and derived datasets.
3. **User Library datasets** — historical joined game data, inning benchmark tables, saved odds/history files, Retrosheet-derived data, park files, etc.
4. **Official external baseball sources** — only if the required field is genuinely missing/stale/failed or requires verification.
5. **Other public sources** — last resort only, and the fallback must be disclosed.

Do not declare a lineup, starter, roster state, or baseball input unavailable until the first three sources have been checked.

For lineups, the machine-readable live MLB feed in the user's upstream system takes priority over consumer-facing lineup webpages when it is fresher.

Do not silently substitute sources.

## 2.2 Confirmed-lineup rule

An **official ranked I2 projection requires the actual confirmed starting batting order**.

Expected/provisional lineups may be used only in a separately labeled shadow analysis. They must not be mixed into the official ranked slate as if they were confirmed.

This rule was strengthened after the 2026-08-16 postmortem showed that five provisional-lineup games did not match the actual lineups.

## 2.3 Market isolation — current narrow exception

The baseball prediction must remain isolated from derivative betting-market information until the I2 probability artifact is frozen.

### Before freeze, the only market-derived field permitted is:

- the **pregame full-game total point**, used solely as a structural run-environment conditioner.

The pre-freeze I2 engine may receive only:

- full-game total point
- bookmaker/source identity needed to map the point to the historical benchmark
- capture timestamp for audit

It may **not** receive before freeze:

- I2 Under/Over prices
- full-game total juice
- moneylines
- run lines
- alternate run lines
- implied probabilities
- consensus prices
- line movement
- betting percentages
- sportsbook projections
- other market-derived features

After the total-conditioned I2 probability is frozen, the separate Market/Decision Engine may retrieve I2 prices and calculate edge, EV, and staking.

## 2.4 No broad-prior fallback for official projections

If the model cannot obtain an approved full-game total that maps to an exact historical total bucket, the game remains **pending**. Do not silently fall back to the broad all-games I2 prior for an official probability.

## 2.5 Do not confuse rank with edge

A game being ranked #1 Under on a slate does not mean it is a strong bet.

Every report should distinguish:

- total-conditioned historical prior
- baseball-only raw probability
- conditioned final probability
- lift/delta versus prior
- fair American odds
- sportsbook price, only after freeze
- actual EV versus sportsbook price

## 2.6 Always show American odds

Every displayed probability should include fair American odds whenever a two-sided market interpretation is relevant.

Examples:

- 58.82% Under -> approximately **-143 fair**
- 41.18% Over -> approximately **+143 fair**

---

# 3. CURRENT MODEL ARCHITECTURE — v0.3

The current official research architecture is:

1. **Shared read-only baseball snapshot**
   - confirmed lineups
   - probable/starting pitchers
   - player season data
   - approved advanced features when available
   - park/environment if available

2. **Isolated run-environment capture**
   - capture only the approved full-game total point
   - prices/juice stripped
   - first observed pregame total is locked for the daily research artifact

3. **Plate-appearance event engine**
   - mutually exclusive outcomes:
     - 1B
     - 2B
     - 3B
     - HR
     - BB
     - HBP
     - K
     - ball-in-play out/residual out class

4. **Simulate I1 from batting slot #1**
   - do not assume I2 starts with the #4 hitter
   - simulate the first inning to produce the distribution of the actual I2 starting slot
   - simulate pitches thrown in I1

5. **Empirical state transitions**
   - runner advancement
   - outs added
   - post-play base occupancy
   - runs scored
   - pitch-count distribution
   - all based on historical Retrosheet plate appearances

6. **Simulate Top 2 and Bottom 2 separately**
   - use the actual batting-order sequence
   - use the opposing starter/opener/bulk mixture

7. **Generate raw baseball-only I2 distribution**

8. **Recenter on the total-conditioned historical prior** using the baseball model's log-odds delta versus the broad historical I2 reference

9. **Freeze** the conditioned probabilities

10. **Only after freeze**, retrieve derivative sportsbook I2 prices and calculate EV/staking

---

# 4. TOTAL-CONDITIONING FORMULA

Implemented in:

`src/model/i2_total_conditioning.js`

The baseball-only simulator first produces a raw full-I2 Over probability, `rawOver`.

Let:

- `broadOver` = broad historical I2 Over probability across totals 6.0–11.0
- `totalPriorOver` = historical I2 Over probability for the game's exact full-game-total bucket

The model computes the baseball matchup signal as a log-odds delta:

`baseballLogitDelta = logit(rawOver) - logit(broadOver)`

Then applies that baseball-only matchup delta to the correct total-specific prior:

`conditionedOver = logistic(logit(totalPriorOver) + baseballLogitDelta)`

Then:

`conditionedUnder = 1 - conditionedOver`

The positive-run exact distribution (`1`, `2`, `3`, `4+`) is proportionally reweighted to sum to the conditioned Over probability while `P(0)` becomes the conditioned Under probability.

Top-2 and Bottom-2 scoring probabilities are shifted by an equal logit amount so they recombine to the conditioned full-I2 probability while retaining their relative baseball-only relationship.

Audit method name in code:

`TOTAL_PRIOR_PLUS_BASEBALL_LOGIT_DELTA`

---

# 5. HISTORICAL TOTAL-CONDITIONED I2 PRIOR

Authoritative file:

`data/derived/i2/i2_total_conditioned_prior.json`

Historical benchmark window:

- 2021–2025 MLB regular season
- DraftKings pregame opening full-game total
- 10,730 matched games in totals 6.0–11.0

Broad I2 baseline:

- Over 0.5: **43.5042%**
- Under 0.5: **56.4958%**

Exact bucket table:

| Full-game total | N | I2 Over | I2 Under |
|---:|---:|---:|---:|
| 6.0 | 12 | 33.33% | 66.67% |
| 6.5 | 149 | 38.26% | 61.74% |
| 7.0 | 655 | 36.18% | 63.82% |
| 7.5 | 1,612 | 41.94% | 58.06% |
| 8.0 | 2,146 | 42.26% | 57.74% |
| 8.5 | 2,871 | 43.85% | 56.15% |
| 9.0 | 1,839 | 44.10% | 55.90% |
| 9.5 | 875 | 47.89% | 52.11% |
| 10.0 | 261 | 49.81% | 50.19% |
| 10.5 | 186 | 55.38% | 44.62% |
| 11.0 | 124 | 52.42% | 47.58% |

Important warning: the 6.0 bucket is tiny (`N=12`) and should not be treated as equally reliable to the central buckets.

The historical work established that full-game total can move the structural I2 Under prior by more than the full spread of many daily model rankings. This is why total conditioning is mandatory in v0.3.

---

# 6. WHY I1 -> I2 BATTING-ORDER STATE IS MANDATORY

I2 is not an average-lineup inning.

Historical I1->I2 state dataset:

- 12,148 regular-season games, 2021–2025
- 24,296 team-game observations
- zero missing I2 starting slots

Observed I2 starting slots include:

- #4 hitter: **8,957** observations (~36.9%)
- #5 hitter: **6,952** observations (~28.6%)
- #6 hitter: **4,246** observations (~17.5%)

Approximately 83% of I2 half-innings begin with slots #4–#6.

The same first-inning pitcher was still pitching at the start of I2 in approximately **98.04%** of historical team-game observations.

Therefore the model simulates I1 rather than assuming a fixed I2 start slot or applying whole-lineup offense indiscriminately.

---

# 7. EMPIRICAL PLAY-STATE CALIBRATION

The model has a historical state-transition calibration built from:

- **913,349 regular-season plate appearances**
- 2021–2025
- all **192 combinations** of:
  - 8 model event classes
  - 3 pre-PA out states
  - 8 base-occupancy states

For each event/out/base state it stores empirical distributions for:

- outs added
- post-play base state
- runs scored

It also stores empirical pitch-count PMFs by event.

This replaced the earliest crude fixed runner-advance assumptions in normal calibrated operation.

Key code:

`src/model/i2_inning_model.js`

Key calibration artifacts live under:

`data/derived/i2/`

---

# 8. CURRENT DATA INVENTORY

## 8.1 Joined 2021–2025 game dataset

Primary Library dataset:

`MLB_Game_Stats_Joined_2021_2025`

Key characteristics:

- 12,148 regular-season games
- inning-by-inning runs
- game metadata
- team statistics
- reconciled sportsbook opening/current game-market fields for 10,911 matched games

This is a benchmark/control dataset and source for total-conditioned historical research.

## 8.2 I2 benchmark tables

Library artifact:

`MLB_Inning_Benchmark_Tables_2021_2025.xlsx`

Contains:

- exact-run distributions
- cumulative 1+/2+/3+/4+ thresholds
- inning-by-inning probabilities
- conditioning by opening full-game total
- fair Under/Over American-odds convention

## 8.3 Retrosheet source archives

2021–2025 Retrosheet season CSV archives were used for historical state and event construction.

## 8.4 2026 team I2 scoring dataset

GitHub:

`data/derived/i2/2026_team_second_inning_runs.csv`

Fields include:

- games
- total I2 runs
- I2 runs/game
- games with at least one I2 run
- I2 scoring frequency

This dataset is currently best treated as an **independent challenger/diagnostic feature**, not automatically inserted into the production predictor.

## 8.5 Park/environment

There is partial Savant park-factor material, but complete historical/current venue coverage has been a limitation.

The daily workflow audits for an internal venue artifact such as:

`data/runtime/i2/savant_venue_profiles_2026_3yr.json`

If a complete internal artifact is missing, venue may remain neutral/missing for dimensional continuity. Do not pretend the environment is populated when it is not.

## 8.6 Advanced historical Statcast

The major missing production dataset remains a leakage-safe **daily/as-of historical Statcast feature store** for 2021–2025.

This is the main blocker for fitting batter/pitcher matchup weights rather than assuming research values.

---

# 9. CURRENT EVENT-PROBABILITY ENGINE LIMITATION

The underlying event engine still uses research/shadow interaction parameters for batter and pitcher event rates.

Historically, the engine combined batter and pitcher event probabilities in log-odds space relative to league rates with approximately equal batter/pitcher weighting unless overridden.

That framework is transparent, but the weights are **not yet validated as production-optimal**.

The most important current technical weakness is that the Monte Carlo framework is more developed than the player-level probability model feeding it.

Do not mistake 50,000 simulations for 50,000 independent sources of predictive information. More simulation trials reduce Monte Carlo error; they do not repair biased batter/pitcher event probabilities.

---

# 10. DAILY RUN WORKFLOW

Current workflow:

`.github/workflows/i2_daily_run.yml`

Current process:

1. Resolve date/cutoff/trials.
2. Audit internal venue artifact.
3. Capture the required total-conditioned run environment using:
   - `src/pipeline/fetch_i2_run_environment.mjs`
   - isolated Netlify function `netlify/functions/i2-run-environment.js`
4. Run:
   - `src/pipeline/run_i2_total_conditioned.mjs`
5. Use the user's upstream MLB data first.
6. Produce:
   - `data/runtime/i2/YYYY-MM-DD_frozen_predictions.json`
   - `data/runtime/i2/YYYY-MM-DD_run_environment.json`
7. Commit the frozen prediction/run-environment artifacts safely back to `main`.

Default Monte Carlo trials per game in the workflow: **50,000**.

The run-environment file locks the first observed approved pregame total for research continuity. `latestObservedTotal`, if present, is audit-only and does not silently replace the locked value.

---

# 11. AUGUST 16, 2026 POSTMORTEM — WHAT FAILED

A systematic postmortem was built and saved under the I2 runtime path.

Key file:

`data/runtime/i2/2026-08-16_postmortem.json`

## 11.1 Aggregate result

13 games were graded at the time of the main postmortem:

- Actual Unders: **6**
- Actual Overs: **7**
- Observed Under rate: **46.15%**
- Mean model Under probability: **55.93%**
- Model expected Unders: **7.27**
- Model Brier score: approximately **0.2584**
- Broad historical baseline Brier: approximately **0.2592**
- Model log loss: approximately **0.7100**

The aggregate 6–7 result by itself is not sufficiently unlikely to prove the model is broken. The bigger problem was **lack of discrimination and workflow contamination**.

## 11.2 Rank-discrimination failure

The actual Unders appeared at model ranks approximately:

- 1
- 6
- 7
- 8
- 12
- 14

Ranking bands:

- ranks 1–5: 1 Under in 4 graded games (~25%)
- ranks 6–10: 3 Unders in 5 games (60%)
- ranks 11–14: 2 Unders in 4 games (50%)

The mean predicted Under probability for actual Unders was essentially the same as for actual Overs.

Conclusion: **the v0.2 ranking had almost no useful discrimination on that slate**.

## 11.3 Relative rank was overstated

The top v0.2 Under probabilities were clustered very close to the broad 56.50% historical baseline.

Example from that slate:

- NYY–TOR: 58.82% Under — real lift of only about +2.32 percentage points vs broad baseline
- MIL–LAD: 57.35% — only about +0.85 pp
- SD–CLE: 56.94% — only about +0.44 pp
- SEA–HOU: 56.60% — only about +0.10 pp
- COL–SF: 56.55% — only about +0.05 pp

Calling such games “strong” merely because they ranked near the top was a reporting error.

## 11.4 Provisional-lineup workflow failure

Five graded games used provisional expected lineups that did not match the actual starting lineups:

- STL–CHC
- COL–SF
- TEX–ATH
- KC–LAA
- MIL–LAD

Four of those five went Over in the early postmortem sample.

Those games were not clean tests of a lineup-aware model.

This produced the hard rule that official rankings require confirmed lineups.

## 11.5 Starting pitchers were not the source problem

The postmortem found no starting-pitcher mismatches in the graded set.

## 11.6 I2 start-slot logic was not the obvious main failure

The postmortem added actual I2 start-slot diagnostics.

Mean probability the model assigned to the actually observed I2 start slot was approximately **29.3%** across the graded sample.

Important example: SD–CLE.

The actual I2 started at slot #4 for both teams, which was among the model's highest-probability states. San Diego then scored after the inning began with the expected middle-order region.

That failure therefore pointed more toward **event-probability calibration** than toward the I1->I2 batting-order state mechanism.

## 11.7 Full-game total was the missing structural conditioner

The largest architectural discovery from the postmortem was that the old v0.2 daily simulator was effectively ranking games around a broad I2 center while the historical research had already shown that I2 probability varies materially by full-game total.

For example:

- historical Under at total 7.0: ~63.82%
- total 8.0: ~57.74%
- total 9.5: ~52.11%
- total 10.5: ~44.62%

That variation is much larger than the entire spread of many daily v0.2 rankings.

This was not a new idea invented after the loss. It had already been finalized in the saved August 9 half-inning reference and was accidentally omitted from the new v0.2 engine — **specification drift**.

v0.3 was created to repair that drift.

---

# 12. WHAT v0.3 FIXES — AND WHAT IT DOES NOT

## v0.3 fixes

- makes the total-conditioned historical I2 prior mandatory
- blocks official projections when no exact total bucket is available
- preserves the baseball-only matchup signal as a delta rather than letting the market total directly dictate the final probability
- keeps derivative-market information isolated until freeze
- preserves the I1->I2 state simulator
- preserves empirical base/out transitions
- retains exact-run distribution outputs
- retains fair American odds

## v0.3 does NOT yet fix

- unvalidated batter/pitcher event weights
- incomplete historical as-of Statcast feature store
- full platoon/pitch-mix interaction calibration
- complete park/roof/weather coverage
- early-inning pitcher performance/fatigue modeling
- robust opener/bulk transition probabilities in every matchup
- reliable historical derivative I2 sportsbook price validation

The total-conditioned fix should be treated as a major structural correction, not proof that the model is now production-ready.

---

# 13. MOST IMPORTANT NEXT VALIDATION EXPERIMENT

Before tuning more coefficients, run a **controlled challenger comparison** over historical and live frozen samples.

At minimum compare:

1. **Broad-prior-only model**
   - constant I2 prior

2. **Total-only model**
   - exact historical I2 prior by full-game total

3. **Baseball-only model**
   - raw I1->I2 simulator without total conditioning

4. **Total-conditioned baseball model (v0.3)**
   - exact total prior + baseball logit delta

5. Optional independent challenger:
   - team 2026 I2 scoring tendencies + basic pitcher metrics

Evaluate:

- Brier score
- log loss
- calibration curve
- observed outcome by probability bucket
- rank discrimination
- Spearman rank correlation versus realized binary result only as a weak secondary diagnostic
- exact-run distribution accuracy, not just U/O 0.5
- Top2 and Bottom2 calibration separately
- performance by total bucket
- performance by lineup certainty class
- performance by starter reliability/opener status

Do not promote v0.3 to production solely because it outperforms v0.2 on August 16.

---

# 14. ADVANCED MODEL BUILD — PRIORITY ORDER

The next major structural research phase should be:

1. Materialize **daily/as-of Statcast 2021–2025**.
2. Build hitter features available before each game:
   - xwOBA/xSLG/xBA
   - barrel%
   - hard-hit%
   - exit velocity / launch-angle quality
   - chase / whiff / contact
   - handedness splits
   - pitch-type performance
3. Build starter features as-of each game:
   - xwOBA allowed
   - K/BB
   - pitch mix
   - velocity/spin/release stability
   - whiff/chase
   - contact quality
   - platoon splits
   - first-time-through-order or early-inning profile if enough data
4. Fit batter/pitcher interaction weights on rolling training windows.
5. Test 2023, 2024, and 2025 as unseen/OOS seasons.
6. Add complete park/environment only after it passes OOS validation.
7. Model I1 pitch workload as an actual I2 pitcher-state modifier rather than a diagnostic only.
8. Improve opener/bulk mixture prediction.
9. Add historical derivative I2 market-price evaluation if reliable timestamped data can be acquired.

---

# 15. CRITICAL FILE MAP

## Governing specification

- `docs/I2_MODEL.md`

## Core model

- `src/model/i2_inning_model.js`
- `src/model/i2_total_conditioning.js`
- `src/event_probability_engine.js`

## Daily pipeline

- `src/pipeline/run_i2_today.mjs`
- `src/pipeline/run_i2_today_upstream_wrapper.mjs`
- `src/pipeline/run_i2_total_conditioned.mjs`
- `src/pipeline/fetch_i2_run_environment.mjs`
- `src/pipeline/run_i2_full_slate_override.mjs` — legacy/provisional path; do not treat provisional output as official confirmed-lineup production

## Run-environment isolation

- `netlify/functions/i2-run-environment.js`

## Workflow

- `.github/workflows/i2_daily_run.yml`

## Postmortem

- `src/analysis/i2_postmortem_today.mjs`
- `.github/workflows/i2_postmortem.yml`
- `data/runtime/i2/2026-08-16_postmortem.json`

## Historical priors / derived data

- `data/derived/i2/i2_total_conditioned_prior.json`
- `data/derived/i2/2026_team_second_inning_runs.csv`
- additional I1->I2 state and play calibration artifacts under `data/derived/i2/`

## Daily runtime artifacts

- `data/runtime/i2/YYYY-MM-DD_frozen_predictions.json`
- `data/runtime/i2/YYYY-MM-DD_run_environment.json`

---

# 16. IMPORTANT REPOSITORY / WORKFLOW INCIDENT TO REMEMBER

A postmortem workflow rerun was attempted on August 17 without explicitly preserving the August 16 date input. The rerun defaulted to the current UTC date and failed because it looked for:

`data/runtime/i2/2026-08-17_frozen_predictions.json`

instead of the August 16 artifact.

Lesson:

- historical reruns must pass the target date explicitly
- do not assume rerunning a GitHub Action automatically preserves the original manual input/date

This is an execution/workflow lesson, not a predictive-model finding.

---

# 17. OUTPUT FORMAT FOR FUTURE DAILY SLATES

For every official projectable game, show at minimum:

| Rank | Game | FG Total | Historical I2 U Prior | Raw Baseball U | Final I2 U | Fair U | Final I2 O | Fair O | Away scores I2 | Home scores I2 | Delta vs total prior | Lineup status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

Then provide, for the strongest candidates:

- starting pitchers and key stats
- actual confirmed lineup
- likely I2 start-slot distribution
- slots #4–#7 hitter statistics
- Top2 and Bottom2 scoring probabilities
- exact 0/1/2/3/4+ run distribution
- model limitations active for that run

Do not call a game a “strong Under” merely because it ranks high. Use language such as:

- “highest model Under probability”
- “largest lift versus total-conditioned prior”
- “highest estimated EV after market comparison”

These are different concepts.

---

# 18. MARKET COMPARISON FORMAT

After the prediction artifact is frozen, retrieve sportsbook I2 prices and compare them against the frozen fair probabilities.

For each book/market show:

- sportsbook
- I2 Under 0.5 price
- I2 Over 0.5 price
- model probability
- fair American odds
- break-even probability
- raw probability edge
- EV
- qualification status
- stake only if the staking framework is explicitly enabled

Never use the sportsbook I2 price to revise the already frozen baseball probability.

---

# 19. THINGS THE RECEIVING AGENT MUST NOT DO

Do **not**:

- switch to public-web lineups first
- declare lineups unavailable before checking user/upstream/GitHub/Library data
- silently use expected lineups in official rankings
- feed I2 sportsbook prices into the pre-freeze prediction engine
- use moneylines/run lines to adjust baseball probabilities
- use broad I2 prior when an exact total bucket is required
- re-fit weights based on one bad slate
- claim that more Monte Carlo trials solve weak inputs
- call a #1 ranking an edge without comparing with the appropriate prior and market price
- omit American fair odds
- change model version/governance without explicit documentation
- overwrite the historical benchmark merely because a short recent sample disagrees

---

# 20. RECOMMENDED CONTINUATION PROMPT FOR A NEW CHAT

Paste or provide this transfer package to the new chat, then use the following instruction:

> Continue the MLB I2 Under/Over project from the attached transfer package and the connected GitHub repository `shanecolter1/mlb-model-v6`. Treat `docs/I2_MODEL.md` as the governing specification. Use my upstream MLB data, GitHub, and Library before public web sources. Do not change the model or workflow silently. Preserve the total-conditioned v0.3 architecture and confirmed-lineup rule. First inspect the current repository head and report any differences from the transfer-package head. Then continue from the highest-priority unfinished validation task rather than rebuilding anything already completed.

---

# 21. CURRENT STATUS IN ONE PARAGRAPH

The project has a solid simulation/state framework: actual batting-order sequencing, I1->I2 start-slot simulation, empirical base/out/run transitions, exact-run distributions, user-data-first retrieval, frozen outputs, postmortem tooling, and now mandatory total-conditioned I2 priors. The August 16 failure showed that v0.2 rankings had weak discrimination, some official-looking results were contaminated by provisional lineups, and the new engine had drifted away from the previously finalized rule that full-game total materially conditions I2 scoring. v0.3 repairs that structural prior problem with a narrow, audited total-only market exception and a logit-delta conditioning method. The model is still **research-only** because the batter/pitcher event-probability layer has not yet been fitted with leakage-safe historical as-of Statcast features and has not passed robust rolling out-of-sample validation.

---

# 22. PACKAGE CHECKSUM / SNAPSHOT NOTE

This transfer package describes the project at repository head:

`1acb32d12d2a58bf2d9916b032e2182ca85cd715`

If the repository head is newer, inspect the intervening commits before assuming this package contains every subsequent change.

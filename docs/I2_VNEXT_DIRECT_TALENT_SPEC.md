# MLB I2 vNext — Direct I2 Talent Model Lock

Status: **SHADOW BUILD / NOT PRODUCTION-PROMOTED**

## Objective
Predict the probability that the **full second inning** is scoreless using baseball information only. No sportsbook, full-game total, moneyline, run line, I2 price, consensus, or market movement may enter before the prediction artifact is frozen.

## Core statistical object
Estimate **hitter I2 talent** and **pitcher I2 talent directly from second-inning plate appearances**. Broader baseball information may support sparse matchup information, but it may not be added as a second independent player-talent layer.

The lean PA model is one jointly regularized multinomial model with:
- batter MLBAM identity (direct I2 effect)
- pitcher MLBAM identity (direct I2 effect allowed)
- batter/pitcher handedness interaction
- one pitcher-arsenal × batter pitch-type response interaction, specific to the batter side when I2 pitch-mix support exists
- home-team/park identity as a fit-only nuisance control so player effects do not absorb park signal

No fixed batter/pitcher blend is permitted.

## Arsenal feature
For historical PA fitting, season Y uses season Y-1 Baseball Savant pitch-arsenal profiles to prevent future leakage. For a live prediction, use the current YTD Savant arsenal snapshot available at the prediction cutoff and archive that snapshot.

The matchup scalar is the pitcher's **I2 pitch-usage distribution versus the actual batter side** weighted by the batter's Savant expected wOBA against those pitch types. If pitcher-side I2 usage is unavailable, fall back to the pitcher's overall Savant arsenal, then league-side/league overall usage. Missing batter pitch-type cells fall back inside the feature calculation to the league pitch-type expectation. This is one matchup interaction, not another talent or probability-adjustment layer.

## Recency and regularization
Do not guess recent/current/prior-season weights.
- Fit exponential recency half-life from a bounded chronological fold that trains through June 30, 2024 and tests the remainder of 2024, so both prior-season and current-season evidence are present when recency is selected.
- Fit the model's L2 regularization strength on that same chronological fold.
- Player effects are regularized jointly inside the single PA model.

Do not apply independent hitter shrinkage, pitcher shrinkage, empirical blending, confidence multipliers, or market conditioning after this fit.

## Final probability calibration
PA probabilities are not post-calibrated separately. After the complete pregame I2 model is replayed chronologically, fit **one final full-model calibration/shrinkage curve** to the full-I2 Under 0.5 probability. Calibration training predictions must be out-of-sample relative to the baseball model fit.

Candidate final calibrators are restricted to:
1. no calibration;
2. a single two-parameter sigmoid/logit calibration.

The sigmoid is selected only if it improves both Brier score and log loss on the later chronological 2025 validation segment; otherwise the identity curve is retained.

## Existing components to reuse
Reuse rather than rebuild:
- lineup/source resolver;
- I1 -> I2 batting-position engine;
- batting-order simulation;
- Retrosheet empirical base/out/run transitions;
- exact half-inning/full-inning state engine;
- Baseball Savant park/environment source;
- market-isolation/freeze workflow;
- input and model audit conventions.

## Park/environment rule
Home-team/park identity is included in fitting only as a nuisance control and is intentionally omitted from neutral live talent inference. Then apply the Savant event park effect exactly once. When handedness-specific absolute park factors exist, use the handedness-specific factor rather than multiplying it by the all-batters absolute factor. Missing venue coverage must be explicit in the audit.

## Opener/bulk handling
- Normal starter: no special opener treatment.
- Opener identification is a workflow/source task.
- Opener survival to I2 may be estimated statistically from historical opener usage.
- Expected follower identity is sourced pregame when known.
- Materially uncertain nonstandard pitching plans are flagged rather than filled with an invented bullpen mixture.

## Initial exclusions
Do not add these unless they later prove incremental out-of-sample value:
- team recent I2 scoring rates;
- raw pitcher scoreless-I2 percentage;
- BvP;
- prior pitches seen by the same hitter;
- generic hot streaks;
- series game number;
- travel;
- catcher/umpire/defense adjustments;
- live pitch-count state;
- full-game market total or any other sportsbook input.

## Historical data
Primary direct-talent dataset: Baseball Savant Statcast **inning 2 only**, regular season, terminal PA outcomes, MLBAM IDs, handedness, pitch type, and home-team park identity. Raw I2 pitches are also aggregated by pitcher × actual batter side × pitch type to build side-specific I2 arsenal usage. Use monthly chunks to keep retrieval bounded.

PA event taxonomy remains compatible with the existing event simulator:
- single
- double
- triple
- home_run
- walk
- hit_by_pitch
- strikeout
- ball_in_play_out

Rare ROE/interference outcomes are mapped explicitly and audited rather than silently dropped as outs.

## Validation
Hyperparameters must be selected chronologically, not by random cross-validation. Primary PA-model metrics are multiclass log loss and multiclass Brier score. The promotion metric is the full-I2 Under 0.5 forecast using Brier, log loss, and reliability/calibration.

The baseball model is never optimized on historical betting profit.

## Promotion
This branch remains shadow-only until:
- the full-model historical replay exists;
- the final calibration curve is estimated from OOS predictions;
- no material leakage is found;
- the live runner uses exactly one canonical model artifact;
- the shadow model demonstrates acceptable calibration and predictive performance against simple and current-model benchmarks.

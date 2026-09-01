# All-Inning Empirical Matchup Variable Selection

Status: architecture reset before final matchup-distribution testing.

## Purpose
Select batter, pitcher, platoon, workload, and interaction statistics empirically before constructing the final I1-I9 matchup distribution. No variable receives production status because it is conventional or intuitively appealing.

## Governance
- Development seasons: 2021-2024 only.
- Final untouched holdout: 2025.
- Same-day history excluded.
- No inning-market data or betting-derived variables.
- Locked M0 opening-total x inning baseline remains champion anchor.
- 1Tap is unchanged.
- Existing M1 K/baserunner/HR/non-HR-hit feature family is now provisional until it survives this broader variable-selection stage.
- Existing arbitrary shrink strength=50 in historical platoon builder is research-only and must not be treated as validated. Shrinkage/reliability values are selected empirically.

## Candidate variable families
### Batter
Candidate variables should include all leakage-safe statistics available from the historical PA layer and any separately reconstructed Statcast layer that can be made strictly as-of, including at minimum:
- strikeout rate
- walk rate
- HBP rate
- single, double, triple, HR rates
- hit rate
- XBH rate
- on-base rate
- contact rate
- non-HR hit rate
- PA volume / sample reliability

If strict-as-of Statcast reconstruction is available, test rather than assume value from:
- xwOBA / xBA / xSLG
- barrel rate
- hard-hit rate
- exit velocity / max exit velocity
- launch-angle / sweet-spot measures
- chase / whiff / contact measures

### Pitcher
Candidate variables should include all leakage-safe PA-derived statistics and, where available with strict as-of timing, Statcast measures:
- K rate
- BB rate / HBP rate
- HR rate
- single/double/triple/non-HR-hit rates allowed
- on-base allowed / contact allowed / XBH allowed
- batters faced and sample reliability
- starter workload / rest / retention variables only when relevant to pitcher-state rather than double-counting M2

Statcast candidates, if reconstructed safely:
- xERA / xwOBA allowed
- K/whiff/chase/contact measures
- barrel / hard-hit allowed
- velocity, spin, pitch mix, release characteristics
- pitch-level quality measures where consistent historically

### Platoon
Platoon must be tested as incremental information over unsplit batter and pitcher skill, not presumed useful.
For each candidate batter/pitcher statistic test:
- overall rate
- vs L / vs R split rate
- platoon delta = split rate - overall rate
- handedness indicator alone
- batter-side x pitcher-hand interaction
- switch-hitter treatment if sample supports it

Candidate lookback windows and shrinkage are tuned only inside development data. Existing 90d/365d windows and shrink=50 are candidates, not defaults.

### Matchup interactions
For variables that survive marginal testing, evaluate incremental pitcher x batter interaction structures including:
- additive batter + pitcher
- multiplicative interaction
- difference / contrast
- platoon-conditioned interaction
- nonlinear transformation only if it improves chronological out-of-sample performance

## Selection hierarchy
Variables are selected in stages to avoid high-dimensional overfit.

1. **Marginal predictive screen**
   Test each variable/family alone against appropriate PA-event outcomes and against the locked M0 run outcome where applicable. Measure chronological out-of-sample log loss/Brier for binary events and multinomial log loss for event classes.

2. **Incremental family tests**
   Add batter family to baseline; pitcher family to baseline; platoon on top of batter+pitcher; Statcast on top of PA-history; workload only in pitcher-state layer. A family must add stable OOS value rather than merely show univariate correlation.

3. **Redundancy / collinearity control**
   Within highly correlated groups, retain the simpler or more stable feature unless the additional feature adds OOS value. Regularization strength is selected chronologically, not hand-set.

4. **Window and reliability selection**
   Candidate season/30d/60d/90d/180d/365d windows may be tested when data support them. Shrinkage/reliability strength is selected empirically by development-fold scoring. No hard-coded 50 pseudo-count is production-approved.

5. **Platoon promotion rule**
   Platoon split information is promoted only if it improves OOS performance over the corresponding overall player skill after accounting for sample size/reliability. Batter platoon and pitcher platoon are tested separately and jointly.

6. **Common family, inning-varying weight**
   First determine a common empirically supported matchup feature family using PA-level evidence. Then test whether weights should be pooled, partially pooled, or inning-specific for I1-I9. Feature membership should not vary by inning unless the data provide strong stable evidence.

7. **Point vs distribution**
   After feature selection is frozen on 2021-2024, test the same selected feature family in both:
   - M4-P point/expected matchup vector
   - M4-D full latent matchup distribution
   Distribution means passing each pitcher x batter/path state through the nonlinear event/scoring function before probability-weighting outputs.

8. **Final run-distribution stage**
   Only after variable family and state representation are frozen do we estimate P(R_i=0,1,2,3,4+) for each I1-I9 and open 2025 once.

## Required reporting
For every candidate/family report:
- source and exact as-of construction
- coverage
- candidate windows / shrinkage
- 2022, 2023, 2024 fold metrics
- mean and worst-year improvement
- selected / rejected status
- incremental value conditional on already-selected features
- stability/sign consistency
- platoon incremental value separately
- final common feature family before M4-P vs M4-D comparison

## Immediate consequence
Do not interpret the existing K/baserunner/HR/non-HR-hit M1 family as the final matchup statistic set. It was a useful initial signal-discovery basis, but the final matchup model must be rebuilt from the empirically selected feature inventory described here before the point-vs-distribution conclusion is accepted.

# Phase 1.3 implementation checklist

- [x] Define mutually exclusive plate-appearance event schema.
- [x] Add neutral batter/pitcher event-combination interface.
- [x] Add event-specific venue adjustment and normalization.
- [x] Add lineup plate-appearance weighting.
- [x] Add preliminary run-expectancy and base/out simulation bridge.
- [x] Add three-arm shadow runner.
- [x] Enforce that the event-vector arm cannot also apply run factor.
- [x] Keep BB/SO venue effects off pending validation.
- [x] Add unit-test scaffolding.
- [ ] Connect actual batter, starter and bullpen event-rate feeds.
- [ ] Replace illustrative advancement rules with fitted play-by-play rules.
- [ ] Add deterministic seeded simulation.
- [ ] Add historical snapshot storage by game and prediction time.
- [ ] Run rolling-origin backtest.
- [ ] Integrate outputs into moneyline and margin engines.
- [ ] Promote only after validation gates pass.

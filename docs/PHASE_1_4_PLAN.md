# Phase 1.4 — Historical input and rolling-backtest plan

## Completed in this scaffold

- Leakage-safe pregame snapshot schema
- Batter and pitcher trailing event-rate builder
- Empirical-Bayes shrinkage for limited samples
- Starter/bullpen event-rate blending
- Lineup plate-appearance share normalization
- Immutable JSON snapshot storage
- Future-information leakage guard
- Seeded simulations
- Rolling-origin fold generator
- MAE, RMSE, Brier score and log-loss evaluation
- Post-game outcome attachment kept outside prediction code

## Required external data before running a real backtest

1. Historical plate-appearance event records
2. Historical lineups and batting order
3. Announced starting pitchers
4. Bullpen usage and availability
5. Venue-profile snapshots available as of each prediction time
6. Final game outcomes

## Non-negotiable time rule

For a game predicted at time T, every feature must have an `as_of` timestamp
less than or equal to T. Final outcomes are merged only after the prediction
file is frozen.

## First real backtest sequence

1. Build daily trailing batter and pitcher event-rate snapshots.
2. Construct one immutable pregame snapshot per game and prediction checkpoint.
3. Generate the three venue arms with identical non-venue inputs.
4. Store predictions before attaching results.
5. Attach final outcomes in a separate process.
6. Evaluate each rolling-origin test fold.
7. Compare aggregate and park-specific calibration.
8. Reject any arm with leakage, material instability, or hidden double counting.

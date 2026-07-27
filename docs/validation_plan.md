# Phase 1 shadow validation plan

## Comparison arms
1. Legacy generic `park_score`
2. Savant run factor only
3. Savant event-vector profile

## Time-series design
- Train/tune only on prior dates.
- Evaluate on later, unseen dates.
- Archive the exact profile snapshot used before each game.
- Never use end-of-season values for an earlier prediction date.

## Primary metrics
- Team expected-runs MAE and RMSE
- Game-total MAE and calibration
- Moneyline Brier score and log loss
- Win-by-2 and win-by-3 Brier score
- Home-run probability Brier score and reliability curves

## Promotion rule
The event-vector profile replaces the legacy input only when it:
- improves at least one primary target materially;
- does not materially worsen the others;
- remains stable across seasons and venue groups;
- has no unresolved leakage or double-counting audit failures.

## Required audits
- Every valid Savant profile disables `park_score`.
- Weather remains a separate input.
- Run factor is applied once.
- HR/1B/2B/3B factors alter event rates, not an additional total-run additive term.
- New/temporary venues are flagged and shrunk or fall back.

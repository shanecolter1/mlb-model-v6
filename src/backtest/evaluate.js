import {
  brierScore, logLoss, meanAbsoluteError, rootMeanSquaredError
} from "./metrics.js";

export function evaluatePredictionRows(rows) {
  if (!rows.length) throw new RangeError("No prediction rows supplied");

  const byArm = new Map();
  for (const row of rows) {
    if (!byArm.has(row.model_arm)) byArm.set(row.model_arm, []);
    byArm.get(row.model_arm).push(row);
  }

  const results = {};
  for (const [arm, armRows] of byArm.entries()) {
    const predictedRuns = armRows.map((r) => r.predicted_total_runs);
    const actualRuns = armRows.map((r) => r.actual_total_runs);
    const winProb = armRows.map((r) => r.home_win_probability);
    const homeWin = armRows.map((r) => r.home_win_outcome);

    results[arm] = {
      games: armRows.length,
      total_runs_mae: meanAbsoluteError(predictedRuns, actualRuns),
      total_runs_rmse: rootMeanSquaredError(predictedRuns, actualRuns),
      moneyline_brier: brierScore(winProb, homeWin),
      moneyline_log_loss: logLoss(winProb, homeWin),
    };
  }
  return results;
}

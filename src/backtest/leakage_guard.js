export function assertPregameSnapshot(snapshot) {
  const predictionTime = new Date(snapshot.prediction_timestamp);
  if (Number.isNaN(predictionTime.valueOf())) {
    throw new RangeError("Invalid prediction timestamp");
  }

  const violations = [];
  const forbiddenTopLevel = [
    "final_home_runs", "final_away_runs", "closing_result",
    "postgame_stats", "observed_result"
  ];

  for (const key of forbiddenTopLevel) {
    if (snapshot[key] != null) violations.push(`Forbidden pregame field: ${key}`);
  }

  const checkAsOf = (obj, label) => {
    if (!obj || !obj.as_of) return;
    const asOf = new Date(obj.as_of);
    if (asOf > predictionTime) {
      violations.push(`${label}.as_of occurs after prediction timestamp`);
    }
  };

  checkAsOf(snapshot.environmental_context, "environmental_context");
  for (const side of ["home", "away"]) {
    const team = snapshot.team_inputs?.[side];
    checkAsOf(team, `team_inputs.${side}`);
    for (const batter of team?.lineup ?? []) {
      checkAsOf(batter.event_rates, `team_inputs.${side}.lineup.${batter.player_id}`);
    }
  }

  if (violations.length) {
    const error = new Error(`Leakage guard failed:\n${violations.join("\n")}`);
    error.violations = violations;
    throw error;
  }
  return true;
}

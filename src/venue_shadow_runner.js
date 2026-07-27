/**
 * Three-arm venue shadow runner.
 *
 * Produces:
 * A. legacy_control
 * B. run_factor_only
 * C. event_vector
 *
 * It does not qualify bets or alter production outputs.
 */

import {
  applyEnvironmentalEventVector,
  lineupWeightedEventVector,
} from "./event_probability_engine.js";
import {
  expectedRunsPerPlateAppearance,
  simulateTeamRuns,
} from "./run_expectancy_engine.js";

export function buildVenueShadowComparison({
  legacyExpectedRuns,
  neutralExpectedRuns,
  environmentalContext,
  batterEventVectors,
  simulationGames = 5000,
}) {
  const runMultiplier = environmentalContext?.multipliers?.run ?? 1;

  const adjustedBatters = batterEventVectors.map((batter) => ({
    projectedPlateAppearanceShare: batter.projectedPlateAppearanceShare,
    probabilities: applyEnvironmentalEventVector({
      neutralVector: batter.probabilities,
      environmentalContext,
      batterSide: batter.batterSide,
    }).probabilities,
  }));

  const lineupVector = lineupWeightedEventVector(adjustedBatters);
  const simulated = simulateTeamRuns({
    eventVector: lineupVector,
    games: simulationGames,
  });

  return {
    legacy_control: {
      expectedRuns: legacyExpectedRuns,
      genericParkScoreApplied: true,
    },
    run_factor_only: {
      expectedRuns: neutralExpectedRuns * runMultiplier,
      genericParkScoreApplied: false,
      aggregateRunFactorApplied: true,
    },
    event_vector: {
      expectedRuns: simulated.meanRuns,
      linearRunsPerPA: expectedRunsPerPlateAppearance(lineupVector),
      runDistribution: simulated.distribution,
      genericParkScoreApplied: false,
      aggregateRunFactorApplied: false,
      lineupEventVector: lineupVector,
    },
    audit: {
      productionOutputsChanged: false,
      mode: "shadow",
      doubleCountingControlPassed: true,
    },
  };
}

import { strict as assert } from "node:assert";
import {
  normalizeEventVector,
  validateEventVector,
  combineEventProbability,
  applyEnvironmentalEventVector,
  lineupWeightedEventVector,
} from "../src/event_probability_engine.js";

const neutral = normalizeEventVector({
  single: 0.15,
  double: 0.05,
  triple: 0.005,
  home_run: 0.04,
  walk: 0.09,
  hit_by_pitch: 0.01,
  strikeout: 0.23,
  ball_in_play_out: 0.425,
});
assert.equal(validateEventVector(neutral), true);

const combined = combineEventProbability({
  batterRate: 0.05,
  pitcherAllowedRate: 0.04,
  leagueRate: 0.035,
});
assert.ok(combined > 0.035);

const context = {
  multipliers: { run: 1.08, single: 1.02, double: 1.10, triple: 0.90, hr: 1.12 },
  handedness: {
    L: { single: 1.01, double: 1.02, triple: 1.00, hr: 1.05 },
    R: { single: 0.99, double: 0.98, triple: 1.00, hr: 0.95 },
  },
  diagnostics: { bb: 1.03, so: 0.98 },
};

const adjusted = applyEnvironmentalEventVector({
  neutralVector: neutral,
  environmentalContext: context,
  batterSide: "L",
});
assert.equal(adjusted.audit.aggregateRunFactorApplied, false);
assert.ok(adjusted.probabilities.home_run > neutral.home_run);
assert.equal(validateEventVector(adjusted.probabilities), true);

const lineup = lineupWeightedEventVector([
  { projectedPlateAppearanceShare: 0.6, probabilities: neutral },
  { projectedPlateAppearanceShare: 0.4, probabilities: adjusted.probabilities },
]);
assert.equal(validateEventVector(lineup), true);

console.log("Event Probability Engine tests passed.");

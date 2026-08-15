import assert from "node:assert/strict";
import { createSeededRandom } from "../src/model/seeded_random.js";
import {
  fairAmericanOdds,
  simulateFullSecondInning,
} from "../src/model/i2_inning_model.js";

function vector() {
  return {
    single: 0.14,
    double: 0.05,
    triple: 0.005,
    home_run: 0.035,
    walk: 0.09,
    hit_by_pitch: 0.01,
    strikeout: 0.24,
    ball_in_play_out: 0.43,
  };
}

const lineup = Array.from({ length: 9 }, () => ({ side: "R", eventRates: vector() }));
const pitcher = { eventRatesAllowed: vector() };
const league = vector();
const random = createSeededRandom(20260815);

const result = simulateFullSecondInning({
  away: { lineup, starter: pitcher },
  home: { lineup, starter: pitcher },
  league,
  trials: 5000,
  random,
});

const exactSum = Object.values(result.fullI2.exact).reduce((a, b) => a + b, 0);
assert.ok(Math.abs(exactSum - 1) < 1e-12);
assert.ok(result.under05 > 0 && result.under05 < 1);
assert.equal(result.governance.marketDataUsed, false);
assert.equal(result.governance.predictionMustBeFrozenBeforeMarketRetrieval, true);
assert.equal(Math.round(fairAmericanOdds(0.6)), -150);
assert.equal(Math.round(fairAmericanOdds(0.4)), 150);

const awaySlotSum = Object.values(result.stateDiagnostics.awayI2StartSlotProbability)
  .reduce((a, b) => a + b, 0);
assert.ok(Math.abs(awaySlotSum - 1) < 1e-12);

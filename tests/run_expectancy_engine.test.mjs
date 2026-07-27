import { strict as assert } from "node:assert";
import { normalizeEventVector } from "../src/event_probability_engine.js";
import {
  expectedRunsPerPlateAppearance,
  simulateHalfInning,
  simulateTeamRuns,
} from "../src/run_expectancy_engine.js";

const vector = normalizeEventVector({
  single: 0.15,
  double: 0.05,
  triple: 0.005,
  home_run: 0.04,
  walk: 0.09,
  hit_by_pitch: 0.01,
  strikeout: 0.23,
  ball_in_play_out: 0.425,
});

assert.ok(expectedRunsPerPlateAppearance(vector) > 0);
const deterministicOut = () => 0.999999;
assert.equal(simulateHalfInning(vector, deterministicOut), 0);

const result = simulateTeamRuns({ eventVector: vector, games: 100, random: Math.random });
assert.ok(result.meanRuns >= 0);
assert.ok(Object.keys(result.distribution).length > 0);

console.log("Run Expectancy Engine tests passed.");

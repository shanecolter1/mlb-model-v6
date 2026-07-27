import { strict as assert } from "node:assert";
import {
  resolveEnvironmentalContext,
  adjustExpectedRuns,
  adjustEventRate,
} from "../src/environmental_context_adapter.js";

const profile = {
  source: "Baseball Savant Statcast Park Factors",
  window: "2024-2026",
  plate_appearances: 49000,
  confidence: 0.98,
  multipliers: { run: 1.06, hr: 1.12, single: 0.98, double: 1.08, triple: 0.96 },
  handedness: {},
};

const c = resolveEnvironmentalContext({
  venueProfile: profile,
  legacyParkScore: 0.2,
  weatherScore: 0.1,
});

assert.equal(c.genericParkScoreEnabled, false);
assert.equal(c.legacyParkScore, null);
assert.equal(adjustExpectedRuns(8.5, c), 9.01);
assert.equal(adjustEventRate(0.03, "hr", c), 0.033600000000000005);

const fallback = resolveEnvironmentalContext({
  venueProfile: null,
  legacyParkScore: 0.2,
  weatherScore: 0.1,
});
assert.equal(fallback.genericParkScoreEnabled, true);
console.log("Environmental Context tests passed.");

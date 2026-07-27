import { strict as assert } from "node:assert";
import { assertPregameSnapshot } from "../src/backtest/leakage_guard.js";

const good = {
  prediction_timestamp: "2026-07-25T20:00:00Z",
  environmental_context: { as_of: "2026-07-25T19:00:00Z" },
  team_inputs: { home: { lineup: [] }, away: { lineup: [] } },
};
assert.equal(assertPregameSnapshot(good), true);

const bad = {
  ...good,
  environmental_context: { as_of: "2026-07-26T00:00:00Z" },
};
assert.throws(() => assertPregameSnapshot(bad));
console.log("Leakage guard tests passed.");

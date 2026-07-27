import { strict as assert } from "node:assert";
import {
  brierScore, logLoss, meanAbsoluteError, rootMeanSquaredError
} from "../src/backtest/metrics.js";

assert.equal(meanAbsoluteError([1,2],[2,2]), 0.5);
assert.ok(rootMeanSquaredError([1,2],[2,2]) > 0);
assert.ok(brierScore([0.8,0.3],[1,0]) < 0.1);
assert.ok(logLoss([0.8,0.3],[1,0]) > 0);
console.log("Backtest metrics tests passed.");

import { strict as assert } from "node:assert";
import { blendPitchingStaffRates } from "../src/model/pitcher_staff_blend.js";
import { validateEventVector } from "../src/event_probability_engine.js";

const starter = {
  single: 0.15, double: 0.05, triple: 0.005, home_run: 0.04,
  walk: 0.09, hit_by_pitch: 0.01, strikeout: 0.25, ball_in_play_out: 0.405
};
const bullpen = {
  single: 0.16, double: 0.05, triple: 0.005, home_run: 0.045,
  walk: 0.10, hit_by_pitch: 0.01, strikeout: 0.24, ball_in_play_out: 0.39
};
const blend = blendPitchingStaffRates({
  starterAllowedRates: starter,
  bullpenAllowedRates: bullpen,
  starterExpectedBattersFaced: 22,
  bullpenExpectedBattersFaced: 16,
});
assert.equal(validateEventVector(blend), true);
assert.ok(blend.home_run > starter.home_run && blend.home_run < bullpen.home_run);
console.log("Pitching staff blend tests passed.");

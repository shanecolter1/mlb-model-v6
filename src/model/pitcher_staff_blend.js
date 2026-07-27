import { normalizeEventVector, EVENT_VECTOR_KEYS } from "../event_probability_engine.js";

export function blendPitchingStaffRates({
  starterAllowedRates,
  bullpenAllowedRates,
  starterExpectedBattersFaced,
  bullpenExpectedBattersFaced,
}) {
  const starterBF = Math.max(0, starterExpectedBattersFaced ?? 0);
  const bullpenBF = Math.max(0, bullpenExpectedBattersFaced ?? 0);
  const total = starterBF + bullpenBF;
  if (total <= 0) throw new RangeError("Expected batters faced must sum above zero");

  const blended = {};
  for (const event of EVENT_VECTOR_KEYS) {
    blended[event] =
      ((starterAllowedRates[event] ?? 0) * starterBF +
       (bullpenAllowedRates[event] ?? 0) * bullpenBF) / total;
  }
  return normalizeEventVector(blended);
}

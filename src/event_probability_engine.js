/**
 * Version 6 Event Probability Engine — Phase 1.3
 *
 * Predicts mutually exclusive plate-appearance outcomes before converting them
 * into runs or game outcomes.
 *
 * Critical anti-double-counting rule:
 * - `run_factor_only` applies the Savant run multiplier to aggregate expected runs.
 * - `event_vector` applies 1B/2B/3B/HR multipliers to event probabilities.
 * - `event_vector` does NOT also apply the Savant run multiplier.
 */

const EVENT_KEYS = [
  "single",
  "double",
  "triple",
  "home_run",
  "walk",
  "hit_by_pitch",
  "strikeout",
  "ball_in_play_out",
];

const EPSILON = 1e-12;

function finiteNonNegative(value, name) {
  if (!Number.isFinite(value) || value < 0) {
    throw new TypeError(`${name} must be a finite non-negative number`);
  }
  return value;
}

export function validateEventVector(vector, tolerance = 1e-9) {
  for (const key of EVENT_KEYS) {
    finiteNonNegative(vector[key], key);
  }
  const total = EVENT_KEYS.reduce((sum, key) => sum + vector[key], 0);
  if (Math.abs(total - 1) > tolerance) {
    throw new RangeError(`Event probabilities must sum to 1; received ${total}`);
  }
  return true;
}

export function normalizeEventVector(raw) {
  const cleaned = {};
  for (const key of EVENT_KEYS) {
    cleaned[key] = finiteNonNegative(raw[key] ?? 0, key);
  }
  const total = EVENT_KEYS.reduce((sum, key) => sum + cleaned[key], 0);
  if (total <= EPSILON) {
    throw new RangeError("Cannot normalize an empty event vector");
  }
  const normalized = Object.fromEntries(
    EVENT_KEYS.map((key) => [key, cleaned[key] / total])
  );
  validateEventVector(normalized);
  return normalized;
}

/**
 * Combines batter and pitcher event rates in log-odds space relative to a
 * league baseline. This is a transparent initial interaction rule for shadow
 * mode; coefficients should later be fitted on time-separated training data.
 */
export function combineEventProbability({
  batterRate,
  pitcherAllowedRate,
  leagueRate,
  batterWeight = 0.5,
  pitcherWeight = 0.5,
}) {
  for (const [name, value] of Object.entries({
    batterRate,
    pitcherAllowedRate,
    leagueRate,
  })) {
    if (!Number.isFinite(value) || value <= 0 || value >= 1) {
      throw new RangeError(`${name} must be strictly between 0 and 1`);
    }
  }
  if (batterWeight < 0 || pitcherWeight < 0 || batterWeight + pitcherWeight <= 0) {
    throw new RangeError("Interaction weights must be non-negative and non-zero");
  }

  const logit = (p) => Math.log(p / (1 - p));
  const logistic = (x) => 1 / (1 + Math.exp(-x));
  const weightTotal = batterWeight + pitcherWeight;

  const batterDelta = logit(batterRate) - logit(leagueRate);
  const pitcherDelta = logit(pitcherAllowedRate) - logit(leagueRate);
  const combinedLogit =
    logit(leagueRate) +
    (batterWeight / weightTotal) * batterDelta +
    (pitcherWeight / weightTotal) * pitcherDelta;

  return logistic(combinedLogit);
}

export function buildNeutralEventVector({
  batter,
  pitcher,
  league,
  weights = { batter: 0.5, pitcher: 0.5 },
}) {
  const modeled = {};
  for (const key of EVENT_KEYS.filter((k) => k !== "ball_in_play_out")) {
    modeled[key] = combineEventProbability({
      batterRate: batter[key],
      pitcherAllowedRate: pitcher[key],
      leagueRate: league[key],
      batterWeight: weights.batter,
      pitcherWeight: weights.pitcher,
    });
  }

  const subtotal = Object.values(modeled).reduce((a, b) => a + b, 0);
  if (subtotal >= 1) {
    // Preserve relative modeled-event odds and leave a small valid residual.
    const scale = 0.999 / subtotal;
    for (const key of Object.keys(modeled)) modeled[key] *= scale;
  }
  modeled.ball_in_play_out =
    1 - Object.values(modeled).reduce((a, b) => a + b, 0);

  return normalizeEventVector(modeled);
}

function sideFactor(context, batterSide, eventName) {
  const side = batterSide === "L" ? "L" : "R";
  const splitName =
    eventName === "home_run" ? "hr" :
    eventName === "single" ? "single" :
    eventName === "double" ? "double" :
    eventName === "triple" ? "triple" :
    null;
  if (!splitName) return 1;
  return context?.handedness?.[side]?.[splitName] ?? 1;
}

/**
 * Applies event-specific venue factors. The aggregate run factor is purposely
 * excluded from this arm.
 */
export function applyEnvironmentalEventVector({
  neutralVector,
  environmentalContext,
  batterSide,
  enableCandidateDisciplineFactors = false,
}) {
  validateEventVector(neutralVector);

  const factors = {
    single:
      (environmentalContext?.multipliers?.single ?? 1) *
      sideFactor(environmentalContext, batterSide, "single"),
    double:
      (environmentalContext?.multipliers?.double ?? 1) *
      sideFactor(environmentalContext, batterSide, "double"),
    triple:
      (environmentalContext?.multipliers?.triple ?? 1) *
      sideFactor(environmentalContext, batterSide, "triple"),
    home_run:
      (environmentalContext?.multipliers?.hr ?? 1) *
      sideFactor(environmentalContext, batterSide, "home_run"),
    walk: enableCandidateDisciplineFactors
      ? environmentalContext?.diagnostics?.bb ?? 1
      : 1,
    strikeout: enableCandidateDisciplineFactors
      ? environmentalContext?.diagnostics?.so ?? 1
      : 1,
    hit_by_pitch: 1,
    ball_in_play_out: 1,
  };

  const adjusted = Object.fromEntries(
    EVENT_KEYS.map((key) => [key, neutralVector[key] * factors[key]])
  );

  // Renormalization is mandatory because event outcomes are mutually exclusive.
  const normalized = normalizeEventVector(adjusted);

  return {
    probabilities: normalized,
    audit: {
      modelArm: "event_vector",
      aggregateRunFactorApplied: false,
      candidateDisciplineFactorsApplied: enableCandidateDisciplineFactors,
      factors,
    },
  };
}

export function lineupWeightedEventVector(batterVectors) {
  if (!Array.isArray(batterVectors) || batterVectors.length === 0) {
    throw new RangeError("At least one batter vector is required");
  }

  let totalWeight = 0;
  const totals = Object.fromEntries(EVENT_KEYS.map((key) => [key, 0]));

  for (const item of batterVectors) {
    const weight = finiteNonNegative(item.projectedPlateAppearanceShare, "projectedPlateAppearanceShare");
    validateEventVector(item.probabilities);
    totalWeight += weight;
    for (const key of EVENT_KEYS) totals[key] += item.probabilities[key] * weight;
  }
  if (totalWeight <= EPSILON) throw new RangeError("Lineup weights must sum above zero");

  return normalizeEventVector(
    Object.fromEntries(EVENT_KEYS.map((key) => [key, totals[key] / totalWeight]))
  );
}

export const EVENT_VECTOR_KEYS = Object.freeze([...EVENT_KEYS]);

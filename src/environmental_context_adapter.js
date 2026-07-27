/**
 * Environmental Context adapter for the MLB prediction engine.
 *
 * Shadow-mode rule:
 * - A valid Savant profile disables legacy park_score.
 * - Weather remains separate.
 * - Event multipliers are routed to their specific engines.
 * - The run factor is applied once, not added again through component factors.
 */

export function resolveEnvironmentalContext({
  venueProfile,
  legacyParkScore = null,
  weatherScore = null,
}) {
  const valid = Boolean(
    venueProfile &&
    venueProfile.multipliers &&
    Number.isFinite(venueProfile.multipliers.run)
  );

  if (!valid) {
    return {
      mode: "fallback",
      genericParkScoreEnabled: true,
      legacyParkScore,
      weatherScore,
      confidence: 0.45,
      multipliers: {
        run: 1,
        hr: 1,
        single: 1,
        double: 1,
        triple: 1,
      },
      audit: { reason: "Missing or unreconciled Savant venue profile" },
    };
  }

  return {
    mode: "shadow",
    genericParkScoreEnabled: false,
    legacyParkScore: null,
    weatherScore,
    confidence: venueProfile.confidence,
    multipliers: {
      run: venueProfile.multipliers.run ?? 1,
      hr: venueProfile.multipliers.hr ?? 1,
      single: venueProfile.multipliers.single ?? 1,
      double: venueProfile.multipliers.double ?? 1,
      triple: venueProfile.multipliers.triple ?? 1,
    },
    handedness: venueProfile.handedness ?? {},
    diagnostics: {
      overall: venueProfile.multipliers.overall ?? null,
      wobacon: venueProfile.multipliers.wobacon ?? null,
      xwobacon: venueProfile.multipliers.xwobacon ?? null,
      hardHit: venueProfile.multipliers.hard_hit ?? null,
      obp: venueProfile.multipliers.obp ?? null,
      bb: venueProfile.multipliers.bb ?? null,
      so: venueProfile.multipliers.so ?? null,
    },
    audit: {
      source: venueProfile.source,
      window: venueProfile.window,
      plateAppearances: venueProfile.plate_appearances,
      genericParkScoreDisabled: true,
    },
  };
}

export function adjustExpectedRuns(neutralExpectedRuns, context) {
  return neutralExpectedRuns * context.multipliers.run;
}

export function adjustEventRate(neutralRate, eventName, context) {
  const factor = context.multipliers[eventName] ?? 1;
  return neutralRate * factor;
}

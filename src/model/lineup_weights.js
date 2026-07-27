export function normalizeLineupShares(lineup) {
  if (!Array.isArray(lineup) || lineup.length === 0) {
    throw new RangeError("Lineup must contain at least one batter");
  }
  const total = lineup.reduce(
    (sum, batter) => sum + Number(batter.projected_plate_appearance_share ?? 0),
    0
  );
  if (!(total > 0)) throw new RangeError("Lineup PA shares must sum above zero");

  return lineup.map((batter) => ({
    ...batter,
    projected_plate_appearance_share:
      Number(batter.projected_plate_appearance_share ?? 0) / total,
  }));
}

/**
 * Initial projected lineup share rule.
 * Confirmed lineups should be replaced by a fitted batting-order PA model later.
 */
export function defaultBattingOrderShares(size = 9) {
  if (size <= 0) throw new RangeError("size must be positive");
  const raw = Array.from({ length: size }, (_, i) => Math.max(0.75, 1.08 - i * 0.04));
  const total = raw.reduce((a, b) => a + b, 0);
  return raw.map((x) => x / total);
}

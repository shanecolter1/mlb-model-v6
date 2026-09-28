// The actual game venue controls park factors. Home-club identity is not a
// safe substitute: clubs can play at a temporary or neutral venue.
function normalizeVenue(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

export function selectI2VenueProfile(game, profiles) {
  const venue = normalizeVenue(game?.venue?.name);
  if (!venue || !Array.isArray(profiles)) return null;
  const matches = profiles.filter(p => normalizeVenue(p?.venue_name) === venue);
  // An ambiguous profile is no better than a missing one. In shadow mode,
  // both conditions stay neutral and ineligible for betting.
  return matches.length === 1 ? matches[0] : null;
}

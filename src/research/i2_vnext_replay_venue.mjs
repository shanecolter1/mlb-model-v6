/** Prior-season park matching for historical Retrosheet research replays. */
const RETRO_SITE_TO_SAVANT_TEAM = {
  ANA01:'ANGELS', PHO01:'D-BACKS', ATL03:'BRAVES', BAL12:'ORIOLES',
  BOS07:'RED SOX', CHI12:'WHITE SOX', CHI11:'CUBS', CIN09:'REDS',
  CLE08:'GUARDIANS', DEN02:'ROCKIES', DET05:'TIGERS', HOU03:'ASTROS',
  KAN06:'ROYALS', LOS03:'DODGERS', MIA02:'MARLINS', MIL06:'BREWERS',
  MIN04:'TWINS', NYC21:'YANKEES', NYC20:'METS', PHI13:'PHILLIES',
  PIT08:'PIRATES', SAN02:'PADRES', SEA03:'MARINERS', SFO03:'GIANTS',
  STL10:'CARDINALS', ARL03:'RANGERS', TOR02:'BLUE JAYS', WAS11:'NATIONALS',
  OAK01:'ATHLETICS', STP01:'RAYS',
};

// Season-specific sites without a valid prior-season Savant home-venue profile.
const EXPLICIT_NEUTRAL_SITES_BY_SEASON = {
  2025: new Set([
  'SAC01', // Athletics at Sutter Health Park
  'TAM02', // Rays at George M. Steinbrenner Field
  'TOK01', // Tokyo Dome
  'BST01', // Bristol Motor Speedway
  'WIL02', // Williamsport special-event site
]),
};
const norm = x => String(x ?? '').trim().toUpperCase();
const venueNameKey = x => norm(x)
  .replace(/[^A-Z0-9]+/g,' ')
  .replace(/\b(STADIUM|BALLPARK|PARK|FIELD)\b/g,' ')
  .replace(/\s+/g,' ')
  .trim();

function matchByVenueName(game, parkProfiles) {
  const target = venueNameKey(game?.venue_name);
  if (!target) return null;
  return parkProfiles.find(p => venueNameKey(p?.venue_name) === target) || null;
}

export function venueForReplayGame(game, parkProfiles, season=2025) {
  const byName = matchByVenueName(game, parkProfiles);
  if (byName) {
    return {profile:byName,status:'MLB_VENUE_NAME_TO_PRIOR_SEASON_SAVANT'};
  }
  const site = norm(game.site);
  const explicit = EXPLICIT_NEUTRAL_SITES_BY_SEASON[Number(season)] || new Set();
  if (explicit.has(site)) {
    return {profile:null,status:`EXPLICIT_${Number(season)}_SITE_NEUTRAL`};
  }
  const savantTeam = RETRO_SITE_TO_SAVANT_TEAM[site];
  if (!savantTeam) return {profile:null,status:'UNMAPPED_RETROSHEET_SITE_NEUTRAL'};
  const profile = parkProfiles.find(p => norm(p.team) === savantTeam);
  return profile
    ? {profile,status:'RETROSHEET_SITE_TO_PRIOR_SEASON_SAVANT'}
    : {profile:null,status:'MAPPED_SITE_SAVANT_PROFILE_MISSING_NEUTRAL'};
}

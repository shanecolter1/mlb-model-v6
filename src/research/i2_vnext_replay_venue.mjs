/** Prior-season park matching for the 2025 Retrosheet research replay. */
const RETRO_SITE_TO_SAVANT_TEAM = {
  ANA01:'ANGELS', PHO01:'D-BACKS', ATL03:'BRAVES', BAL12:'ORIOLES',
  BOS07:'RED SOX', CHI12:'WHITE SOX', CHI11:'CUBS', CIN09:'REDS',
  CLE08:'GUARDIANS', DEN02:'ROCKIES', DET05:'TIGERS', HOU03:'ASTROS',
  KAN06:'ROYALS', LOS03:'DODGERS', MIA02:'MARLINS', MIL06:'BREWERS',
  MIN04:'TWINS', NYC21:'YANKEES', NYC20:'METS', PHI13:'PHILLIES',
  PIT08:'PIRATES', SAN02:'PADRES', SEA03:'MARINERS', SFO03:'GIANTS',
  STL10:'CARDINALS', ARL03:'RANGERS', TOR02:'BLUE JAYS', WAS11:'NATIONALS',
};

// The actual 2025 venue has no valid 2024 Savant profile at these sites.
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

export function venueForReplayGame(game, parkProfiles, season=2025) {
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

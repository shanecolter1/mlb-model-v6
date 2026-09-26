function norm(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function sameName(a, b) {
  const na = norm(a), nb = norm(b);
  if (!na || !nb) return false;
  if (na === nb) return true;
  const aa = na.split(' '), bb = nb.split(' ');
  return aa.at(-1) === bb.at(-1) && aa[0]?.[0] === bb[0]?.[0];
}

export function top4Agreement(a = [], b = []) {
  const x = a.slice(0, 4), y = b.slice(0, 4);
  if (x.length < 4 || y.length < 4) return { exactOrder: 0, overlap: 0, score: 0 };
  let exactOrder = 0;
  for (let i = 0; i < 4; i += 1) if (sameName(x[i], y[i])) exactOrder += 1;
  let overlap = 0;
  for (const name of x) if (y.some(other => sameName(name, other))) overlap += 1;
  return {
    exactOrder,
    overlap,
    score: Number(((exactOrder * 0.7 + overlap * 0.3) / 4).toFixed(4)),
  };
}

export function lineupDelta(selected = [], prior = []) {
  if (selected.length !== 9 || prior.length !== 9) {
    return { available: false, changedPlayers: null, changedTop4Players: null, orderChanges: null };
  }
  const priorSet = new Set(prior.map(norm));
  const changedPlayers = selected.filter(name => !priorSet.has(norm(name))).length;
  const priorTop = prior.slice(0, 4);
  const changedTop4Players = selected.slice(0, 4).filter(name => !priorTop.some(other => sameName(name, other))).length;
  let orderChanges = 0;
  for (let i = 0; i < 9; i += 1) if (!sameName(selected[i], prior[i])) orderChanges += 1;
  return { available: true, changedPlayers, changedTop4Players, orderChanges };
}

function validLineup(x) {
  return Array.isArray(x) && x.length === 9 && x.every(Boolean);
}

export function chooseLineup({
  official = null,
  teamBeat = null,
  rotowire = null,
  rosterResource = null,
  previous = null,
} = {}) {
  const officialLineup = validLineup(official?.lineup) ? official.lineup : null;
  if (officialLineup) {
    return {
      lineup: officialLineup,
      selectedSource: 'MLB_CONFIRMED',
      status: 'CONFIRMED',
      confidence: 'FINAL',
      rationale: 'MLB-confirmed batting order overrides all provisional inputs.',
      crossCheck: null,
      deltaVsPrevious: lineupDelta(officialLineup, previous?.lineup || []),
      sourceRecord: official,
    };
  }

  const teamBeatLineup = validLineup(teamBeat?.lineup) ? teamBeat.lineup : null;
  if (teamBeatLineup && ['CONFIRMED', 'EXPECTED'].includes(String(teamBeat?.status || '').toUpperCase())) {
    const cross = validLineup(rotowire?.lineup) ? top4Agreement(teamBeatLineup, rotowire.lineup) : null;
    return {
      lineup: teamBeatLineup,
      selectedSource: String(teamBeat?.sourceType || 'TEAM_BEAT').toUpperCase(),
      status: String(teamBeat.status || 'EXPECTED').toUpperCase(),
      confidence: String(teamBeat.status || '').toUpperCase() === 'CONFIRMED' ? 'HIGH' : (cross?.overlap >= 3 ? 'HIGH' : 'MEDIUM'),
      rationale: 'Explicit sourced team/beat lineup override.',
      crossCheck: cross ? { source: 'ROTOWIRE', ...cross } : null,
      deltaVsPrevious: lineupDelta(teamBeatLineup, previous?.lineup || []),
      sourceRecord: teamBeat,
    };
  }

  const rw = validLineup(rotowire?.lineup) ? rotowire.lineup : null;
  const rr = validLineup(rosterResource?.lineup) ? rosterResource.lineup : null;
  if (rw) {
    const cross = rr ? top4Agreement(rw, rr) : null;
    let confidence = 'MEDIUM';
    if (String(rotowire?.status || '').toUpperCase() === 'CONFIRMED') confidence = 'HIGH';
    else if (cross?.exactOrder === 4) confidence = 'HIGH';
    else if (cross && cross.overlap < 3) confidence = 'LOW';
    return {
      lineup: rw,
      selectedSource: 'ROTOWIRE_PROJECTED',
      status: String(rotowire?.status || 'PROJECTED').toUpperCase(),
      confidence,
      rationale: cross
        ? 'RotoWire daily projection selected; RosterResource platoon lineup used as structural cross-check.'
        : 'RotoWire daily projection selected; RosterResource cross-check unavailable.',
      crossCheck: cross ? { source: 'ROSTERRESOURCE', ...cross } : null,
      deltaVsPrevious: lineupDelta(rw, previous?.lineup || []),
      sourceRecord: rotowire,
    };
  }

  if (rr) {
    return {
      lineup: rr,
      selectedSource: 'ROSTERRESOURCE_PLATOON',
      status: 'PROJECTED',
      confidence: 'LOW',
      rationale: 'RotoWire unavailable; RosterResource go-to platoon lineup used as secondary provisional source.',
      crossCheck: null,
      deltaVsPrevious: lineupDelta(rr, previous?.lineup || []),
      sourceRecord: rosterResource,
    };
  }

  const prev = validLineup(previous?.lineup) ? previous.lineup : null;
  if (prev) {
    return {
      lineup: prev,
      selectedSource: 'MLB_PREVIOUS_GAME_FALLBACK',
      status: 'FALLBACK',
      confidence: 'LOW',
      rationale: 'No current projected lineup source available; previous completed MLB game lineup used as emergency fallback.',
      crossCheck: null,
      deltaVsPrevious: lineupDelta(prev, prev),
      sourceRecord: previous,
    };
  }

  return {
    lineup: null,
    selectedSource: 'MISSING',
    status: 'MISSING',
    confidence: 'MISSING',
    rationale: 'No usable lineup source.',
    crossCheck: null,
    deltaVsPrevious: { available: false, changedPlayers: null, changedTop4Players: null, orderChanges: null },
    sourceRecord: null,
  };
}

function pitcherName(x) {
  if (!x) return null;
  return x.name || [x.FirstName, x.LastName].filter(Boolean).join(' ') || null;
}

export function rotowireI2Pitcher(team = {}) {
  const opener = team?.OpenerPitcher || team?.openerPitcher || null;
  const primary = team?.PrimaryPitcher || team?.primaryPitcher || null;
  const starter = team?.StartingPitcher || team?.startingPitcher || null;

  if (opener && primary) {
    return {
      name: pitcherName(primary),
      id: primary?.MlbId || primary?.MLBId || null,
      role: 'PRIMARY_AFTER_OPENER',
      opener: pitcherName(opener),
      source: 'ROTOWIRE_PROJECTED_STARTERS',
      note: 'RotoWire identifies an opener and primary pitcher; I2 uses the primary pitcher rather than the opener.',
    };
  }
  if (opener && !primary) {
    return {
      name: null,
      id: null,
      role: 'OPENER_PRIMARY_UNRESOLVED',
      opener: pitcherName(opener),
      source: 'ROTOWIRE_PROJECTED_STARTERS',
      blocker: 'OPENER_WITHOUT_PRIMARY_I2_PITCHER',
    };
  }
  if (starter) {
    return {
      name: pitcherName(starter),
      id: starter?.MlbId || starter?.MLBId || null,
      role: 'STARTER',
      opener: null,
      source: 'ROTOWIRE_PROJECTED_STARTERS',
      note: null,
    };
  }
  return null;
}

export function chooseStarter({
  actual = null,
  teamBeat = null,
  rotowire = null,
  rosterResource = null,
  mlbProbable = null,
} = {}) {
  const actualName = pitcherName(actual);
  if (actualName) {
    return {
      name: actualName,
      id: actual?.id || null,
      role: actual?.role || 'ACTUAL_STARTER',
      selectedSource: 'MLB_ACTUAL_STARTER',
      confidence: 'FINAL',
      forcePreGameOverride: false,
      sourceRecord: actual,
    };
  }

  const teamBeatName = pitcherName(teamBeat);
  if (teamBeatName && ['CONFIRMED', 'EXPECTED'].includes(String(teamBeat?.status || '').toUpperCase())) {
    return {
      name: teamBeatName,
      id: teamBeat?.id || null,
      role: teamBeat?.role || 'STARTER',
      selectedSource: String(teamBeat?.sourceType || 'TEAM_BEAT').toUpperCase(),
      confidence: String(teamBeat?.status || '').toUpperCase() === 'CONFIRMED' ? 'HIGH' : 'MEDIUM',
      forcePreGameOverride: true,
      sourceRecord: teamBeat,
    };
  }

  if (rotowire?.name) {
    const rrName = pitcherName(rosterResource);
    const agrees = rrName ? sameName(rotowire.name, rrName) : null;
    return {
      name: rotowire.name,
      id: rotowire.id || null,
      role: rotowire.role || 'STARTER',
      opener: rotowire.opener || null,
      selectedSource: 'ROTOWIRE_PROJECTED_STARTERS',
      confidence: agrees === true ? 'HIGH' : 'MEDIUM',
      forcePreGameOverride: true,
      blocker: rotowire.blocker || null,
      sourceRecord: rotowire,
      crossCheck: rrName ? { source: 'ROSTERRESOURCE', name: rrName, agrees } : null,
    };
  }

  const rrName = pitcherName(rosterResource);
  if (rrName) {
    return {
      name: rrName,
      id: rosterResource?.id || null,
      role: rosterResource?.role || 'STARTER',
      selectedSource: 'ROSTERRESOURCE_PROBABLE',
      confidence: 'LOW',
      forcePreGameOverride: true,
      sourceRecord: rosterResource,
    };
  }

  const mlbName = pitcherName(mlbProbable);
  if (mlbName) {
    return {
      name: mlbName,
      id: mlbProbable?.id || null,
      role: 'STARTER',
      selectedSource: 'MLB_PROBABLE_FALLBACK',
      confidence: 'LOW',
      forcePreGameOverride: false,
      sourceRecord: mlbProbable,
    };
  }

  return {
    name: null,
    id: null,
    role: null,
    selectedSource: 'MISSING',
    confidence: 'MISSING',
    forcePreGameOverride: false,
    blocker: 'NO_PROBABLE_I2_PITCHER',
    sourceRecord: null,
  };
}

export { norm, sameName };

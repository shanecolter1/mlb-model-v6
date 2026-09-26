export function normName(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

export function sameName(a, b) {
  return normName(a) === normName(b);
}

export function lineupDelta(primary = [], comparator = []) {
  const p = Array.isArray(primary) ? primary : [];
  const c = Array.isArray(comparator) ? comparator : [];
  const top4Differences = [];
  for (let i = 0; i < 4; i++) {
    if (!sameName(p[i], c[i])) top4Differences.push({ spot: i + 1, primary: p[i] || null, comparator: c[i] || null });
  }
  const ps = new Set(p.map(normName).filter(Boolean));
  const cs = new Set(c.map(normName).filter(Boolean));
  return {
    top4Differences,
    top4Agreement: 4 - top4Differences.length,
    playersAdded: p.filter(x => !cs.has(normName(x))),
    playersMissing: c.filter(x => !ps.has(normName(x))),
    fullOrderExact: p.length === 9 && c.length === 9 && p.every((x, i) => sameName(x, c[i])),
    playerOverlap: p.filter(x => cs.has(normName(x))).length,
  };
}

function usableLineup(x) {
  return Array.isArray(x?.lineup) && x.lineup.length === 9 && new Set(x.lineup.map(normName)).size === 9;
}

function confidenceFor(selected, rosterResource, previous) {
  if (!selected) return 'MISSING';
  if (selected.source === 'TEAM_BEAT_NEWS_CONFIRMED') return 'HIGH';
  if (selected.source === 'ROTOWIRE_CONFIRMED') return 'HIGH';
  if (selected.source === 'MLB_PREVIOUS_GAME_FALLBACK') return 'LOW';
  if (selected.source?.startsWith('ROTOWIRE')) {
    if (usableLineup(rosterResource)) {
      const d = lineupDelta(selected.lineup, rosterResource.lineup);
      if (d.top4Differences.length === 0 && d.playerOverlap >= 8) return 'HIGH';
      if (d.top4Differences.length <= 1 && d.playerOverlap >= 7) return 'MEDIUM';
      return 'LOW';
    }
    if (usableLineup(previous)) {
      const d = lineupDelta(selected.lineup, previous.lineup);
      return d.top4Differences.length <= 1 ? 'MEDIUM' : 'LOW';
    }
    return 'MEDIUM';
  }
  return 'LOW';
}

export function chooseLineup({ news = null, rotowire = null, rosterResource = null, previous = null } = {}) {
  let selected = null;
  if (usableLineup(news) && String(news.status || '').toUpperCase().includes('CONFIRMED')) {
    selected = { ...news, source: 'TEAM_BEAT_NEWS_CONFIRMED' };
  } else if (usableLineup(rotowire)) {
    const status = String(rotowire.status || '').toUpperCase();
    selected = { ...rotowire, source: status.includes('CONFIRMED') || status === 'C' ? 'ROTOWIRE_CONFIRMED' : 'ROTOWIRE_EXPECTED' };
  } else if (usableLineup(news)) {
    selected = { ...news, source: 'TEAM_BEAT_NEWS_EXPECTED' };
  } else if (usableLineup(previous)) {
    selected = { ...previous, source: 'MLB_PREVIOUS_GAME_FALLBACK' };
  }

  const rrDelta = selected && usableLineup(rosterResource) ? lineupDelta(selected.lineup, rosterResource.lineup) : null;
  const previousDelta = selected && usableLineup(previous) ? lineupDelta(selected.lineup, previous.lineup) : null;
  return {
    selected,
    confidence: confidenceFor(selected, rosterResource, previous),
    rosterResourceDelta: rrDelta,
    previousGameDelta: previousDelta,
    top4Disagreement: Boolean(rrDelta?.top4Differences?.length),
  };
}

function usableStarter(x) {
  return Boolean(x?.name);
}

export function chooseStarter({ news = null, rotowire = null, mlb = null } = {}) {
  if (usableStarter(news) && String(news.status || '').toUpperCase().includes('CONFIRMED')) {
    return { ...news, source: 'TEAM_BEAT_NEWS_CONFIRMED', overrideMlbProbable: true, confidence: 'HIGH' };
  }
  if (usableStarter(rotowire)) {
    return { ...rotowire, source: 'ROTOWIRE_PROJECTED_STARTERS', overrideMlbProbable: true, confidence: 'MEDIUM_HIGH' };
  }
  if (usableStarter(news)) {
    return { ...news, source: 'TEAM_BEAT_NEWS_EXPECTED', overrideMlbProbable: true, confidence: 'MEDIUM' };
  }
  if (usableStarter(mlb)) {
    return { ...mlb, source: 'MLB_PROBABLE_FALLBACK', overrideMlbProbable: false, confidence: 'MEDIUM' };
  }
  return null;
}

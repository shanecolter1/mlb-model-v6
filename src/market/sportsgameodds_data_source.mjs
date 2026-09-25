const DEFAULT_BASE_URL = 'https://api.sportsgameodds.com/v2';

export const SPORTSGAMEODDS_DATA_SOURCE = Object.freeze({
  provider: 'SPORTSGAMEODDS',
  providerHost: 'sportsgameodds.com',
  baseUrl: DEFAULT_BASE_URL,
  apiKeyEnv: 'SPORTSGAMEODDS_API_KEY',
  policyVersion: '1.0.0',
});

export const ROOKIE_TARGET_BOOKMAKERS = Object.freeze([
  'draftkings',
  'fanduel',
  'betmgm',
  'caesars',
]);

function requireApiKey(apiKey = process.env[SPORTSGAMEODDS_DATA_SOURCE.apiKeyEnv]) {
  const value = String(apiKey || '').trim();
  if (!value) throw new Error(`${SPORTSGAMEODDS_DATA_SOURCE.apiKeyEnv} is required`);
  return value;
}

function assertInning(inning) {
  const n = Number(inning);
  if (!Number.isInteger(n) || n < 1 || n > 9) throw new Error(`Invalid MLB inning: ${inning}`);
  return n;
}

export function buildMlbInningOddIds({ innings = [1,2,3,4,5,6,7,8,9] } = {}) {
  const ids = [];
  for (const raw of innings) {
    const inning = assertInning(raw);
    const period = `${inning}i`;

    for (const entity of ['away', 'home']) {
      ids.push(`points-${entity}-${period}-ou-over`);
      ids.push(`points-${entity}-${period}-ou-under`);
    }

    if (inning <= 8) {
      ids.push(`points-all-${period}-ou-over`);
      ids.push(`points-all-${period}-ou-under`);
      ids.push(`points-away-${period}-ml3way-away`);
      ids.push(`points-all-${period}-ml3way-draw`);
      ids.push(`points-home-${period}-ml3way-home`);
    }
  }
  return ids;
}

export function buildMlbNinthInningCandidateOddIds() {
  return [
    'points-all-9i-ou-over',
    'points-all-9i-ou-under',
    'points-away-9i-ml3way-away',
    'points-all-9i-ml3way-draw',
    'points-home-9i-ml3way-home',
  ];
}

function buildUrl(pathname, params = {}) {
  const url = new URL(`${SPORTSGAMEODDS_DATA_SOURCE.baseUrl}${pathname}`);
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value)) url.searchParams.set(key, value.join(','));
    else url.searchParams.set(key, String(value));
  }
  return url;
}

async function fetchSgoJson(pathname, params = {}, options = {}) {
  const apiKey = requireApiKey(options.apiKey);
  const url = buildUrl(pathname, params);
  const response = await fetch(url, {
    headers: {
      accept: 'application/json',
      'x-api-key': apiKey,
      'user-agent': options.userAgent || 'MLB-Model-SportsGameOdds/1.0',
    },
    signal: options.signal,
  });
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch { body = { raw: text }; }
  if (!response.ok) {
    const detail = body?.message || body?.error || body?.detail || text.slice(0, 300);
    throw new Error(`SportsGameOdds ${response.status} ${response.statusText}${detail ? `: ${detail}` : ''}`);
  }
  return body;
}

export function assertPostFreezeContext(context = {}) {
  if (context.projectionFrozen !== true) throw new Error('SportsGameOdds price retrieval is post-freeze only: projectionFrozen must be true');
  const frozenAt = String(context.frozenAt || '').trim();
  if (!frozenAt || Number.isNaN(Date.parse(frozenAt))) throw new Error('SportsGameOdds price retrieval requires a valid frozenAt timestamp');
  return true;
}

export async function fetchMlbMarketSupport({ oddIDs, apiKey, signal } = {}) {
  // SportsGameOdds currently rejects combining leagueID and bookmakerID on /markets.
  // Query MLB + oddID here; bookmaker-specific availability is verified on /events.
  return fetchSgoJson('/markets', {
    leagueID: 'MLB',
    oddID: oddIDs,
  }, { apiKey, signal });
}

function parseAmericanOdds(value) {
  if (value === undefined || value === null || value === '') return null;
  const n = Number(String(value).replace('+', ''));
  return Number.isFinite(n) ? n : null;
}

function parseLine(value) {
  if (value === undefined || value === null || value === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function classifyMlbOdd(odd = {}) {
  const inningMatch = String(odd.periodID || '').match(/^([1-9])i$/);
  if (!inningMatch || String(odd.statID || '') !== 'points') return null;
  const inning = Number(inningMatch[1]);
  const entity = String(odd.statEntityID || '');
  const betType = String(odd.betTypeID || '');
  const side = String(odd.sideID || '');

  if (betType === 'ou' && entity === 'all' && ['over','under'].includes(side)) {
    return { marketType: 'FULL_INNING_TOTAL', inning, segment: 'full', side };
  }
  if (betType === 'ou' && ['away','home'].includes(entity) && ['over','under'].includes(side)) {
    return { marketType: 'TEAM_HALF_INNING_TOTAL', inning, segment: entity === 'away' ? 'top' : 'bottom', teamSide: entity, side };
  }
  if (betType === 'ml3way' && ['away','home','draw'].includes(side)) {
    return { marketType: 'FULL_INNING_3WAY', inning, segment: 'full', side };
  }
  return null;
}

export function normalizeSgoEvent(event, { bookmakerIDs = ROOKIE_TARGET_BOOKMAKERS } = {}) {
  const requestedBooks = bookmakerIDs?.length ? new Set(bookmakerIDs.map(x => String(x).toLowerCase())) : null;
  const markets = [];
  for (const [oddID, odd] of Object.entries(event?.odds || {})) {
    const classification = classifyMlbOdd(odd);
    if (!classification) continue;
    const prices = [];
    for (const [bookmakerID, book] of Object.entries(odd?.byBookmaker || {})) {
      const key = String(bookmakerID).toLowerCase();
      if (requestedBooks && !requestedBooks.has(key)) continue;
      const entries = [
        { source: book, isAlternateLine: false },
        ...(Array.isArray(book?.altLines) ? book.altLines.map(source => ({ source, isAlternateLine: true })) : []),
      ];
      const seen = new Set();
      for (const { source, isAlternateLine } of entries) {
        const americanOdds = parseAmericanOdds(source?.odds);
        const line = parseLine(source?.overUnder);
        const dedupeKey = `${americanOdds}|${line}|${source?.available === true}`;
        if (seen.has(dedupeKey)) continue;
        seen.add(dedupeKey);
        prices.push({
          bookmakerID: key,
          available: source?.available === true,
          americanOdds,
          line,
          isAlternateLine,
          lastUpdatedAt: source?.lastUpdatedAt || book?.lastUpdatedAt || null,
          deeplink: source?.deeplink || book?.deeplink || null,
          openAmericanOdds: isAlternateLine ? null : parseAmericanOdds(book?.openOdds),
          closeAmericanOdds: isAlternateLine ? null : parseAmericanOdds(book?.closeOdds),
          openLine: isAlternateLine ? null : parseLine(book?.openOverUnder),
          closeLine: isAlternateLine ? null : parseLine(book?.closeOverUnder),
        });
      }
    }
    markets.push({
      oddID,
      ...classification,
      providerFairOdds: parseAmericanOdds(odd?.fairOdds),
      providerConsensusOdds: parseAmericanOdds(odd?.bookOdds),
      providerFairLine: parseLine(odd?.fairOverUnder),
      providerConsensusLine: parseLine(odd?.bookOverUnder),
      started: odd?.started === true,
      ended: odd?.ended === true,
      cancelled: odd?.cancelled === true,
      prices,
    });
  }

  return {
    provider: SPORTSGAMEODDS_DATA_SOURCE.provider,
    eventID: String(event?.eventID || ''),
    leagueID: event?.leagueID || null,
    startTime: event?.startTime || event?.status?.startsAt || null,
    status: event?.status || null,
    away: {
      teamID: event?.teams?.away?.teamID || null,
      name: event?.teams?.away?.names?.long || event?.teams?.away?.name || null,
      short: event?.teams?.away?.names?.short || null,
    },
    home: {
      teamID: event?.teams?.home?.teamID || null,
      name: event?.teams?.home?.names?.long || event?.teams?.home?.name || null,
      short: event?.teams?.home?.names?.short || null,
    },
    markets,
  };
}

export async function fetchMlbInningEvents({
  freezeContext,
  bookmakerIDs = ROOKIE_TARGET_BOOKMAKERS,
  includeOpenCloseOdds = false,
  includeAltLines = true,
  oddIDs = buildMlbInningOddIds(),
  apiKey,
  signal,
  limit = 100,
} = {}) {
  assertPostFreezeContext(freezeContext);
  const payload = await fetchSgoJson('/events', {
    leagueID: 'MLB',
    oddsAvailable: true,
    started: false,
    oddID: oddIDs,
    bookmakerID: bookmakerIDs,
    includeOpenCloseOdds,
    includeAltLines,
    limit,
  }, { apiKey, signal });
  const events = Array.isArray(payload?.data) ? payload.data : [];
  return {
    fetchedAt: new Date().toISOString(),
    provider: SPORTSGAMEODDS_DATA_SOURCE.provider,
    freezeContext,
    nextCursor: payload?.nextCursor || null,
    events: events.map(event => normalizeSgoEvent(event, { bookmakerIDs })),
  };
}

export function localDateForEvent(startTime, timeZone = 'America/Chicago') {
  if (!startTime || Number.isNaN(Date.parse(startTime))) return null;
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date(startTime));
  const map = Object.fromEntries(parts.map(x => [x.type, x.value]));
  return `${map.year}-${map.month}-${map.day}`;
}

export function filterEventsByLocalDate(events, date, timeZone = 'America/Chicago') {
  return (events || []).filter(event => localDateForEvent(event.startTime, timeZone) === date);
}

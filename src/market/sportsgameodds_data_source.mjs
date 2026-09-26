const DEFAULT_BASE_URL = 'https://api.sportsgameodds.com/v2';

export const SPORTSGAMEODDS_DATA_SOURCE = Object.freeze({
  provider: 'SPORTSGAMEODDS',
  providerHost: 'sportsgameodds.com',
  baseUrl: DEFAULT_BASE_URL,
  apiKeyEnv: 'SPORTSGAMEODDS_API_KEY',
  policyVersion: '2.0.0',
});

export const ROOKIE_TARGET_BOOKMAKERS = Object.freeze([
  'draftkings',
  'fanduel',
  'betmgm',
  'caesars',
]);


export const MARKET_ISOLATION = Object.freeze({
  preFreeze: Object.freeze({
    allowedBookmaker: 'draftkings',
    allowedMarket: 'points-all-game-ou-over',
    allowedFields: Object.freeze(['eventId', 'commenceTime', 'awayTeam', 'homeTeam', 'fullGameTotal', 'bookmaker', 'lastUpdate']),
    scope: 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES',
  }),
  postFreeze: Object.freeze({
    allowed: true,
    purpose: 'MARKET_ENUMERATION_PRICE_EV_ONLY',
  }),
});

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
      // Sportsbooks may express the exact same 0.5-run proposition as
      // "Any Runs? Yes/No" instead of an Over/Under. These are canonical
      // economic equivalents for full-inning I2 price discovery:
      //   NO  = Under 0.5
      //   YES = Over 0.5
      ids.push(`points-all-${period}-yn-yes`);
      ids.push(`points-all-${period}-yn-no`);
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

export function extractMlbDraftKingsFullGameTotalPoints(payload) {
  const events = Array.isArray(payload?.data) ? payload.data : [];
  const oddID = 'points-all-game-ou-over';
  return events.flatMap(event => {
    const odd = event?.odds?.[oddID];
    const book = odd?.byBookmaker?.draftkings;
    const point = parseLine(book?.overUnder);
    if (!book || book.available !== true || !Number.isFinite(point)) return [];
    return [{
      eventId: String(event?.eventID || ''),
      commenceTime: event?.startTime || event?.status?.startsAt || null,
      awayTeam: event?.teams?.away?.names?.long || event?.teams?.away?.name || null,
      homeTeam: event?.teams?.home?.names?.long || event?.teams?.home?.name || null,
      fullGameTotal: point,
      bookmaker: 'draftkings',
      lastUpdate: book?.lastUpdatedAt || null,
    }];
  }).filter(row => row.eventId && row.commenceTime && row.awayTeam && row.homeTeam);
}

export async function fetchMlbDraftKingsFullGameTotalPoints({ apiKey, signal, limit = 100 } = {}) {
  const payload = await fetchSgoJson('/events', {
    leagueID: 'MLB',
    oddsAvailable: true,
    started: false,
    oddID: MARKET_ISOLATION.preFreeze.allowedMarket,
    bookmakerID: MARKET_ISOLATION.preFreeze.allowedBookmaker,
    includeAltLines: false,
    includeOpenCloseOdds: false,
    limit,
  }, { apiKey, signal });
  return extractMlbDraftKingsFullGameTotalPoints(payload);
}

export function assertPreFreezeIsolation(record) {
  const keys = Object.keys(record || {});
  const allowed = new Set(MARKET_ISOLATION.preFreeze.allowedFields);
  const forbidden = keys.filter(key => !allowed.has(key));
  if (forbidden.length) throw new Error(`Pre-freeze SportsGameOdds payload contains forbidden fields: ${forbidden.join(', ')}`);
  if (record.bookmaker !== MARKET_ISOLATION.preFreeze.allowedBookmaker) {
    throw new Error(`Pre-freeze bookmaker must be ${MARKET_ISOLATION.preFreeze.allowedBookmaker}`);
  }
  if (!Number.isFinite(Number(record.fullGameTotal))) throw new Error('Pre-freeze fullGameTotal must be numeric');
  return true;
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

export async function fetchMlbSecondInningMarketCatalog({ apiKey, signal, limit = 10000 } = {}) {
  const fetchCatalog = async isSupported => {
    const rows = [];
    let cursor = null;
    do {
      const payload = await fetchSgoJson('/markets', {
        leagueID: 'MLB',
        isSupported,
        limit,
        cursor: cursor || undefined,
      }, { apiKey, signal });
      if (Array.isArray(payload?.data)) rows.push(...payload.data);
      cursor = payload?.nextCursor || null;
    } while (cursor);
    return rows;
  };

  const [supported, unsupported] = await Promise.all([
    fetchCatalog(true),
    fetchCatalog(false),
  ]);
  const byOddID = new Map();
  for (const market of [...unsupported, ...supported]) {
    if (market?.oddID) byOddID.set(market.oddID, market);
  }
  const allMarkets = [...byOddID.values()];
  const secondInningPattern = /(\b2i\b|2nd\s+inning|second\s+inning|inning\s*2)/i;
  const isSecondInning = market => {
    if (String(market?.periodID || '').toLowerCase() === '2i') return true;
    const searchable = [
      market?.oddID,
      market?.marketGroupID,
      market?.marketGroupName,
      ...Object.values(market?.marketGroupNameBySport || {}),
    ].filter(Boolean).join(' ');
    return secondInningPattern.test(searchable);
  };
  const markets = allMarkets.filter(isSecondInning);
  return {
    fetchedAt: new Date().toISOString(),
    leagueID: 'MLB',
    discoveryScope: 'ALL_MLB_MARKETS_SUPPORTED_AND_UNSUPPORTED',
    allMarketCount: allMarkets.length,
    supportedMarketCount: supported.length,
    unsupportedMarketCount: unsupported.length,
    secondInningCandidateCount: markets.length,
    markets,
  };
}

export async function fetchMlbRawEventsForOddIds({
  freezeContext,
  oddIDs = [],
  bookmakerIDs = [],
  includeOpenCloseOdds = false,
  includeAltLines = true,
  apiKey,
  signal,
  limit = 100,
  chunkSize = 75,
} = {}) {
  assertPostFreezeContext(freezeContext);
  const ids = [...new Set((oddIDs || []).map(String).filter(Boolean))];
  const merged = new Map();
  for (let i = 0; i < ids.length; i += chunkSize) {
    const chunk = ids.slice(i, i + chunkSize);
    const payload = await fetchSgoJson('/events', {
      leagueID: 'MLB',
      oddsAvailable: true,
      started: false,
      oddID: chunk,
      bookmakerID: bookmakerIDs?.length ? bookmakerIDs : undefined,
      includeOpenCloseOdds,
      includeAltLines,
      limit,
    }, { apiKey, signal });
    for (const event of Array.isArray(payload?.data) ? payload.data : []) {
      const eventID = String(event?.eventID || '');
      if (!eventID) continue;
      if (!merged.has(eventID)) {
        merged.set(eventID, { ...event, odds: { ...(event?.odds || {}) } });
      } else {
        const current = merged.get(eventID);
        current.odds = { ...(current.odds || {}), ...(event?.odds || {}) };
      }
    }
  }
  return {
    fetchedAt: new Date().toISOString(),
    provider: SPORTSGAMEODDS_DATA_SOURCE.provider,
    freezeContext,
    requestedOddIDs: ids,
    events: [...merged.values()],
  };
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
  if (betType === 'yn' && entity === 'all' && ['yes','no'].includes(side) && inning <= 8) {
    return {
      marketType: 'FULL_INNING_ANY_RUNS',
      inning,
      segment: 'full',
      side,
      equivalentMarketType: 'FULL_INNING_TOTAL',
      equivalentSide: side === 'no' ? 'under' : 'over',
      equivalentLine: 0.5,
    };
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
    bookmakerID: bookmakerIDs?.length ? bookmakerIDs : undefined,
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

const DEFAULT_BASE_URL = 'https://api.theoddsapi.com';

export const SPORTSBOOK_DATA_SOURCE = Object.freeze({
  provider: 'THE_ODDS_API',
  providerHost: 'theoddsapi.com',
  baseUrl: DEFAULT_BASE_URL,
  apiKeyEnv: 'ODDS_API_KEY',
  policyVersion: '1.2.0',
});

export const REQUIRED_POST_FREEZE_BOOKS = Object.freeze([
  Object.freeze({ displayName: 'FanDuel', key: 'fanduel' }),
  Object.freeze({ displayName: 'DraftKings', key: 'draftkings' }),
  Object.freeze({ displayName: 'Hard Rock Bet', key: 'hardrockbet' }),
  Object.freeze({ displayName: 'Fanatics', key: 'fanatics' }),
  Object.freeze({ displayName: 'Caesars', key: 'caesars' }),
  Object.freeze({ displayName: 'BetMGM', key: 'betmgm' }),
  Object.freeze({ displayName: 'Kalshi', key: 'kalshi' }),
  Object.freeze({ displayName: 'Polymarket', key: 'polymarket' }),
  Object.freeze({ displayName: 'Pinnacle', key: 'pinnacle' }),
]);

export const MARKET_ISOLATION = Object.freeze({
  preFreeze: Object.freeze({
    allowedBookmaker: 'draftkings',
    allowedMarket: 'totals',
    allowedFields: Object.freeze(['eventId', 'commenceTime', 'awayTeam', 'homeTeam', 'fullGameTotal', 'bookmaker', 'lastUpdate']),
    scope: 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES',
  }),
  postFreeze: Object.freeze({
    allowed: true,
    purpose: 'MARKET_ENUMERATION_PRICE_EV_ONLY',
    requiredBooks: REQUIRED_POST_FREEZE_BOOKS,
  }),
});

function requireApiKey(apiKey = process.env[SPORTSBOOK_DATA_SOURCE.apiKeyEnv]) {
  const value = String(apiKey || '').trim();
  if (!value) throw new Error(`${SPORTSBOOK_DATA_SOURCE.apiKeyEnv} is required for sportsbook data retrieval`);
  return value;
}

function buildUrl(pathname, params = {}) {
  const url = new URL(`${SPORTSBOOK_DATA_SOURCE.baseUrl}${pathname}`);
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    url.searchParams.set(key, String(value));
  }
  return url;
}

export async function fetchOddsApiJson(pathname, params = {}, options = {}) {
  const apiKey = requireApiKey(options.apiKey);
  const url = buildUrl(pathname, params);
  const response = await fetch(url, {
    headers: {
      accept: 'application/json',
      'x-api-key': apiKey,
      'user-agent': options.userAgent || 'MLB-Model-Sportsbook-Data/1.2',
    },
  });
  const text = await response.text();
  if (!response.ok) {
    let detail = '';
    try {
      const parsed = JSON.parse(text);
      detail = parsed?.detail || parsed?.message || parsed?.error || '';
    } catch {}
    throw new Error(`The Odds API ${response.status} ${response.statusText}${detail ? `: ${detail}` : ''}`);
  }
  return text ? JSON.parse(text) : null;
}

function unwrapEvents(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function normalizeBookEntries(event) {
  const books = Array.isArray(event?.books) ? event.books : [];
  return books.map(entry => ({
    key: String(entry.book || entry.key || '').toLowerCase(),
    market: String(entry.market || '').toLowerCase(),
    updatedAt: entry.updated_at || entry.last_update || null,
    outcomes: Array.isArray(entry.outcomes) ? entry.outcomes : [],
    raw: entry,
  }));
}

export async function fetchMlbDraftKingsFullGameTotals(options = {}) {
  const payload = await fetchOddsApiJson('/odds/', {
    sport_key: 'baseball_mlb',
    markets: 'totals',
    bookmakers: 'draftkings',
    oddsFormat: 'american',
    commenceTimeFrom: options.commenceTimeFrom,
    commenceTimeTo: options.commenceTimeTo,
  }, options);

  return unwrapEvents(payload).flatMap(event => {
    const entry = normalizeBookEntries(event).find(x => x.key === 'draftkings' && x.market === 'totals');
    if (!entry) return [];
    const over = entry.outcomes.find(x => String(x.name || '').toLowerCase() === 'over');
    const under = entry.outcomes.find(x => String(x.name || '').toLowerCase() === 'under');
    const point = Number(over?.point ?? over?.line ?? under?.point ?? under?.line);
    if (!Number.isFinite(point)) return [];
    return [{
      eventId: String(event.event_id || event.id || ''),
      commenceTime: event.start_time || event.commence_time || null,
      awayTeam: event.away_team,
      homeTeam: event.home_team,
      fullGameTotal: point,
      bookmaker: 'draftkings',
      lastUpdate: entry.updatedAt,
    }];
  });
}

export async function fetchSportsbookMarkets({ sportKey = 'baseball_mlb', eventId, bookmakers, markets, regions, apiKey } = {}) {
  if (!eventId) throw new Error('eventId is required for post-freeze event market retrieval');
  const payload = await fetchOddsApiJson('/odds/', {
    sport_key: sportKey,
    event_id: eventId,
    bookmakers,
    markets,
    regions,
    oddsFormat: 'american',
  }, { apiKey });
  return unwrapEvents(payload)[0] || null;
}

export async function fetchRequiredPostFreezeMarkets({ sportKey = 'baseball_mlb', eventId, markets, apiKey } = {}) {
  const event = await fetchSportsbookMarkets({ sportKey, eventId, markets, apiKey });
  const entries = normalizeBookEntries(event);
  return REQUIRED_POST_FREEZE_BOOKS.flatMap(config => {
    const matched = entries.filter(x => x.key === config.key && (!markets || String(markets).split(',').includes(x.market)));
    if (!matched.length) return [];
    return [{ displayName: config.displayName, bookmakerKey: config.key, availability: 'AVAILABLE', markets: matched.map(x => x.raw) }];
  });
}

export function extractBookMarketEntries(event) {
  return normalizeBookEntries(event);
}

export function unwrapOddsEvents(payload) {
  return unwrapEvents(payload);
}

export function assertPreFreezeIsolation(record) {
  const keys = Object.keys(record || {});
  const allowed = new Set(MARKET_ISOLATION.preFreeze.allowedFields);
  const forbidden = keys.filter(key => !allowed.has(key));
  if (forbidden.length) throw new Error(`Pre-freeze sportsbook payload contains forbidden fields: ${forbidden.join(', ')}`);
  if (record.bookmaker !== MARKET_ISOLATION.preFreeze.allowedBookmaker) throw new Error(`Pre-freeze bookmaker must be ${MARKET_ISOLATION.preFreeze.allowedBookmaker}`);
  if (!Number.isFinite(Number(record.fullGameTotal))) throw new Error('Pre-freeze fullGameTotal must be numeric');
  return true;
}

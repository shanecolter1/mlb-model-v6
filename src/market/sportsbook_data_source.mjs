const DEFAULT_BASE_URL = 'https://api.the-odds-api.com/v4';

export const SPORTSBOOK_DATA_SOURCE = Object.freeze({
  provider: 'THE_ODDS_API',
  baseUrl: DEFAULT_BASE_URL,
  apiKeyEnv: 'ODDS_API_KEY',
  policyVersion: '1.0.0',
});

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
  }),
});

function requireApiKey(apiKey = process.env[SPORTSBOOK_DATA_SOURCE.apiKeyEnv]) {
  const value = String(apiKey || '').trim();
  if (!value) throw new Error(`${SPORTSBOOK_DATA_SOURCE.apiKeyEnv} is required for sportsbook data retrieval`);
  return value;
}

function buildUrl(pathname, params = {}, apiKey) {
  const url = new URL(`${SPORTSBOOK_DATA_SOURCE.baseUrl}${pathname}`);
  url.searchParams.set('apiKey', requireApiKey(apiKey));
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    url.searchParams.set(key, String(value));
  }
  return url;
}

export async function fetchOddsApiJson(pathname, params = {}, options = {}) {
  const url = buildUrl(pathname, params, options.apiKey);
  const response = await fetch(url, {
    headers: {
      accept: 'application/json',
      'user-agent': options.userAgent || 'MLB-Model-Sportsbook-Data/1.0',
    },
  });
  if (!response.ok) {
    throw new Error(`The Odds API ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function fetchMlbDraftKingsFullGameTotals(options = {}) {
  const events = await fetchOddsApiJson('/sports/baseball_mlb/odds', {
    regions: options.regions || 'us',
    markets: 'totals',
    bookmakers: 'draftkings',
    oddsFormat: 'american',
    dateFormat: 'iso',
  }, options);

  return (events || []).flatMap(event => {
    const book = (event.bookmakers || []).find(x => x.key === 'draftkings');
    if (!book) return [];
    const market = (book.markets || []).find(x => x.key === 'totals');
    if (!market) return [];
    const over = (market.outcomes || []).find(x => String(x.name).toLowerCase() === 'over');
    const under = (market.outcomes || []).find(x => String(x.name).toLowerCase() === 'under');
    const point = Number(over?.point ?? under?.point);
    if (!Number.isFinite(point)) return [];
    return [{
      eventId: String(event.id),
      commenceTime: event.commence_time,
      awayTeam: event.away_team,
      homeTeam: event.home_team,
      fullGameTotal: point,
      bookmaker: 'draftkings',
      lastUpdate: book.last_update || market.last_update || null,
    }];
  });
}

export async function fetchSportsbookMarkets({ sportKey = 'baseball_mlb', eventId, bookmakers, markets, regions = 'us', apiKey } = {}) {
  if (!eventId) throw new Error('eventId is required for post-freeze event market retrieval');
  if (!markets) throw new Error('markets is required for post-freeze event market retrieval');
  return fetchOddsApiJson(`/sports/${sportKey}/events/${eventId}/odds`, {
    regions,
    bookmakers,
    markets,
    oddsFormat: 'american',
    dateFormat: 'iso',
  }, { apiKey });
}

export function assertPreFreezeIsolation(record) {
  const keys = Object.keys(record || {});
  const allowed = new Set(MARKET_ISOLATION.preFreeze.allowedFields);
  const forbidden = keys.filter(key => !allowed.has(key));
  if (forbidden.length) {
    throw new Error(`Pre-freeze sportsbook payload contains forbidden fields: ${forbidden.join(', ')}`);
  }
  if (record.bookmaker !== MARKET_ISOLATION.preFreeze.allowedBookmaker) {
    throw new Error(`Pre-freeze bookmaker must be ${MARKET_ISOLATION.preFreeze.allowedBookmaker}`);
  }
  if (!Number.isFinite(Number(record.fullGameTotal))) {
    throw new Error('Pre-freeze fullGameTotal must be numeric');
  }
  return true;
}

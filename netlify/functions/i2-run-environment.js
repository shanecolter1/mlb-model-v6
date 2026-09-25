const REPO_RAW_BASE = 'https://raw.githubusercontent.com/shanecolter1/mlb-model-v6/main/data/runtime/i2';
const EXPECTED_SOURCE = 'SPORTSGAMEODDS';
const EXPECTED_SCOPE = 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES';

function chicagoDate() {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Chicago',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  }).formatToParts(new Date());
  const map = Object.fromEntries(parts.map(x => [x.type, x.value]));
  return `${map.year}-${map.month}-${map.day}`;
}

function validDate(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(value || ''));
}

function sanitizeArtifact(raw) {
  if (!raw || raw.source !== EXPECTED_SOURCE) {
    throw new Error(`Unexpected run-environment source: ${raw?.source || 'missing'}`);
  }
  if (raw.scope !== EXPECTED_SCOPE) {
    throw new Error(`Unexpected run-environment scope: ${raw?.scope || 'missing'}`);
  }
  if (raw?.marketIsolation?.pricesExposedToPredictionEngine !== false) {
    throw new Error('Run-environment artifact failed market-isolation validation');
  }

  const events = (Array.isArray(raw.events) ? raw.events : []).map(event => ({
    eventId: String(event?.eventId || ''),
    commenceTime: event?.commenceTime || null,
    awayTeam: event?.awayTeam || null,
    homeTeam: event?.homeTeam || null,
    fullGameTotal: Number(event?.fullGameTotal),
    bookmaker: event?.bookmaker || null,
    firstSeenAt: event?.firstSeenAt || null,
    lastSeenAt: event?.lastSeenAt || null,
    latestObservedTotal: Number(event?.latestObservedTotal),
    latestBookmakerUpdate: event?.latestBookmakerUpdate || null
  })).filter(event =>
    event.eventId &&
    event.commenceTime &&
    event.awayTeam &&
    event.homeTeam &&
    Number.isFinite(event.fullGameTotal) &&
    event.bookmaker === 'draftkings'
  );

  return {
    date: raw.date || null,
    capturedAt: raw.capturedAt || null,
    source: EXPECTED_SOURCE,
    sourcePolicyVersion: raw.sourcePolicyVersion || null,
    scope: EXPECTED_SCOPE,
    totalDefinition: raw.totalDefinition || null,
    marketIsolation: {
      phase: 'PRE_FREEZE',
      allowedBookmaker: 'draftkings',
      allowedMarket: 'points-all-game-ou-over',
      pricesExposedToPredictionEngine: false
    },
    events
  };
}

exports.handler = async function(event) {
  const headers = {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'access-control-allow-origin': '*'
  };
  const send = (statusCode, data) => ({ statusCode, headers, body: JSON.stringify(data) });

  try {
    const requested = event?.queryStringParameters?.date || chicagoDate();
    if (!validDate(requested)) {
      return send(400, { error: 'date must be YYYY-MM-DD', events: [] });
    }

    const url = `${REPO_RAW_BASE}/${requested}_run_environment.json`;
    const response = await fetch(url, {
      headers: {
        accept: 'application/json',
        'user-agent': 'MLB-I2-Netlify-Run-Environment/2.0'
      }
    });

    if (response.status === 404) {
      return send(404, {
        error: `No committed SportsGameOdds run-environment artifact for ${requested}`,
        date: requested,
        source: EXPECTED_SOURCE,
        scope: EXPECTED_SCOPE,
        events: []
      });
    }

    const text = await response.text();
    if (!response.ok) {
      return send(502, {
        error: `Unable to retrieve committed run-environment artifact (HTTP ${response.status})`,
        date: requested,
        events: []
      });
    }

    let raw;
    try {
      raw = JSON.parse(text);
    } catch {
      return send(502, { error: 'Committed run-environment artifact is not valid JSON', date: requested, events: [] });
    }

    const sanitized = sanitizeArtifact(raw);
    return send(200, {
      ...sanitized,
      servedAt: new Date().toISOString(),
      delivery: 'GITHUB_COMMITTED_SANITIZED_ARTIFACT',
      upstreamArtifact: `data/runtime/i2/${requested}_run_environment.json`
    });
  } catch (error) {
    return send(502, { error: String(error?.message || error), events: [] });
  }
};

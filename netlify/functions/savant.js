const memoryCache = new Map();

const SAVANT = 'https://baseballsavant.mlb.com';
const UA = 'MLB-Live-Command-Center/11.0 (personal analytics dashboard)';

function send(statusCode, data, extraHeaders = {}) {
  return {
    statusCode,
    headers: {
      'content-type': 'application/json; charset=utf-8',
      'access-control-allow-origin': '*',
      ...extraHeaders
    },
    body: JSON.stringify(data)
  };
}

function parseCSV(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i], next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') { field += '"'; i++; }
      else if (ch === '"') quoted = false;
      else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(field); field = ''; }
    else if (ch === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += ch;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  if (!rows.length) return [];
  const headers = rows.shift().map(x => String(x || '').trim());
  return rows.filter(r => r.some(v => v !== '')).map(r => Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ''])));
}

function numberOrNull(value) {
  if (value === '' || value === null || value === undefined) return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : value;
}

function normalizeRows(rows) {
  return rows.map(row => Object.fromEntries(Object.entries(row).map(([k, v]) => [k, numberOrNull(v)])));
}

async function cachedFetch(url, ttlMs, accept = 'text/csv,*/*') {
  const hit = memoryCache.get(url);
  if (hit && Date.now() - hit.savedAt < ttlMs) return { ...hit, cache: 'memory' };
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { accept, 'user-agent': UA, referer: `${SAVANT}/statcast_search` }
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`Baseball Savant returned ${response.status}`);
    const value = { savedAt: Date.now(), text, status: response.status };
    memoryCache.set(url, value);
    return { ...value, cache: 'origin' };
  } finally { clearTimeout(timeout); }
}

function parseEmbeddedParkData(html) {
  // Baseball Savant's park-factor leaderboard embeds its table payload in a
  // JavaScript assignment named `data`. Keep this parser deliberately narrow:
  // it reads only venue_name + index_woba and never feeds the prediction engine.
  const patterns = [
    /(?:^|\n)\s*data\s*=\s*(\[[^\n]*\])\s*;/,
    /\bdata\s*=\s*(\[[\s\S]*?\])\s*;\s*(?:\n|$)/
  ];
  for (const pattern of patterns) {
    const m = html.match(pattern);
    if (!m) continue;
    try {
      const parsed = JSON.parse(m[1]);
      if (Array.isArray(parsed)) return parsed;
    } catch (_) {}
  }
  throw new Error('Unable to parse Baseball Savant park-factor payload');
}

function parkPercentiles(rows) {
  const clean = rows.map(row => ({
    venue: String(row.venue_name || row.venue || '').trim(),
    factor: Number(row.index_woba ?? row.index_wOBA ?? row.park_factor)
  })).filter(x => x.venue && Number.isFinite(x.factor));
  if (!clean.length) return [];
  const values = clean.map(x => x.factor).sort((a, b) => a - b);
  const n = values.length;
  return clean.map(x => {
    const less = values.filter(v => v < x.factor).length;
    const equal = values.filter(v => v === x.factor).length;
    // Midrank percentile is stable for tied rounded park factors.
    const percentile = Math.round(100 * (less + 0.5 * equal) / n);
    return { venue: x.venue, percentile, factor: x.factor };
  }).sort((a, b) => b.factor - a.factor || a.venue.localeCompare(b.venue));
}

exports.handler = async function(event) {
  try {
    const q = event.queryStringParameters || {};
    if (q.type === 'leaderboard') {
      const season = /^\d{4}$/.test(q.season || '') ? q.season : String(new Date().getFullYear());
      const group = q.group === 'pitcher' ? 'pitcher' : 'batter';
      const min = /^\d+$/.test(q.min || '') ? q.min : '1';
      const url = `${SAVANT}/leaderboard/statcast?type=${group}&year=${season}&position=&team=&min=${min}&csv=true`;
      const result = await cachedFetch(url, 6 * 60 * 60 * 1000);
      const rows = normalizeRows(parseCSV(result.text));
      return send(200, { source: 'Baseball Savant', type: 'leaderboard', group, season, fetchedAt: result.savedAt, cache: result.cache, rows }, {
        'cache-control': 'public, max-age=900, s-maxage=21600, stale-while-revalidate=86400'
      });
    }

    if (q.type === 'parkFactors') {
      const year = /^\d{4}$/.test(q.year || '') ? q.year : String(new Date().getFullYear());
      const url = `${SAVANT}/leaderboard/statcast-park-factors?type=year&year=${encodeURIComponent(year)}&batSide=&stat=index_wOBA&condition=All&rolling=3&parks=mlb`;
      const result = await cachedFetch(url, 6 * 60 * 60 * 1000, 'text/html,*/*');
      const raw = parseEmbeddedParkData(result.text);
      const parks = parkPercentiles(raw);
      return send(200, {
        source: 'Baseball Savant Statcast Park Factors',
        type: 'parkFactors',
        year,
        rollingYears: 3,
        metric: 'index_wOBA',
        displayOnly: true,
        fetchedAt: result.savedAt,
        cache: result.cache,
        parks
      }, {
        'cache-control': 'public, max-age=900, s-maxage=21600, stale-while-revalidate=86400'
      });
    }

    if (q.type === 'game') {
      if (!/^\d+$/.test(q.gamePk || '')) return send(400, { error: 'Invalid gamePk' });
      const url = `${SAVANT}/statcast_search/csv?all=true&type=details&game_pk=${encodeURIComponent(q.gamePk)}`;
      const result = await cachedFetch(url, 12000);
      const rows = normalizeRows(parseCSV(result.text));
      rows.sort((a, b) => Number(b.at_bat_number || 0) - Number(a.at_bat_number || 0) || Number(b.pitch_number || 0) - Number(a.pitch_number || 0));
      return send(200, { source: 'Baseball Savant', type: 'game', gamePk: q.gamePk, fetchedAt: result.savedAt, cache: result.cache, rowCount: rows.length, rows }, {
        'cache-control': 'public, max-age=5, s-maxage=12, stale-while-revalidate=60'
      });
    }

    return send(400, { error: 'Use type=leaderboard, type=parkFactors, or type=game' });
  } catch (error) {
    return send(502, { error: 'Unable to reach Baseball Savant', detail: String(error && error.message ? error.message : error) }, {
      'cache-control': 'no-store'
    });
  }
};

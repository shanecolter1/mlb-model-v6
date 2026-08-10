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

async function cachedFetch(url, ttlMs) {
  const hit = memoryCache.get(url);
  if (hit && Date.now() - hit.savedAt < ttlMs) return { ...hit, cache: 'memory' };
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
      headers: { accept: 'text/csv,*/*', 'user-agent': UA, referer: `${SAVANT}/statcast_search` }
    });
    const text = await response.text();
    if (!response.ok) throw new Error(`Baseball Savant returned ${response.status}`);
    const value = { savedAt: Date.now(), text, status: response.status };
    memoryCache.set(url, value);
    return { ...value, cache: 'origin' };
  } finally { clearTimeout(timeout); }
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

    return send(400, { error: 'Use type=leaderboard or type=game' });
  } catch (error) {
    return send(502, { error: 'Unable to reach Baseball Savant', detail: String(error && error.message ? error.message : error) }, {
      'cache-control': 'no-store'
    });
  }
};

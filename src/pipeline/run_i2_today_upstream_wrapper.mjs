import fs from 'node:fs';

const upstreamBase = String(process.env.MLB_OTHER_MODEL_BASE_URL || '').trim().replace(/\/$/, '');
if (!upstreamBase) {
  throw new Error('MLB_OTHER_MODEL_BASE_URL is required: user-owned upstream MLB data must be queried first');
}

const originalFetch = globalThis.fetch.bind(globalThis);
const audit = {
  rule: 'USER_DATA_FIRST',
  upstreamBase,
  upstreamAttempts: 0,
  upstreamSuccesses: 0,
  officialFallbacks: 0,
  fallbackEvents: [],
  unmappedOfficialCalls: 0,
};

async function upstreamFetch(type, params = {}) {
  const u = new URL(`${upstreamBase}/.netlify/functions/mlb`);
  u.searchParams.set('type', type);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') u.searchParams.set(k, String(v));
  }
  audit.upstreamAttempts += 1;
  const response = await originalFetch(u, {
    headers: { accept: 'application/json', 'user-agent': 'MLB-I2-Upstream-First/0.3' },
  });
  if (!response.ok) throw new Error(`upstream ${type} ${response.status} ${response.statusText}`);
  audit.upstreamSuccesses += 1;
  return response;
}

function fallback(reason, url) {
  audit.officialFallbacks += 1;
  audit.fallbackEvents.push({ reason: String(reason), url: String(url).replace(/([?&](?:apiKey|key|token)=)[^&]+/gi, '$1REDACTED') });
}

globalThis.fetch = async function upstreamFirstFetch(input, init = {}) {
  const raw = typeof input === 'string' || input instanceof URL ? String(input) : String(input?.url || input);
  let u;
  try { u = new URL(raw); } catch { return originalFetch(input, init); }

  if (u.hostname !== 'statsapi.mlb.com') return originalFetch(input, init);

  try {
    if (u.pathname === '/api/v1/schedule') {
      const date = u.searchParams.get('date') || process.env.I2_DATE || new Date().toISOString().slice(0, 10);
      return await upstreamFetch('schedule', { date });
    }

    const feedMatch = u.pathname.match(/^\/api\/v1\.1\/game\/(\d+)\/feed\/live$/);
    if (feedMatch) {
      return await upstreamFetch('feed', { gamePk: feedMatch[1] });
    }

    const statsMatch = u.pathname.match(/^\/api\/v1\/people\/(\d+)\/stats$/);
    if (statsMatch) {
      const personId = statsMatch[1];
      const group = u.searchParams.get('group') || 'hitting';
      const season = u.searchParams.get('season') || String(new Date().getUTCFullYear());
      const response = await upstreamFetch('peopleStats', { ids: personId, season });
      const payload = await response.json();
      const stat = payload?.players?.[personId]?.[group] || {};
      return new Response(JSON.stringify({ stats: [{ splits: [{ stat }] }] }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
  } catch (error) {
    fallback(error, raw);
    return originalFetch(input, init);
  }

  audit.unmappedOfficialCalls += 1;
  fallback('unmapped statsapi.mlb.com request', raw);
  return originalFetch(input, init);
};

await import('./run_i2_today.mjs');

const output = process.env.I2_OUTPUT || `data/runtime/i2/${process.env.I2_DATE || new Date().toISOString().slice(0, 10)}_frozen_predictions.json`;
if (fs.existsSync(output)) {
  const payload = JSON.parse(fs.readFileSync(output, 'utf8'));
  payload.sourcePriorityAudit = audit;
  payload.primaryBaseballSource = 'user-owned MLB Live Command Center upstream data layer';
  payload.fallbackBaseballSource = 'official MLB Stats API only when the upstream request fails';
  fs.writeFileSync(output, JSON.stringify(payload, null, 2));
}

console.log(JSON.stringify({ sourcePriorityAudit: audit }, null, 2));

import fs from 'node:fs';
import path from 'node:path';
import {
  SPORTSBOOK_DATA_SOURCE,
  MARKET_ISOLATION,
  fetchMlbDraftKingsFullGameTotals,
  assertPreFreezeIsolation,
} from '../market/sportsbook_data_source.mjs';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const OUTPUT = process.env.I2_RUN_ENVIRONMENT || `data/runtime/i2/${DATE}_run_environment.json`;
const ATTEMPTS = Number(process.env.I2_RUN_ENVIRONMENT_ATTEMPTS || 3);
const RETRY_MS = Number(process.env.I2_RUN_ENVIRONMENT_RETRY_MS || 3000);

function loadExisting() {
  if (!fs.existsSync(OUTPUT)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(OUTPUT, 'utf8'));
    if (parsed?.scope !== MARKET_ISOLATION.preFreeze.scope) return null;
    if (!Array.isArray(parsed?.events) || parsed.events.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let events = null;
let lastError = null;

for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
  try {
    events = await fetchMlbDraftKingsFullGameTotals();
    if (!Array.isArray(events) || events.length === 0) {
      throw new Error('The Odds API returned no DraftKings MLB full-game totals');
    }
    for (const event of events) assertPreFreezeIsolation(event);
    break;
  } catch (error) {
    lastError = error;
    if (attempt < ATTEMPTS) {
      console.warn(`[I2 run environment] The Odds API attempt ${attempt}/${ATTEMPTS} failed: ${String(error?.message || error)}; retrying in ${RETRY_MS}ms`);
      await sleep(RETRY_MS);
    }
  }
}

const existing = loadExisting();
if (!events) {
  if (existing) {
    console.warn(`[I2 run environment] The Odds API unavailable; using validated locked existing artifact ${OUTPUT}`);
    console.log(JSON.stringify({
      output: OUTPUT,
      events: existing.events.length,
      scope: existing.scope,
      source: SPORTSBOOK_DATA_SOURCE.provider,
      fallback: 'VALIDATED_EXISTING_LOCKED_ARTIFACT',
    }, null, 2));
    process.exit(0);
  }
  throw lastError || new Error('Unable to retrieve I2 run environment from The Odds API');
}

const priorById = new Map((existing?.events || []).filter(x => x.eventId).map(x => [String(x.eventId), x]));
const now = new Date().toISOString();
const merged = [];

for (const event of events) {
  const id = String(event.eventId || '');
  if (!id) continue;
  const previous = priorById.get(id);
  if (previous) {
    merged.push({
      ...previous,
      lastSeenAt: now,
      latestObservedTotal: Number(event.fullGameTotal),
      latestBookmakerUpdate: event.lastUpdate || null,
    });
    priorById.delete(id);
  } else {
    merged.push({
      eventId: id,
      commenceTime: event.commenceTime,
      awayTeam: event.awayTeam,
      homeTeam: event.homeTeam,
      fullGameTotal: Number(event.fullGameTotal),
      bookmaker: 'draftkings',
      firstSeenAt: now,
      lastSeenAt: now,
      latestObservedTotal: Number(event.fullGameTotal),
      latestBookmakerUpdate: event.lastUpdate || null,
    });
  }
}

for (const stale of priorById.values()) merged.push(stale);
merged.sort((a, b) => Date.parse(a.commenceTime || 0) - Date.parse(b.commenceTime || 0));

const out = {
  date: DATE,
  capturedAt: now,
  source: SPORTSBOOK_DATA_SOURCE.provider,
  sourcePolicyVersion: SPORTSBOOK_DATA_SOURCE.policyVersion,
  scope: MARKET_ISOLATION.preFreeze.scope,
  totalDefinition: 'FIRST_OBSERVED_DRAFTKINGS_PREGAME_TOTAL',
  marketIsolation: {
    phase: 'PRE_FREEZE',
    allowedBookmaker: MARKET_ISOLATION.preFreeze.allowedBookmaker,
    allowedMarket: MARKET_ISOLATION.preFreeze.allowedMarket,
    pricesExposedToPredictionEngine: false,
  },
  note: 'The Odds API is the canonical sportsbook-data source. Before prediction freeze, only the DraftKings full-game total point is retained. The first observed total is locked for conditioning; latestObservedTotal is audit-only.',
  events: merged,
};

fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
fs.writeFileSync(OUTPUT, JSON.stringify(out, null, 2));
console.log(JSON.stringify({ output: OUTPUT, events: merged.length, scope: out.scope, source: out.source }, null, 2));

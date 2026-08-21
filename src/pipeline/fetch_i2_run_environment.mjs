import fs from 'node:fs';
import path from 'node:path';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const UPSTREAM = String(process.env.MLB_OTHER_MODEL_BASE_URL || '').trim().replace(/\/$/, '');
const OUTPUT = process.env.I2_RUN_ENVIRONMENT || `data/runtime/i2/${DATE}_run_environment.json`;
const ATTEMPTS = Number(process.env.I2_RUN_ENVIRONMENT_ATTEMPTS || 12);
const RETRY_MS = Number(process.env.I2_RUN_ENVIRONMENT_RETRY_MS || 10000);

if (!UPSTREAM) throw new Error('MLB_OTHER_MODEL_BASE_URL is required for the isolated I2 run-environment feed');

function loadExisting() {
  if (!fs.existsSync(OUTPUT)) return null;
  try {
    const parsed = JSON.parse(fs.readFileSync(OUTPUT, 'utf8'));
    if (parsed?.scope !== 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES') return null;
    if (!Array.isArray(parsed?.events) || parsed.events.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const url = `${UPSTREAM}/.netlify/functions/i2-run-environment`;
let payload = null;
let lastError = null;
for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
  try {
    const response = await fetch(url, {
      headers: { accept: 'application/json', 'user-agent': 'MLB-I2-Run-Environment-Capture/0.2' }
    });
    if (!response.ok) throw new Error(`I2 run-environment endpoint ${response.status} ${response.statusText}`);
    payload = await response.json();
    if (payload?.scope !== 'FULL_GAME_TOTAL_POINT_ONLY_NO_PRICES') {
      throw new Error(`Unexpected run-environment scope: ${payload?.scope || 'missing'}`);
    }
    break;
  } catch (error) {
    lastError = error;
    if (attempt < ATTEMPTS) {
      console.warn(`[I2 run environment] attempt ${attempt}/${ATTEMPTS} failed: ${String(error?.message || error)}; retrying in ${RETRY_MS}ms`);
      await sleep(RETRY_MS);
    }
  }
}

const existing = loadExisting();
if (!payload) {
  if (existing) {
    console.warn(`[I2 run environment] endpoint unavailable; using validated existing artifact ${OUTPUT}`);
    console.log(JSON.stringify({ output: OUTPUT, events: existing.events.length, scope: existing.scope, fallback: 'VALIDATED_EXISTING_ARTIFACT' }, null, 2));
    process.exit(0);
  }
  throw lastError || new Error('Unable to retrieve I2 run environment');
}

const priorById = new Map((existing?.events || []).filter(x => x.eventId).map(x => [String(x.eventId), x]));
const now = payload.capturedAt || new Date().toISOString();
const merged = [];

for (const event of payload.events || []) {
  const id = String(event.eventId || '');
  if (!id) continue;
  const previous = priorById.get(id);
  if (previous) {
    merged.push({
      ...previous,
      lastSeenAt: now,
      latestObservedTotal: event.fullGameTotal,
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
      bookmaker: event.bookmaker || 'draftkings',
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
  source: payload.source,
  scope: payload.scope,
  totalDefinition: 'FIRST_OBSERVED_DRAFTKINGS_PREGAME_TOTAL',
  note: 'The first observed total is locked for conditioning on subsequent reruns. latestObservedTotal is audit-only and does not replace the locked value.',
  events: merged,
};

fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
fs.writeFileSync(OUTPUT, JSON.stringify(out, null, 2));
console.log(JSON.stringify({ output: OUTPUT, events: merged.length, scope: out.scope }, null, 2));

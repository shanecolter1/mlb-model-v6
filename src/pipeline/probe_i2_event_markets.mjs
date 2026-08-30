import fs from 'node:fs/promises';
import path from 'node:path';
import { fetchOddsApiJson, unwrapOddsEvents, extractBookMarketEntries } from '../market/sportsbook_data_source.mjs';

const date = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const frozenPath = process.env.I2_OUTPUT || `data/runtime/i2/${date}_frozen_predictions.json`;
const outPath = process.env.I2_MARKET_CATALOG || `data/runtime/i2/${date}_event_market_catalog.json`;

const frozen = JSON.parse(await fs.readFile(frozenPath, 'utf8'));
const ranked = Array.isArray(frozen.ranking) ? frozen.ranking : [];
const start = `${date}T00:00:00Z`;
const end = `${date}T23:59:59Z`;

const payload = await fetchOddsApiJson('/odds/', {
  sport_key: 'baseball_mlb',
  commenceTimeFrom: start,
  commenceTimeTo: end,
  oddsFormat: 'american',
});
const events = unwrapOddsEvents(payload);

const normalize = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
function scoreMatch(row, ev) {
  const text = normalize(row.matchup);
  const away = normalize(ev.away_team);
  const home = normalize(ev.home_team);
  let score = 0;
  if (away && text.includes(away)) score += 2;
  if (home && text.includes(home)) score += 2;
  const dt = row.gameDate ? new Date(row.gameDate).getTime() : NaN;
  const et = (ev.start_time || ev.commence_time) ? new Date(ev.start_time || ev.commence_time).getTime() : NaN;
  if (Number.isFinite(dt) && Number.isFinite(et) && Math.abs(dt - et) <= 15 * 60 * 1000) score += 1;
  return score;
}

const focusBooks = new Set(['fanduel','draftkings','hardrockbet','hardrock','fanatics','caesars','williamhill_us','betmgm','kalshi','polymarket','pinnacle']);
const results = [];
for (const row of ranked) {
  const best = events.map(ev => ({ ev, score: scoreMatch(row, ev) })).sort((a, b) => b.score - a.score)[0];
  if (!best || best.score < 4) {
    results.push({ matchup: row.matchup, gamePk: row.gamePk, status: 'EVENT_NOT_MATCHED' });
    continue;
  }
  const ev = best.ev;
  const entries = extractBookMarketEntries(ev);
  const focused = entries.filter(x => focusBooks.has(x.key));
  const distinctMarketsByBook = {};
  for (const entry of focused) {
    if (!distinctMarketsByBook[entry.key]) distinctMarketsByBook[entry.key] = [];
    if (!distinctMarketsByBook[entry.key].includes(entry.market)) distinctMarketsByBook[entry.key].push(entry.market);
  }
  results.push({
    matchup: row.matchup,
    gamePk: row.gamePk,
    eventId: String(ev.event_id || ev.id || ''),
    commenceTime: ev.start_time || ev.commence_time || null,
    status: 'OK',
    marketsByBook: distinctMarketsByBook,
    focusedEntries: focused.map(x => x.raw),
  });
}

const output = {
  date,
  generatedAt: new Date().toISOString(),
  source: 'THEODDSAPI_COM_LIVE_ODDS_EVENT_PROBE',
  results,
};
await fs.mkdir(path.dirname(outPath), { recursive: true });
await fs.writeFile(outPath, `${JSON.stringify(output, null, 2)}\n`);
console.log(JSON.stringify({ outPath, games: results.length, eventCount: events.length, statuses: results.reduce((a, x) => ((a[x.status] = (a[x.status] || 0) + 1), a), {}) }, null, 2));
import fs from 'node:fs/promises';
import path from 'node:path';
import { fetchOddsApiJson } from '../market/sportsbook_data_source.mjs';

const date = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const frozenPath = process.env.I2_OUTPUT || `data/runtime/i2/${date}_frozen_predictions.json`;
const outPath = process.env.I2_MARKET_CATALOG || `data/runtime/i2/${date}_event_market_catalog.json`;

const frozen = JSON.parse(await fs.readFile(frozenPath, 'utf8'));
const ranked = Array.isArray(frozen.ranking) ? frozen.ranking : [];

const normalize = s => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
const candidates = await fetchOddsApiJson('/sports/baseball_mlb/odds', {
  regions: 'us',
  markets: 'h2h',
  bookmakers: 'fanduel',
  oddsFormat: 'american',
  dateFormat: 'iso',
});

function scoreMatch(row, ev) {
  const text = normalize(`${row.matchup} ${row.awayTeam || ''} ${row.homeTeam || ''}`);
  const away = normalize(ev.away_team);
  const home = normalize(ev.home_team);
  let score = 0;
  if (away && text.includes(away)) score += 2;
  if (home && text.includes(home)) score += 2;
  const dt = row.gameDate ? new Date(row.gameDate).getTime() : NaN;
  const et = ev.commence_time ? new Date(ev.commence_time).getTime() : NaN;
  if (Number.isFinite(dt) && Number.isFinite(et) && Math.abs(dt - et) <= 15 * 60 * 1000) score += 1;
  return score;
}

const results = [];
for (const row of ranked) {
  const best = (candidates || []).map(ev => ({ ev, score: scoreMatch(row, ev) })).sort((a, b) => b.score - a.score)[0];
  if (!best || best.score < 4) {
    results.push({ matchup: row.matchup, gamePk: row.gamePk, status: 'EVENT_NOT_MATCHED' });
    continue;
  }

  const eventId = String(best.ev.id);
  try {
    const catalog = await fetchOddsApiJson(`/sports/baseball_mlb/events/${eventId}/markets`, {
      regions: 'us',
      bookmakers: 'fanduel,draftkings,hardrockbet,fanatics,williamhill_us,betmgm',
      oddsFormat: 'american',
      dateFormat: 'iso',
    });
    results.push({
      matchup: row.matchup,
      gamePk: row.gamePk,
      eventId,
      commenceTime: best.ev.commence_time,
      status: 'OK',
      catalog,
    });
  } catch (error) {
    results.push({
      matchup: row.matchup,
      gamePk: row.gamePk,
      eventId,
      commenceTime: best.ev.commence_time,
      status: 'ERROR',
      error: String(error?.message || error),
    });
  }
}

const output = {
  date,
  generatedAt: new Date().toISOString(),
  source: 'THE_ODDS_API_EVENT_MARKET_CATALOG',
  bookmakerFocus: ['fanduel', 'draftkings', 'hardrockbet', 'fanatics', 'williamhill_us', 'betmgm'],
  results,
};

await fs.mkdir(path.dirname(outPath), { recursive: true });
await fs.writeFile(outPath, `${JSON.stringify(output, null, 2)}\n`);
console.log(JSON.stringify({ outPath, games: results.length, statuses: results.reduce((a, x) => ((a[x.status] = (a[x.status] || 0) + 1), a), {}) }, null, 2));
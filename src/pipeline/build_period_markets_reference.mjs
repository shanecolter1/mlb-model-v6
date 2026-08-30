import fs from 'node:fs/promises';
import path from 'node:path';

const API_BASE = 'https://api.theoddsapi.com';
const apiKey = String(process.env.ODDS_API_KEY || '').trim();
if (!apiKey) throw new Error('ODDS_API_KEY is required');

const date = process.env.I2_DATE || new Date().toISOString().slice(0, 10);
const rawPath = process.env.PERIOD_RAW || `data/runtime/i2/${date}_period_markets_raw.json`;
const mdPath = process.env.PERIOD_MD || `docs/period_markets/${date}_mlb_period_markets.md`;
const csvPath = process.env.PERIOD_CSV || `docs/period_markets/${date}_mlb_period_markets.csv`;

async function getJson(endpoint, params = {}) {
  const url = new URL(endpoint, API_BASE);
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  const r = await fetch(url, { headers: { accept: 'application/json', 'x-api-key': apiKey, 'user-agent': 'MLB-I2-Period-Markets-Reference/1.0' } });
  const text = await r.text();
  let body;
  try { body = JSON.parse(text); } catch { body = { raw: text }; }
  if (!r.ok) throw new Error(`${url.pathname} ${r.status} ${r.statusText}: ${JSON.stringify(body).slice(0, 600)}`);
  return body;
}

const from = `${date}T00:00:00Z`;
const to = `${date}T23:59:59Z`;
const eventsResp = await getJson('/odds/', { sport_key: 'baseball_mlb', markets: 'h2h', commenceTimeFrom: from, commenceTimeTo: to });
const events = Array.isArray(eventsResp?.data) ? eventsResp.data : [];

const probes = [];
for (const ev of events) {
  const eventId = ev.event_id;
  try {
    const resp = await getJson('/period-markets/', { sport_key: 'baseball_mlb', event_id: eventId });
    probes.push({ event_id: eventId, home_team: ev.home_team, away_team: ev.away_team, start_time: ev.start_time, response: resp });
  } catch (error) {
    probes.push({ event_id: eventId, home_team: ev.home_team, away_team: ev.away_team, start_time: ev.start_time, error: String(error?.message || error) });
  }
}

const rows = [];
for (const probe of probes) {
  const data = Array.isArray(probe.response?.data) ? probe.response.data : [];
  for (const item of data) {
    for (const market of item.markets || []) {
      for (const book of market.books || []) {
        const outcomes = Array.isArray(book.outcomes) ? book.outcomes : [];
        rows.push({
          event_id: item.event_id || probe.event_id,
          matchup: `${item.away_team || probe.away_team} @ ${item.home_team || probe.home_team}`,
          start_time: item.start_time || probe.start_time,
          market: market.market,
          sportsbook: book.book,
          outcomes,
        });
      }
    }
  }
}

const books = [...new Set(rows.map(r => r.sportsbook))].sort();
const markets = [...new Set(rows.map(r => r.market))].sort();
const availability = new Map();
for (const r of rows) {
  const key = `${r.sportsbook}|||${r.market}`;
  availability.set(key, (availability.get(key) || 0) + 1);
}

const esc = v => String(v ?? '').replaceAll('|', '\\|');
const md = [];
md.push(`# MLB Business Period Markets Reference — ${date}`);
md.push('');
md.push(`Generated: ${new Date().toISOString()}`);
md.push('');
md.push('Source: TheOddsAPI Business `GET /period-markets/?sport_key=baseball_mlb&event_id=...`, queried separately for every MLB event returned for the date. This is post-freeze market enumeration only and is not used by the I2 prediction engine.');
md.push('');
md.push(`Events probed: **${events.length}**  |  Period-market rows returned: **${rows.length}**  |  Sportsbooks returned: **${books.length}**  |  Distinct market keys: **${markets.length}**`);
md.push('');
md.push('## Sportsbook × market availability');
md.push('');
if (books.length && markets.length) {
  md.push(`| Sportsbook | ${markets.map(esc).join(' | ')} |`);
  md.push(`|---|${markets.map(() => '---:').join('|')}|`);
  for (const book of books) md.push(`| ${esc(book)} | ${markets.map(m => availability.get(`${book}|||${m}`) || 0).join(' | ')} |`);
} else {
  md.push('_No period-market rows were returned at query time._');
}
md.push('');
md.push('Counts are the number of today\'s event responses in which that sportsbook/market combination appeared.');
md.push('');
md.push('## Available bets by sportsbook');
md.push('');
for (const book of books) {
  md.push(`### ${book}`);
  const bookRows = rows.filter(r => r.sportsbook === book).sort((a,b) => String(a.start_time).localeCompare(String(b.start_time)) || a.market.localeCompare(b.market));
  md.push('| Matchup | Market | Outcomes | Updated/Start |');
  md.push('|---|---|---|---|');
  for (const r of bookRows) {
    const outcomeText = r.outcomes.map(o => [o.name, o.point !== undefined ? `line ${o.point}` : null, o.price !== undefined ? `price ${o.price}` : null].filter(Boolean).join(' ')).join('; ');
    md.push(`| ${esc(r.matchup)} | ${esc(r.market)} | ${esc(outcomeText)} | ${esc(r.start_time)} |`);
  }
  md.push('');
}
md.push('## Probe audit');
md.push('');
md.push('| Matchup | Event ID | Status | Active-window events | Next event |');
md.push('|---|---|---|---:|---|');
for (const p of probes) {
  const aw = p.response?.active_window || {};
  md.push(`| ${esc(`${p.away_team} @ ${p.home_team}`)} | ${esc(p.event_id)} | ${p.error ? 'ERROR' : 'OK'} | ${esc(aw.events_in_window ?? '')} | ${esc(aw.next_event_at ?? '')} |`);
}

const csvHeader = ['event_id','matchup','start_time','sportsbook','market','outcome_name','point','price'];
const csvLines = [csvHeader.join(',')];
const q = v => `"${String(v ?? '').replaceAll('"','""')}"`;
for (const r of rows) {
  for (const o of r.outcomes) csvLines.push([r.event_id,r.matchup,r.start_time,r.sportsbook,r.market,o.name,o.point ?? '',o.price ?? ''].map(q).join(','));
}

const raw = {
  date,
  generatedAt: new Date().toISOString(),
  endpoint: '/period-markets/',
  queryMethod: 'EVENT_BY_EVENT',
  sport_key: 'baseball_mlb',
  eventCount: events.length,
  events: events.map(e => ({ event_id: e.event_id, home_team: e.home_team, away_team: e.away_team, start_time: e.start_time })),
  probes,
  normalized: { books, markets, rows },
};

for (const p of [rawPath, mdPath, csvPath]) await fs.mkdir(path.dirname(p), { recursive: true });
await fs.writeFile(rawPath, `${JSON.stringify(raw, null, 2)}\n`);
await fs.writeFile(mdPath, `${md.join('\n')}\n`);
await fs.writeFile(csvPath, `${csvLines.join('\n')}\n`);
console.log(JSON.stringify({ date, events: events.length, periodRows: rows.length, books, markets, rawPath, mdPath, csvPath }, null, 2));

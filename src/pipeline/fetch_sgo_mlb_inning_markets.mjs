import fs from 'node:fs/promises';
import path from 'node:path';
import {
  ROOKIE_TARGET_BOOKMAKERS,
  buildMlbInningOddIds,
  fetchMlbInningEvents,
  fetchMlbMarketSupport,
  filterEventsByLocalDate,
} from '../market/sportsgameodds_data_source.mjs';

const date = String(process.env.I2_DATE || new Date().toISOString().slice(0, 10));
const timeZone = String(process.env.I2_TIME_ZONE || 'America/Chicago');
const frozenPath = String(process.env.FROZEN_PROJECTION_PATH || `data/runtime/i2/${date}_frozen_predictions.json`);
const outputPath = String(process.env.SGO_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_inning_markets.json`);
const csvPath = String(process.env.SGO_CSV || `docs/inning_markets/${date}_sportsgameodds_inning_markets.csv`);
const supportPath = String(process.env.SGO_SUPPORT_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_market_support.json`);
const includeOpenCloseOdds = /^true$/i.test(String(process.env.SGO_INCLUDE_OPEN_CLOSE || 'false'));
const bookmakerIDs = String(process.env.SGO_BOOKMAKERS || ROOKIE_TARGET_BOOKMAKERS.join(','))
  .split(',').map(x => x.trim().toLowerCase()).filter(Boolean);

let frozen;
try {
  frozen = JSON.parse(await fs.readFile(frozenPath, 'utf8'));
} catch (error) {
  throw new Error(`Frozen projection artifact is required before sportsbook retrieval: ${frozenPath} (${error.message})`);
}

const frozenAt = frozen.generatedAt || frozen.generated_at || frozen.createdAt || frozen.timestamp || frozen.cutoff;
if (!frozenAt || Number.isNaN(Date.parse(frozenAt))) {
  throw new Error(`Frozen projection artifact does not expose a valid generated/frozen timestamp: ${frozenPath}`);
}

const freezeContext = {
  projectionFrozen: true,
  frozenAt,
  artifact: frozenPath,
  model: frozen.model || null,
  date: frozen.date || date,
};

const oddIDs = buildMlbInningOddIds();
const [support, feed] = await Promise.all([
  fetchMlbMarketSupport({ bookmakerIDs, oddIDs }),
  fetchMlbInningEvents({ freezeContext, bookmakerIDs, includeOpenCloseOdds }),
]);

const events = filterEventsByLocalDate(feed.events, date, timeZone);
const rows = [];
for (const event of events) {
  const matchup = `${event.away.name || event.away.short || 'Away'} @ ${event.home.name || event.home.short || 'Home'}`;
  for (const market of event.markets) {
    for (const price of market.prices) {
      if (!price.available || price.americanOdds === null) continue;
      rows.push({
        provider: feed.provider,
        eventID: event.eventID,
        startTime: event.startTime,
        matchup,
        awayTeam: event.away.name,
        homeTeam: event.home.name,
        inning: market.inning,
        segment: market.segment,
        marketType: market.marketType,
        side: market.side,
        teamSide: market.teamSide || '',
        oddID: market.oddID,
        bookmakerID: price.bookmakerID,
        line: price.line ?? market.providerConsensusLine ?? market.providerFairLine ?? '',
        americanOdds: price.americanOdds,
        lastUpdatedAt: price.lastUpdatedAt || '',
        deeplink: price.deeplink || '',
        providerFairOdds: market.providerFairOdds ?? '',
        providerConsensusOdds: market.providerConsensusOdds ?? '',
      });
    }
  }
}

const coverage = {
  eventCount: events.length,
  priceRows: rows.length,
  bookmakers: [...new Set(rows.map(x => x.bookmakerID))].sort(),
  marketTypes: [...new Set(rows.map(x => x.marketType))].sort(),
  innings: [...new Set(rows.map(x => x.inning))].sort((a,b) => a-b),
  fullInningTotalRows: rows.filter(x => x.marketType === 'FULL_INNING_TOTAL').length,
  halfInningTotalRows: rows.filter(x => x.marketType === 'TEAM_HALF_INNING_TOTAL').length,
  threeWayRows: rows.filter(x => x.marketType === 'FULL_INNING_3WAY').length,
};

const output = {
  schemaVersion: '1.0.0',
  generatedAt: new Date().toISOString(),
  date,
  timeZone,
  provider: feed.provider,
  marketIsolation: {
    phase: 'POST_FREEZE_ONLY',
    frozenProjection: freezeContext,
    oddsNotAvailableToPredictionEngine: true,
  },
  requested: { bookmakerIDs, oddIDs, includeOpenCloseOdds },
  coverage,
  events,
  rows,
};

const csvColumns = [
  'provider','eventID','startTime','matchup','awayTeam','homeTeam','inning','segment','marketType','side','teamSide','oddID',
  'bookmakerID','line','americanOdds','lastUpdatedAt','providerFairOdds','providerConsensusOdds','deeplink',
];
const quote = value => `"${String(value ?? '').replaceAll('"','""')}"`;
const csv = [csvColumns.join(','), ...rows.map(row => csvColumns.map(c => quote(row[c])).join(','))].join('\n') + '\n';

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(path.dirname(csvPath), { recursive: true });
await fs.mkdir(path.dirname(supportPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(output, null, 2) + '\n');
await fs.writeFile(csvPath, csv);
await fs.writeFile(supportPath, JSON.stringify({ generatedAt: new Date().toISOString(), date, bookmakerIDs, oddIDs, response: support }, null, 2) + '\n');

console.log(JSON.stringify({ date, frozenAt, outputPath, csvPath, supportPath, coverage }, null, 2));

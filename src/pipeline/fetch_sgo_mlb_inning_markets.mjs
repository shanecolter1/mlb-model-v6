import fs from 'node:fs/promises';
import path from 'node:path';
import {
  ROOKIE_TARGET_BOOKMAKERS,
  buildMlbInningOddIds,
  buildMlbNinthInningCandidateOddIds,
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
const includeAltLines = !/^false$/i.test(String(process.env.SGO_INCLUDE_ALT_LINES || 'true'));
const bookmakerSpec = String(process.env.SGO_BOOKMAKERS ?? ROOKIE_TARGET_BOOKMAKERS.join(',')).trim();
const bookmakerIDs = ['*','all'].includes(bookmakerSpec.toLowerCase())
  ? []
  : bookmakerSpec.split(',').map(x => x.trim().toLowerCase()).filter(Boolean);

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

const guaranteedOddIDs = buildMlbInningOddIds();
const ninthInningCandidates = buildMlbNinthInningCandidateOddIds();
const support = await fetchMlbMarketSupport({ oddIDs: [...guaranteedOddIDs, ...ninthInningCandidates] });
const supportedOddIDs = new Set((Array.isArray(support?.data) ? support.data : []).filter(x => x?.isSupported !== false).map(x => x?.oddID).filter(Boolean));
const activeOddIDs = [...new Set([...guaranteedOddIDs, ...ninthInningCandidates.filter(id => supportedOddIDs.has(id))])];
const i2TargetBookmakers = [
  'draftkings','fanduel','betmgm','caesars','bet365','fanatics',
  'hardrockbet','thescorebet','kalshi','pinnacle','fliff','betrivers'
];
const supportRows = Array.isArray(support?.data) ? support.data : [];
const supportByOddID = new Map(supportRows.map(x => [x?.oddID, x]));
const supports = (oddID, bookmakerID) =>
  supportByOddID.get(oddID)?.support?.MLB?.[bookmakerID]?.supported === true;

const i2BookmakerSupport = i2TargetBookmakers.map(bookmakerID => {
  const cleanOuUnder = supports('points-all-2i-ou-under', bookmakerID);
  const cleanOuOver = supports('points-all-2i-ou-over', bookmakerID);
  const cleanYnNo = supports('points-all-2i-yn-no', bookmakerID);
  const cleanYnYes = supports('points-all-2i-yn-yes', bookmakerID);
  const fallbackDraw = supports('points-all-2i-ml3way-draw', bookmakerID);
  return {
    bookmakerID,
    cleanOuUnder,
    cleanOuOver,
    cleanYnNo,
    cleanYnYes,
    fallbackDraw,
    preferredUnderPath: cleanOuUnder ? 'OU_UNDER_0.5' : (cleanYnNo ? 'ANY_RUNS_NO' : (fallbackDraw ? '3WAY_DRAW_FALLBACK' : null)),
    preferredOverPath: cleanOuOver ? 'OU_OVER_0.5' : (cleanYnYes ? 'ANY_RUNS_YES' : null),
  };
});
const feed = await fetchMlbInningEvents({ freezeContext, bookmakerIDs, includeOpenCloseOdds, includeAltLines, oddIDs: activeOddIDs });

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
        equivalentMarketType: market.equivalentMarketType || '',
        equivalentSide: market.equivalentSide || '',
        equivalentLine: market.equivalentLine ?? '',
        teamSide: market.teamSide || '',
        oddID: market.oddID,
        bookmakerID: price.bookmakerID,
        line: price.line ?? market.equivalentLine ?? market.providerConsensusLine ?? market.providerFairLine ?? '',
        isAlternateLine: price.isAlternateLine === true,
        americanOdds: price.americanOdds,
        lastUpdatedAt: price.lastUpdatedAt || '',
        deeplink: price.deeplink || '',
        providerFairLine: market.providerFairLine ?? '',
        providerConsensusLine: market.providerConsensusLine ?? '',
        providerFairOdds: price.line !== null && price.line === market.providerFairLine ? (market.providerFairOdds ?? '') : '',
        providerConsensusOdds: price.line !== null && price.line === market.providerConsensusLine ? (market.providerConsensusOdds ?? '') : '',
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
  anyRunsRows: rows.filter(x => x.marketType === 'FULL_INNING_ANY_RUNS').length,
  ninthFullInningEnabled: activeOddIDs.some(id => id.includes('-9i-') && (id.includes('-ml3way-') || id.startsWith('points-all-9i-ou-'))),
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
  requested: { bookmakerIDs, guaranteedOddIDs, ninthInningCandidates, activeOddIDs, includeOpenCloseOdds, includeAltLines },
  i2BookmakerSupport,
  coverage,
  events,
  rows,
};

const csvColumns = [
  'provider','eventID','startTime','matchup','awayTeam','homeTeam','inning','segment','marketType','side','equivalentMarketType','equivalentSide','equivalentLine','teamSide','oddID',
  'bookmakerID','line','isAlternateLine','americanOdds','lastUpdatedAt','providerFairLine','providerConsensusLine','providerFairOdds','providerConsensusOdds','deeplink',
];
const quote = value => `"${String(value ?? '').replaceAll('"','""')}"`;
const csv = [csvColumns.join(','), ...rows.map(row => csvColumns.map(c => quote(row[c])).join(','))].join('\n') + '\n';

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(path.dirname(csvPath), { recursive: true });
await fs.mkdir(path.dirname(supportPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(output, null, 2) + '\n');
await fs.writeFile(csvPath, csv);
await fs.writeFile(supportPath, JSON.stringify({ generatedAt: new Date().toISOString(), date, bookmakerIDs, guaranteedOddIDs, ninthInningCandidates, activeOddIDs, i2BookmakerSupport, response: support }, null, 2) + '\n');

console.log(JSON.stringify({ date, frozenAt, outputPath, csvPath, supportPath, coverage, i2BookmakerSupport }, null, 2));

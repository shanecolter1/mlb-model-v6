import fs from 'node:fs/promises';
import path from 'node:path';
import {
  ROOKIE_TARGET_BOOKMAKERS,
  buildMlbInningOddIds,
  buildMlbNinthInningCandidateOddIds,
  fetchMlbMarketSupport,
  fetchMlbSecondInningMarketCatalog,
  filterEventsByLocalDate,
  normalizeSgoEvent,
} from '../market/sportsgameodds_data_source.mjs';
import {
  discoverMlbI2EventLevelMarkets,
  fetchMlbEventsExhaustive,
  mergeI2DiscoveryRows,
  summarizeI2DiscoveryByBookmaker,
} from '../market/sportsgameodds_i2_event_discovery.mjs';

const date = String(process.env.I2_DATE || new Date().toISOString().slice(0, 10));
const timeZone = String(process.env.I2_TIME_ZONE || 'America/Chicago');
const frozenPath = String(process.env.FROZEN_PROJECTION_PATH || `data/runtime/i2/${date}_frozen_predictions.json`);
const outputPath = String(process.env.SGO_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_inning_markets.json`);
const csvPath = String(process.env.SGO_CSV || `docs/inning_markets/${date}_sportsgameodds_inning_markets.csv`);
const supportPath = String(process.env.SGO_SUPPORT_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_market_support.json`);
const exhaustiveCatalogPath = String(process.env.SGO_I2_CATALOG_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_i2_market_catalog.json`);
const exhaustiveRawPath = String(process.env.SGO_I2_RAW_OUTPUT || `data/runtime/i2/${date}_sportsgameodds_i2_raw_events.json`);
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
const exhaustiveCatalog = await fetchMlbSecondInningMarketCatalog();
const exhaustiveOddIDs = [...new Set((exhaustiveCatalog.markets || []).map(x => x?.oddID).filter(Boolean))];

// Post-freeze event-level discovery deliberately avoids oddID, periodID, and betTypeID
// restrictions. This is the only way to see SportsGameOdds Event.type=prop custom
// markets whose settlement definition lives on the Event object instead of /markets.
const caesarsPropSweep = await fetchMlbEventsExhaustive({
  freezeContext,
  type: 'prop',
  bookmakerIDs: ['caesars'],
  oddsPresent: true,
  started: false,
  includeOpenCloseOdds,
  includeAltLines,
  sourceLabel: 'CAESARS_PROP_EVENTS',
});
// A no-type /events query can return only match Events. Run the prop namespace
// explicitly across all bookmakers as a separate exhaustive superset search.
const allBookPropSweep = await fetchMlbEventsExhaustive({
  freezeContext,
  type: 'prop',
  bookmakerIDs: [],
  oddsPresent: true,
  started: false,
  includeOpenCloseOdds,
  includeAltLines,
  sourceLabel: 'ALL_BOOK_PROP_EVENTS',
});
const unrestrictedEventSweep = await fetchMlbEventsExhaustive({
  freezeContext,
  bookmakerIDs: [],
  oddsPresent: true,
  started: false,
  includeOpenCloseOdds,
  includeAltLines,
  sourceLabel: 'UNRESTRICTED_MLB_EVENTS',
});
const caesarsPropRows = discoverMlbI2EventLevelMarkets(caesarsPropSweep.events, { sourceLabel: caesarsPropSweep.sourceLabel });
const allBookPropRows = discoverMlbI2EventLevelMarkets(allBookPropSweep.events, { sourceLabel: allBookPropSweep.sourceLabel });
const unrestrictedDiscoveryRows = discoverMlbI2EventLevelMarkets(unrestrictedEventSweep.events, { sourceLabel: unrestrictedEventSweep.sourceLabel });
const eventLevelDiscoveryRows = mergeI2DiscoveryRows([
  { sourceLabel: caesarsPropSweep.sourceLabel, rows: caesarsPropRows },
  { sourceLabel: allBookPropSweep.sourceLabel, rows: allBookPropRows },
  { sourceLabel: unrestrictedEventSweep.sourceLabel, rows: unrestrictedDiscoveryRows },
]);
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
// Reuse the unrestricted sweep for standard I2 normalization so this audit does
// not pay for a redundant third /events request.
const normalizedAllEvents = unrestrictedEventSweep.events.map(event => normalizeSgoEvent(event, { bookmakerIDs }));
const feed = { provider: 'SPORTSGAMEODDS', events: normalizedAllEvents };
const events = filterEventsByLocalDate(feed.events, date, timeZone);

const eventLevelDiscoveryToday = eventLevelDiscoveryRows.filter(row => {
  const startTime = row.startTime;
  if (!startTime || Number.isNaN(Date.parse(startTime))) return false;
  const probe = { startTime };
  return filterEventsByLocalDate([probe], date, timeZone).length === 1;
});
const eventLevelBookmakerAudit = summarizeI2DiscoveryByBookmaker(eventLevelDiscoveryToday, {
  bookmakerIDs: [
    ...i2TargetBookmakers,
    ...Object.keys((exhaustiveCatalog.markets || []).reduce((acc, market) => {
      for (const bookmakerID of Object.keys(market?.support?.MLB || {})) acc[bookmakerID] = true;
      return acc;
    }, {})),
  ],
});
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

const exhaustiveCatalogByOddID = new Map((exhaustiveCatalog.markets || []).map(x => [x?.oddID, x]));
const exhaustiveRawEventsWithStartTime = (unrestrictedEventSweep.events || []).map(event => ({
  ...event,
  startTime: event?.startTime || event?.status?.startsAt || event?.commenceTime || null,
}));
const exhaustiveTodayRawEvents = filterEventsByLocalDate(exhaustiveRawEventsWithStartTime, date, timeZone);
const exhaustivePriceRows = [];
for (const event of exhaustiveTodayRawEvents) {
  const matchup = `${event?.teams?.away?.names?.long || event?.teams?.away?.name || 'Away'} @ ${event?.teams?.home?.names?.long || event?.teams?.home?.name || 'Home'}`;
  for (const [oddID, odd] of Object.entries(event?.odds || {})) {
    // This table is the standard I2 catalog audit. The unrestricted Event sweep
    // carries the full odds tree separately; do not serialize unrelated MLB
    // markets into the support artifact.
    if (!exhaustiveCatalogByOddID.has(oddID)) continue;
    const def = exhaustiveCatalogByOddID.get(oddID) || odd || {};
    for (const [bookmakerID, book] of Object.entries(odd?.byBookmaker || {})) {
      exhaustivePriceRows.push({
        eventID: String(event?.eventID || ''),
        startTime: event?.startTime || event?.status?.startsAt || null,
        matchup,
        oddID,
        statID: def?.statID ?? odd?.statID ?? null,
        statEntityID: def?.statEntityID ?? odd?.statEntityID ?? null,
        periodID: def?.periodID ?? odd?.periodID ?? null,
        betTypeID: def?.betTypeID ?? odd?.betTypeID ?? null,
        sideID: def?.sideID ?? odd?.sideID ?? null,
        marketGroupID: def?.marketGroupID ?? odd?.marketGroupID ?? null,
        marketGroupName: def?.marketGroupName ?? odd?.marketGroupName ?? null,
        bookmakerID: String(bookmakerID).toLowerCase(),
        available: book?.available === true,
        americanOdds: book?.odds ?? null,
        line: book?.overUnder ?? null,
        lastUpdatedAt: book?.lastUpdatedAt ?? null,
        deeplink: book?.deeplink ?? null,
        altLines: Array.isArray(book?.altLines) ? book.altLines : [],
      });
    }
  }
}
exhaustivePriceRows.sort((a,b)=>
  String(a.matchup).localeCompare(String(b.matchup)) ||
  String(a.bookmakerID).localeCompare(String(b.bookmakerID)) ||
  String(a.oddID).localeCompare(String(b.oddID))
);

const exhaustiveSupportByBook = {};
for (const market of exhaustiveCatalog.markets || []) {
  const leagueSupport = market?.support?.MLB || {};
  for (const [bookmakerID, supportEntry] of Object.entries(leagueSupport)) {
    if (supportEntry?.supported !== true) continue;
    if (!exhaustiveSupportByBook[bookmakerID]) exhaustiveSupportByBook[bookmakerID] = [];
    exhaustiveSupportByBook[bookmakerID].push({
      oddID: market.oddID,
      statID: market.statID ?? null,
      statEntityID: market.statEntityID ?? null,
      periodID: market.periodID ?? null,
      betTypeID: market.betTypeID ?? null,
      sideID: market.sideID ?? null,
      marketGroupID: market.marketGroupID ?? null,
      marketGroupName: market.marketGroupName ?? null,
      isMainMarket: market.isMainMarket === true,
      isMainDerivative: market.isMainDerivative === true,
      isProp: market.isProp === true,
      isSubPeriod: market.isSubPeriod === true,
    });
  }
}
for (const rows of Object.values(exhaustiveSupportByBook)) {
  rows.sort((a,b)=>String(a.oddID).localeCompare(String(b.oddID)));
}

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
  exhaustiveDiscovery: {
    allMlbMarketCount: exhaustiveCatalog.allMarketCount ?? exhaustiveCatalog.markets.length,
    catalogMarketCount: exhaustiveCatalog.markets.length,
    supportedMarketCount: exhaustiveCatalog.supportedMarketCount ?? null,
    unsupportedMarketCount: exhaustiveCatalog.unsupportedMarketCount ?? null,
    oddIDCount: exhaustiveOddIDs.length,
    rawEventCount: unrestrictedEventSweep.events.length,
    todayRawEventCount: exhaustiveTodayRawEvents.length,
    exhaustivePriceRowCount: exhaustivePriceRows.length,
    supportedBookmakers: Object.keys(exhaustiveSupportByBook).sort(),
    eventLevel: {
      caesarsPropPages: caesarsPropSweep.pageCount,
      caesarsPropEvents: caesarsPropSweep.eventCount,
      allBookPropPages: allBookPropSweep.pageCount,
      allBookPropEvents: allBookPropSweep.eventCount,
      unrestrictedPages: unrestrictedEventSweep.pageCount,
      unrestrictedEvents: unrestrictedEventSweep.eventCount,
      cursorExhausted: caesarsPropSweep.cursorExhausted && allBookPropSweep.cursorExhausted && unrestrictedEventSweep.cursorExhausted,
      candidateRows: eventLevelDiscoveryToday.length,
      bookmakerAudit: eventLevelBookmakerAudit,
      rows: eventLevelDiscoveryToday,
    },
  },
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
await fs.mkdir(path.dirname(exhaustiveCatalogPath), { recursive: true });
await fs.mkdir(path.dirname(exhaustiveRawPath), { recursive: true });
await fs.writeFile(outputPath, JSON.stringify(output, null, 2) + '\n');
await fs.writeFile(csvPath, csv);
await fs.writeFile(supportPath, JSON.stringify({
  generatedAt: new Date().toISOString(),
  date,
  bookmakerIDs,
  guaranteedOddIDs,
  ninthInningCandidates,
  activeOddIDs,
  i2BookmakerSupport,
  exhaustiveDiscovery: {
    allMlbMarketCount: exhaustiveCatalog.allMarketCount ?? exhaustiveCatalog.markets.length,
    catalogMarketCount: exhaustiveCatalog.markets.length,
    supportedMarketCount: exhaustiveCatalog.supportedMarketCount ?? null,
    unsupportedMarketCount: exhaustiveCatalog.unsupportedMarketCount ?? null,
    oddIDCount: exhaustiveOddIDs.length,
    rawEventCount: unrestrictedEventSweep.events.length,
    todayRawEventCount: exhaustiveTodayRawEvents.length,
    exhaustivePriceRowCount: exhaustivePriceRows.length,
    supportByBookmaker: exhaustiveSupportByBook,
    priceRows: exhaustivePriceRows,
    eventLevel: {
      caesarsPropQuery: caesarsPropSweep.query,
      caesarsPropPages: caesarsPropSweep.pageCount,
      caesarsPropEvents: caesarsPropSweep.eventCount,
      caesarsPropCursorExhausted: caesarsPropSweep.cursorExhausted,
      allBookPropQuery: allBookPropSweep.query,
      allBookPropPages: allBookPropSweep.pageCount,
      allBookPropEvents: allBookPropSweep.eventCount,
      allBookPropCursorExhausted: allBookPropSweep.cursorExhausted,
      unrestrictedQuery: unrestrictedEventSweep.query,
      unrestrictedPages: unrestrictedEventSweep.pageCount,
      unrestrictedEvents: unrestrictedEventSweep.eventCount,
      unrestrictedCursorExhausted: unrestrictedEventSweep.cursorExhausted,
      candidateRows: eventLevelDiscoveryToday.length,
      bookmakerAudit: eventLevelBookmakerAudit,
      rows: eventLevelDiscoveryToday,
    },
  },
  response: support,
}, null, 2) + '\n');
await fs.writeFile(exhaustiveCatalogPath, JSON.stringify({
  ...exhaustiveCatalog,
  supportByBookmaker: exhaustiveSupportByBook,
}, null, 2) + '\n');
await fs.writeFile(exhaustiveRawPath, JSON.stringify({ unrestrictedEventSweep, caesarsPropSweep, allBookPropSweep }, null, 2) + '\n');

console.log(JSON.stringify({
  date,
  frozenAt,
  outputPath,
  csvPath,
  supportPath,
  exhaustiveCatalogPath,
  exhaustiveRawPath,
  coverage,
  exhaustiveDiscovery: {
    allMlbMarketCount: exhaustiveCatalog.allMarketCount ?? exhaustiveCatalog.markets.length,
    catalogMarketCount: exhaustiveCatalog.markets.length,
    supportedMarketCount: exhaustiveCatalog.supportedMarketCount ?? null,
    unsupportedMarketCount: exhaustiveCatalog.unsupportedMarketCount ?? null,
    oddIDCount: exhaustiveOddIDs.length,
    rawEventCount: unrestrictedEventSweep.events.length,
    supportedBookmakers: Object.keys(exhaustiveSupportByBook).sort(),
    eventLevelCandidateRows: eventLevelDiscoveryToday.length,
    caesarsPropPages: caesarsPropSweep.pageCount,
    allBookPropPages: allBookPropSweep.pageCount,
    unrestrictedPages: unrestrictedEventSweep.pageCount,
    eventLevelBookmakerAudit,
  },
  i2BookmakerSupport,
}, null, 2));

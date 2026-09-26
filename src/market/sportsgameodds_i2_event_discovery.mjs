import {
  SPORTSGAMEODDS_DATA_SOURCE,
  assertPostFreezeContext,
} from './sportsgameodds_data_source.mjs';

const SECOND_INNING_RE = /(?:\b2nd\s+inning\b|\bsecond\s+inning\b|\binning\s*2\b|(?:^|[^a-z0-9])2i(?:[^a-z0-9]|$))/i;
const ZERO_RUN_RE = /(?:\b(?:exact(?:ly)?\s+)?(?:0|zero)\s+runs?\b|\bno\s+runs?\b)/i;
const ONE_RUN_RE = /\b(?:exact(?:ly)?\s+)?1\s+run(?:s)?\b/i;
const TWO_RUN_RE = /\b(?:exact(?:ly)?\s+)?2\s+runs?\b/i;
const THREE_PLUS_RE = /\b(?:3\s*\+|3\s+or\s+more|at\s+least\s+3)\s*runs?\b/i;
const ONE_PLUS_RE = /\b(?:1\s*\+|1\s+or\s+more|at\s+least\s+1|one\s+or\s+more)\s*runs?\b/i;
const RUN_CONCEPT_RE = /\bruns?\b|\bnumber\s+of\s+runs?\b|\binning\s+runs?\b/i;
const ANY_RUN_RE = /\bany\s+runs?\b/i;
const EVEN_ODD_RE = /\beven\b|\bodd\b/i;

function requireApiKey(apiKey = process.env[SPORTSGAMEODDS_DATA_SOURCE.apiKeyEnv]) {
  const value = String(apiKey || '').trim();
  if (!value) throw new Error(SPORTSGAMEODDS_DATA_SOURCE.apiKeyEnv + ' is required');
  return value;
}

function buildUrl(pathname, params = {}) {
  const url = new URL(SPORTSGAMEODDS_DATA_SOURCE.baseUrl + pathname);
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    if (Array.isArray(value)) {
      if (value.length) url.searchParams.set(key, value.join(','));
    } else {
      url.searchParams.set(key, String(value));
    }
  }
  return url;
}

async function fetchJson(pathname, params, { apiKey, signal } = {}) {
  const key = requireApiKey(apiKey);
  const url = buildUrl(pathname, params);
  const response = await fetch(url, {
    headers: {
      accept: 'application/json',
      'x-api-key': key,
      'user-agent': 'MLB-Model-SportsGameOdds-I2-Discovery/1.0',
    },
    signal,
  });
  const raw = await response.text();
  let body = null;
  try { body = raw ? JSON.parse(raw) : null; } catch { body = { raw }; }
  if (!response.ok) {
    const detail = body?.message || body?.error || body?.detail || raw.slice(0, 300);
    throw new Error('SportsGameOdds ' + response.status + ' ' + response.statusText + (detail ? ': ' + detail : ''));
  }
  return body;
}

export async function fetchMlbEventsExhaustive({
  freezeContext,
  type,
  bookmakerIDs = [],
  oddsPresent = true,
  oddsAvailable,
  started = false,
  includeAltLines = true,
  includeOpenCloseOdds = false,
  apiKey,
  signal,
  limit = 100,
  sourceLabel = 'UNRESTRICTED_MLB_EVENTS',
} = {}) {
  assertPostFreezeContext(freezeContext);
  const events = [];
  const seenEventIDs = new Set();
  const seenCursors = new Set();
  let cursor = null;
  let pageCount = 0;

  do {
    if (cursor && seenCursors.has(cursor)) {
      throw new Error('SportsGameOdds pagination cursor repeated before exhaustion for ' + sourceLabel);
    }
    if (cursor) seenCursors.add(cursor);

    const params = {
      leagueID: 'MLB',
      type: type || undefined,
      bookmakerID: bookmakerIDs?.length ? bookmakerIDs : undefined,
      oddsPresent,
      oddsAvailable,
      started,
      includeAltLines,
      includeOpenCloseOdds,
      limit,
      cursor: cursor || undefined,
    };
    const payload = await fetchJson('/events', params, { apiKey, signal });
    pageCount += 1;
    for (const event of Array.isArray(payload?.data) ? payload.data : []) {
      const eventID = String(event?.eventID || '');
      const key = eventID || sourceLabel + ':page:' + pageCount + ':index:' + events.length;
      if (seenEventIDs.has(key)) continue;
      seenEventIDs.add(key);
      events.push(event);
    }
    cursor = payload?.nextCursor || null;
    if (pageCount > 1000) throw new Error('SportsGameOdds pagination exceeded 1000 pages for ' + sourceLabel);
  } while (cursor);

  return {
    fetchedAt: new Date().toISOString(),
    provider: SPORTSGAMEODDS_DATA_SOURCE.provider,
    freezeContext,
    sourceLabel,
    query: {
      leagueID: 'MLB',
      type: type || null,
      bookmakerIDs: bookmakerIDs?.length ? [...bookmakerIDs] : [],
      oddsPresent,
      oddsAvailable: oddsAvailable ?? null,
      started,
      oddID: null,
      betTypeID: null,
      periodID: null,
      includeAltLines,
      includeOpenCloseOdds,
      limit,
    },
    pageCount,
    cursorExhausted: cursor === null,
    eventCount: events.length,
    events,
  };
}

function flattenText(value, path = '', out = [], depth = 0) {
  if (value === null || value === undefined || depth > 7) return out;
  const type = typeof value;
  if (type === 'string' || type === 'number' || type === 'boolean') {
    const text = String(value).trim();
    if (text) out.push({ path, value: text });
    return out;
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < Math.min(value.length, 100); i += 1) {
      flattenText(value[i], path + '[' + i + ']', out, depth + 1);
    }
    return out;
  }
  if (type === 'object') {
    for (const [key, child] of Object.entries(value)) {
      flattenText(child, path ? path + '.' + key : key, out, depth + 1);
    }
  }
  return out;
}

function eventMetadataFields(event) {
  const copy = {};
  for (const [key, value] of Object.entries(event || {})) {
    if (['odds', 'results', 'lineups'].includes(key)) continue;
    copy[key] = value;
  }
  return flattenText(copy, 'event').slice(0, 240);
}

function oddMetadataFields(oddID, odd) {
  const copy = { oddID };
  for (const [key, value] of Object.entries(odd || {})) {
    if (['byBookmaker', 'fairOdds', 'bookOdds', 'fairSpread', 'bookSpread', 'fairOverUnder', 'bookOverUnder'].includes(key)) continue;
    copy[key] = value;
  }
  return flattenText(copy, 'odd').slice(0, 160);
}

function priceMetadataFields(book, source, isAlternateLine) {
  return flattenText({
    bookmaker: book,
    selection: source,
    isAlternateLine,
  }, 'price').filter(x => !/\.odds$|openOdds$|closeOdds$/i.test(x.path)).slice(0, 120);
}

function sideMetadataFields(allFields, sideID) {
  const side = String(sideID || '').toLowerCase();
  if (!side) return [];
  const aliases = [side];
  if (side === 'side1') aliases.push('option1', 'outcome1', 'choice1', 'selection1', 'answer1');
  if (side === 'side2') aliases.push('option2', 'outcome2', 'choice2', 'selection2', 'answer2');
  return allFields.filter(field => {
    const path = String(field.path || '').toLowerCase();
    return aliases.some(alias => path.includes(alias));
  });
}

function compactText(fields) {
  return (fields || []).map(x => x.value).filter(Boolean).join(' ');
}

function parseNumber(value) {
  if (value === undefined || value === null || value === '') return null;
  const n = Number(String(value).replace('+', ''));
  return Number.isFinite(n) ? n : null;
}

function firstString(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) return value.trim();
  }
  return null;
}

function eventDescriptor(event) {
  return {
    eventName: firstString(event?.name, event?.eventName, event?.info?.name, event?.info?.displayName, event?.info?.shortName),
    eventTitle: firstString(event?.title, event?.eventTitle, event?.info?.title, event?.info?.displayTitle),
    eventDescription: firstString(event?.description, event?.eventDescription, event?.info?.description, event?.info?.details),
    parentEventID: firstString(event?.parentEventID, event?.parentEventId, event?.matchEventID, event?.gameEventID, event?.info?.parentEventID),
  };
}

function classifyStandard({ odd, line, contextText }) {
  const periodID = String(odd?.periodID || '');
  const statID = String(odd?.statID || '');
  const statEntityID = String(odd?.statEntityID || '');
  const betTypeID = String(odd?.betTypeID || '');
  const sideID = String(odd?.sideID || '');

  if (periodID !== '2i') return null;

  if (statID === 'points' && statEntityID === 'all' && betTypeID === 'ou') {
    if (line === 0.5 && sideID === 'under') return { classification:'A', semantic:'EXACT_UNDER_EQUIVALENT', reason:'Full 2nd-inning total Under 0.5' };
    if (line === 0.5 && sideID === 'over') return { classification:'B', semantic:'EXACT_OVER_EQUIVALENT', reason:'Full 2nd-inning total Over 0.5' };
    return { classification:'C', semantic:'FULL_INNING_TOTAL_OTHER_LINE', reason:'Full 2nd-inning total at a non-0.5 line is separately modelable' };
  }

  if (statID === 'points' && statEntityID === 'all' && betTypeID === 'yn') {
    if (sideID === 'no') return { classification:'A', semantic:'EXACT_UNDER_EQUIVALENT', reason:'2nd-inning Any Runs = No is identical to Under 0.5' };
    if (sideID === 'yes') return { classification:'B', semantic:'EXACT_OVER_EQUIVALENT', reason:'2nd-inning Any Runs = Yes is identical to Over 0.5' };
  }

  if (betTypeID === 'ml3way') {
    return { classification:'C', semantic:'THREE_WAY_INNING_RESULT', reason:'3-way inning result is separately modelable; draw/tie is not mapped to Under 0.5' };
  }

  if (betTypeID === 'eo' || EVEN_ODD_RE.test(contextText)) {
    return { classification:'C', semantic:'EVEN_ODD_INNING_RUNS', reason:'Even/Odd is separately modelable from the inning run distribution' };
  }

  if (statID === 'points' && ['home','away'].includes(statEntityID)) {
    return { classification:'C', semantic:'TEAM_HALF_INNING_MARKET', reason:'Team-side 2nd-inning market is not a full-inning equivalent but can be modeled separately' };
  }

  return { classification:'D', semantic:'NOT_FULL_I2_EQUIVALENT', reason:'Second-inning market does not settle as the full-inning 0.5 proposition' };
}

function classifyCustomProp({ event, odd, contextText, sideText }) {
  const eventType = String(event?.type || '').toLowerCase();
  const betTypeID = String(odd?.betTypeID || '').toLowerCase();
  if (eventType !== 'prop' && betTypeID !== 'prop') return null;

  const sideZero = ZERO_RUN_RE.test(sideText);
  const sideOnePlus = ONE_PLUS_RE.test(sideText);
  const sideExactOne = ONE_RUN_RE.test(sideText);
  const sideExactTwo = TWO_RUN_RE.test(sideText);
  const sideThreePlus = THREE_PLUS_RE.test(sideText);
  const sideYes = /\byes\b/i.test(sideText);
  const sideNo = /\bno\b/i.test(sideText);
  const zeroContext = ZERO_RUN_RE.test(contextText);
  const anyRunsContext = ANY_RUN_RE.test(contextText);

  if (sideZero) return { classification:'A', semantic:'EXACT_UNDER_EQUIVALENT', reason:'Custom prop selection explicitly settles on exactly 0 total 2nd-inning runs' };
  if (sideOnePlus) return { classification:'B', semantic:'EXACT_OVER_EQUIVALENT', reason:'Custom prop selection explicitly settles on 1+ total 2nd-inning runs' };

  if (anyRunsContext && sideNo) return { classification:'A', semantic:'EXACT_UNDER_EQUIVALENT', reason:'Custom 2nd-inning Any Runs = No is identical to Under 0.5' };
  if (anyRunsContext && sideYes) return { classification:'B', semantic:'EXACT_OVER_EQUIVALENT', reason:'Custom 2nd-inning Any Runs = Yes is identical to Over 0.5' };

  if (zeroContext && sideYes) return { classification:'A', semantic:'EXACT_UNDER_EQUIVALENT', reason:'Custom prop asks whether exactly 0 runs occur and this side is Yes' };
  if (zeroContext && sideNo) return { classification:'B', semantic:'EXACT_OVER_EQUIVALENT', reason:'Custom prop asks whether exactly 0 runs occur and this side is No, i.e. 1+ runs' };

  if (sideExactOne || sideExactTwo || sideThreePlus || ONE_RUN_RE.test(contextText) || TWO_RUN_RE.test(contextText) || THREE_PLUS_RE.test(contextText)) {
    return { classification:'C', semantic:'EXACT_RUN_CATEGORY', reason:'Exact-run category is separately modelable but is not by itself the full Over 0.5 proposition' };
  }

  if (RUN_CONCEPT_RE.test(contextText)) {
    return { classification:'C', semantic:'CUSTOM_SECOND_INNING_RUN_PROP', reason:'Custom 2nd-inning run prop found; settlement text is preserved for distribution-based modeling' };
  }

  return { classification:'D', semantic:'CUSTOM_PROP_NOT_RUN_EQUIVALENT', reason:'Custom 2nd-inning prop lacks enough run-settlement semantics to treat as an equivalent' };
}

function flattenBookSelections(book) {
  const selections = [{ source: book, isAlternateLine: false }];
  if (Array.isArray(book?.altLines)) {
    for (const source of book.altLines) selections.push({ source, isAlternateLine: true });
  }
  return selections;
}

export function discoverMlbI2EventLevelMarkets(events, { sourceLabel = 'UNRESTRICTED_MLB_EVENTS' } = {}) {
  const rows = [];
  for (const event of events || []) {
    const eventFields = eventMetadataFields(event);
    const eventText = compactText(eventFields);
    const descriptor = eventDescriptor(event);

    for (const [oddID, odd] of Object.entries(event?.odds || {})) {
      const oddFields = oddMetadataFields(oddID, odd);
      const oddText = compactText(oddFields);
      const contextText = [eventText, oddText, oddID].filter(Boolean).join(' ');
      const isSecondInning = String(odd?.periodID || '') === '2i' || SECOND_INNING_RE.test(contextText);
      if (!isSecondInning) continue;

      const standardI2 = String(odd?.periodID || '') === '2i' && String(odd?.statID || '') === 'points';
      const customProp = String(event?.type || '').toLowerCase() === 'prop' || String(odd?.betTypeID || '').toLowerCase() === 'prop';
      if (!standardI2 && !customProp && !RUN_CONCEPT_RE.test(contextText)) continue;

      const allSideFields = [...eventFields, ...oddFields];
      const sideFields = sideMetadataFields(allSideFields, odd?.sideID);

      for (const [bookmakerIDRaw, book] of Object.entries(odd?.byBookmaker || {})) {
        const bookmakerID = String(bookmakerIDRaw).toLowerCase();
        for (const { source, isAlternateLine } of flattenBookSelections(book)) {
          const priceFields = priceMetadataFields(book, source, isAlternateLine);
          const sideText = compactText([...sideFields, ...priceFields]);
          const line = parseNumber(source?.overUnder ?? source?.line ?? source?.total);
          const standard = classifyStandard({ odd, line, contextText });
          const custom = standard || classifyCustomProp({ event, odd, contextText, sideText });
          const classification = custom || { classification:'D', semantic:'NOT_USEFUL', reason:'No full-I2 economic equivalence established' };

          rows.push({
            sourceLabel,
            eventID: String(event?.eventID || ''),
            eventType: event?.type || null,
            ...descriptor,
            startTime: event?.startTime || event?.status?.startsAt || null,
            oddID,
            statID: odd?.statID ?? null,
            statEntityID: odd?.statEntityID ?? null,
            periodID: odd?.periodID ?? null,
            betTypeID: odd?.betTypeID ?? null,
            sideID: odd?.sideID ?? null,
            marketGroupID: odd?.marketGroupID ?? null,
            marketGroupName: odd?.marketGroupName ?? null,
            bookmakerID,
            available: source?.available === true,
            americanOdds: parseNumber(source?.odds),
            line,
            selection: firstString(source?.selection, source?.selectionName, source?.name, source?.label, source?.outcome, source?.description),
            isAlternateLine,
            timestamp: source?.lastUpdatedAt || book?.lastUpdatedAt || null,
            deeplink: source?.deeplink || book?.deeplink || null,
            classification: classification.classification,
            semantic: classification.semantic,
            classificationReason: classification.reason,
            rawSettlementSemantics: {
              eventMetadata: eventFields,
              sideMetadata: sideFields,
              oddMetadata: oddFields,
              priceMetadata: priceFields,
            },
          });
        }
      }
    }
  }
  rows.sort((a,b) =>
    String(a.bookmakerID).localeCompare(String(b.bookmakerID)) ||
    String(a.eventID).localeCompare(String(b.eventID)) ||
    String(a.oddID).localeCompare(String(b.oddID)) ||
    Number(a.isAlternateLine) - Number(b.isAlternateLine)
  );
  return rows;
}

function rowKey(row) {
  return [
    row.eventID,
    row.oddID,
    row.bookmakerID,
    row.isAlternateLine ? 'alt' : 'main',
    row.americanOdds ?? '',
    row.line ?? '',
    row.selection ?? '',
  ].join('|');
}

export function mergeI2DiscoveryRows(groups = []) {
  const merged = new Map();
  for (const group of groups) {
    const source = group?.sourceLabel || 'UNKNOWN';
    for (const row of group?.rows || []) {
      const key = rowKey(row);
      if (!merged.has(key)) {
        merged.set(key, { ...row, discoverySources: [source] });
      } else {
        const current = merged.get(key);
        current.discoverySources = [...new Set([...(current.discoverySources || []), source])];
      }
    }
  }
  return [...merged.values()].sort((a,b) =>
    String(a.bookmakerID).localeCompare(String(b.bookmakerID)) ||
    String(a.eventID).localeCompare(String(b.eventID)) ||
    String(a.oddID).localeCompare(String(b.oddID))
  );
}

export function summarizeI2DiscoveryByBookmaker(rows, { bookmakerIDs = [] } = {}) {
  const books = new Set((bookmakerIDs || []).map(x => String(x).toLowerCase()));
  for (const row of rows || []) books.add(String(row.bookmakerID || '').toLowerCase());

  return [...books].filter(Boolean).sort().map(bookmakerID => {
    const bookRows = (rows || []).filter(row => row.bookmakerID === bookmakerID);
    const availableRows = bookRows.filter(row => row.available && row.americanOdds !== null);
    const exactUnder = availableRows.filter(row => row.classification === 'A');
    const exactOver = availableRows.filter(row => row.classification === 'B');
    const fallback = availableRows.filter(row => row.classification === 'C');
    const notUseful = availableRows.filter(row => row.classification === 'D');
    return {
      bookmakerID,
      candidateRows: bookRows.length,
      availableRows: availableRows.length,
      exactUnderRows: exactUnder.length,
      exactOverRows: exactOver.length,
      fallbackRows: fallback.length,
      notUsefulRows: notUseful.length,
      exactUnderOddIDs: [...new Set(exactUnder.map(row => row.oddID))].sort(),
      exactOverOddIDs: [...new Set(exactOver.map(row => row.oddID))].sort(),
      fallbackOddIDs: [...new Set(fallback.map(row => row.oddID))].sort(),
      discoverySources: [...new Set(bookRows.flatMap(row => row.discoverySources || [row.sourceLabel]).filter(Boolean))].sort(),
      status: exactUnder.length || exactOver.length
        ? 'EXACT_EQUIVALENT_AVAILABLE'
        : (fallback.length ? 'FALLBACK_ONLY' : (bookRows.length ? 'NO_AVAILABLE_EQUIVALENT' : 'NO_I2_EVENT_ROWS')),
    };
  });
}

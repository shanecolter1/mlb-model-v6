import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

// This collector stays in the repository that already holds the API credential.
// Only the provider response and non-sensitive provenance leave the job.
const key = String(process.env.SPORTSGAMEODDS_API_KEY || '').trim();
if (!key) throw new Error('SPORTSGAMEODDS_API_KEY is not configured in the MLB repository');
const slate = process.env.NCAA_SLATE_ID || '2026-W08';
const after = process.env.NCAA_STARTS_AFTER || '2026-10-19T00:00:00Z';
const before = process.env.NCAA_STARTS_BEFORE || '2026-10-26T05:00:00Z';
if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,60}$/.test(slate)) throw new Error('Invalid slate identifier');
const start = Date.parse(after), end = Date.parse(before);
if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start || end - start > 14 * 86400000) {
  throw new Error('Snapshot needs a valid, increasing window of at most fourteen days');
}
const books = ['draftkings', 'fanduel', 'betmgm', 'caesars', 'bet365', 'fanatics', 'hardrockbet', 'thescorebet', 'ballybet', 'circa', 'pinnacle'];
// Slate discovery must not depend on a bookmaker posting odds or on pregame status.
// Capture every NCAAF event SGO returns in the requested week, including FCS/lower-division games.
const params = new URLSearchParams({
  leagueID: 'NCAAF',
  includeAltLines: 'true', includeOpenCloseOdds: 'true',
  // Omitting bookmakerID returns the books authorized for the existing key.
  // Explicitly requesting an unlicensed book makes the provider reject the
  // whole request. Apply the intended board scope after the authorized fetch.
  limit: '100', startsAfter: after, startsBefore: before,
});
const events = new Map(), seenCursors = new Set();
let pageCount = 0, cursor;
do {
  if (++pageCount > 10) throw new Error('Pagination exceeded the ten-page safety bound');
  if (cursor) params.set('cursor', cursor);
  const response = await fetch(`https://api.sportsgameodds.com/v2/events?${params}`, {
    headers: { accept: 'application/json', 'x-api-key': key },
    signal: AbortSignal.timeout(120000),
  });
  if (!response.ok) {
    const errorText = (await response.text()).replaceAll(key, '[redacted]').slice(0, 1200);
    throw new Error(`SportsGameOdds returned HTTP ${response.status}: ${errorText}`);
  }
  const body = await response.json();
  if (body.success !== true || !Array.isArray(body.data)) throw new Error('Invalid SportsGameOdds response');
  const captured = new Date().toISOString();
  for (const event of body.data) {
    const kickoff = Date.parse(event.status?.startsAt || event.startTime || '');
    if (event.leagueID !== 'NCAAF' || !event.eventID || !Number.isFinite(kickoff)) throw new Error('Unexpected event identity in response');
    if (kickoff < start || kickoff >= end) continue;
    if (events.has(event.eventID)) throw new Error('Duplicate event across API pages');
    const scopedOdds = Object.fromEntries(Object.entries(event.odds || {}).map(([oddID, odd]) => [oddID, {
      ...odd, byBookmaker: Object.fromEntries(Object.entries(odd.byBookmaker || {}).filter(([book]) => books.includes(book))),
    }]));
    events.set(event.eventID, { ...event, odds: scopedOdds, _captured_at_utc: captured });
  }
  cursor = body.nextCursor;
  if (cursor && seenCursors.has(cursor)) throw new Error('Repeated pagination cursor');
  if (cursor) seenCursors.add(cursor);
} while (cursor);

const payload = {
  success: true,
  captured_at_utc: new Date().toISOString(),
  provenance: {
    provider: 'SportsGameOdds', league_id: 'NCAAF', slate_id: slate,
    starts_after: after, starts_before: before, requested_books: books,
    bookmaker_filter: 'subscription_authorized_books_then_requested_scope',
    page_count: pageCount, repository: process.env.GITHUB_REPOSITORY || null,
    workflow_run_id: process.env.GITHUB_RUN_ID || null,
    workflow_run_attempt: process.env.GITHUB_RUN_ATTEMPT || null,
    source_commit: process.env.GITHUB_SHA || null,
  },
  data: [...events.values()],
};
const content = JSON.stringify(payload);
if (content.includes(key)) throw new Error('Response failed the credential-disclosure check');
const out = resolve(process.env.NCAA_OUTPUT_DIR || 'data/runtime/ncaa-sgo');
await mkdir(out, { recursive: true });
await writeFile(resolve(out, 'sportsgameodds.json'), content + '\n');
const counts = {};
for (const event of events.values()) {
  for (const odd of Object.values(event.odds || {})) {
    for (const [book, quote] of Object.entries(odd.byBookmaker || {})) {
      if (quote.available === true) counts[book] = (counts[book] || 0) + 1;
    }
  }
}
const summary = {
  ...payload.provenance, captured_at_utc: payload.captured_at_utc,
  events: events.size, available_main_quotes_by_book: counts,
  payload_sha256: createHash('sha256').update(content + '\n').digest('hex'),
};
await writeFile(resolve(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.log(JSON.stringify(summary, null, 2));
if (!events.size) throw new Error('No NCAA events were returned for the requested slate');

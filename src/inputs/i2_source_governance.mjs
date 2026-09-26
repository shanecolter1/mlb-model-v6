// Baseball input policy only. Never import model, market, calibration or staking code.
export const norm = value => String(value || '').normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase().replace(/[^a-z0-9]/g, '');
export function assertBaseballOnly(value) {
  if (!value || typeof value !== 'object') return;
  for (const [key, child] of Object.entries(value)) {
    if (/odds|sportsbook|bookmaker|implied.*prob|kelly|expectedvalue|fullgametotal|moneyline|spread|price|probabilityadjustment/i.test(key)) throw new Error(`NON_BASEBALL_FIELD:${key}`);
    assertBaseballOnly(child);
  }
}
export const validOrder = a => Array.isArray(a) && a.length === 9 && a.every(x => typeof x === 'string' && x.trim()) && new Set(a.map(norm)).size === 9;
export function lineupDelta(before = [], after = []) {
  const b = before.map(norm), a = after.map(norm);
  const moves = after.flatMap((player, i) => b.includes(a[i]) && b.indexOf(a[i]) !== i ? [{ player, from: b.indexOf(a[i]) + 1, to: i + 1 }] : []);
  return { added: after.filter((_, i) => !b.includes(a[i])), removed: before.filter((_, i) => !a.includes(b[i])), moves,
    top4Changes: after.slice(0, 4).flatMap((player, i) => a[i] !== b[i] ? [{position: i + 1, before: before[i] || null, after: player}] : []),
    exactSlots: after.filter((_, i) => a[i] === b[i]).length,
    // Audit metric only; never consumed by prediction weights.
    weightedSlotAccuracy: validOrder(before) && validOrder(after) ? after.reduce((n, _, i) => n + (a[i] === b[i] ? (i < 4 ? 2 : 1) : 0), 0) / 13 : null };
}
export function provisionalConfidence(rw, rr, unresolved = false) {
  if (unresolved || !validOrder(rw) || !validOrder(rr)) return 'LOW';
  const d = lineupDelta(rw, rr);
  if (d.top4Changes.length || d.added.length > 1) return 'LOW';
  return d.exactSlots >= 8 ? 'HIGH' : 'MEDIUM';
}
const ranks = { TEAM: 0, BEAT: 1, ROTOWIRE: 2, ROSTERRESOURCE: 3, REPORTING: 4, MLB: 5, PREVIOUS_GAME: 6, FALLBACK: 7 };
export function fresh(record, now = Date.now(), hours = 24) {
  const t = Date.parse(record?.timestamp || record?.retrievedAt);
  return Number.isFinite(t) && t <= now + 60000 && now - t <= hours * 3600000;
}
function metadata(record, status, confidence) {
  return { ...record, status, confidence, inputStatus: record.provider === 'PREVIOUS_GAME' || record.provider === 'FALLBACK' ? 'FALLBACK' : 'PRIMARY', fallback: ['PREVIOUS_GAME','FALLBACK'].includes(record.provider) };
}
export function selectLineup(candidates = [], news = [], now = Date.now()) {
  assertBaseballOnly(candidates); assertBaseballOnly(news);
  const rows = candidates.filter(c => validOrder(c.players) && (c.provider === 'PREVIOUS_GAME' || fresh(c, now)));
  const confirmed = rows.filter(c => c.confirmed && ['TEAM','BEAT','ROTOWIRE','MLB'].includes(c.provider));
  // MLB is final system confirmation; newer explicit external corrections still win.
  confirmed.sort((a,b) => Number(b.provider === 'MLB') - Number(a.provider === 'MLB') || ranks[a.provider] - ranks[b.provider] || Date.parse(b.timestamp || b.retrievedAt) - Date.parse(a.timestamp || a.retrievedAt));
  const rw = rows.find(c => c.provider === 'ROTOWIRE'), rr = rows.find(c => c.provider === 'ROSTERRESOURCE');
  const chosen = confirmed[0] || rw || rr || rows.find(c => c.provider === 'PREVIOUS_GAME');
  if (!chosen) return { status: 'MISSING', inputStatus: 'MISSING', confidence: 'LOW', source: null, timestamp: null, retrievedAt: null, fallback: false, players: [], unresolved: true };
  let players = [...chosen.players];
  const changes = [];
  let unresolved = false;
  for (const n of news.filter(n => n.verified === true && n.explicit === true && fresh(n, now) && Date.parse(n.timestamp) > Date.parse(chosen.timestamp || chosen.retrievedAt))) {
    if (['REST','SCRATCH','INACTIVE'].includes(n.action) && players.some(p => norm(p) === norm(n.player))) {
      if (n.replacement && Number.isInteger(n.position) && norm(players[n.position - 1]) === norm(n.player)) {
        players[n.position - 1] = n.replacement;
        changes.push(n);
      } else unresolved = true; // No invented substitute or batting slot.
    }
    if (['PLATOON_UNCERTAINTY','ROSTER_UNRESOLVED','INJURY_UNCERTAINTY'].includes(n.action)) unresolved = true;
  }
  if (!validOrder(players)) unresolved = true;
  const status = changes.length ? 'PROVISIONAL_NEWS_OVERRIDE' : chosen.confirmed ? (chosen.provider === 'MLB' ? 'CONFIRMED_MLB' : 'CONFIRMED_EXTERNAL') : chosen.provider === 'ROTOWIRE' ? 'PROVISIONAL_ROTOWIRE' : chosen.provider === 'ROSTERRESOURCE' ? 'PROVISIONAL_ROSTERRESOURCE' : 'FALLBACK_PREVIOUS_GAME';
  const previous = rows.find(c => c.provider === 'PREVIOUS_GAME');
  return { ...metadata(chosen, status, chosen.confirmed && !unresolved && !changes.length ? 'HIGH' : provisionalConfidence(players, rr?.players, unresolved)), players, unresolved, changes,
    confidenceSource: 'DERIVED', top4: players.slice(0,4),
    audit: { candidates: rows, vsPrevious: lineupDelta(previous?.players, players), rotowireVsRosterResource: rw && rr ? lineupDelta(rw.players, rr.players) : null,
      projectedVsConfirmed: chosen.confirmed ? rows.filter(c => !c.confirmed && c.provider !== 'PREVIOUS_GAME').map(c => ({ source: c.source, ...lineupDelta(c.players, players) })) : [] } };
}
export function selectStarter(candidates = [], now = Date.now()) {
  assertBaseballOnly(candidates);
  const rows = candidates.filter(c => c.name && fresh(c, now)).sort((a,b) => ranks[a.provider] - ranks[b.provider] || Date.parse(b.timestamp || b.retrievedAt) - Date.parse(a.timestamp || a.retrievedAt));
  if (!rows.length) return { name: null, status: 'TBD', inputStatus: 'MISSING', confidence: 'LOW', fallback: false, source: null, timestamp: null, retrievedAt: null };
  const identities = new Set(rows.map(c => norm(c.name)));
  const chosen = rows[0];
  const conflict = identities.size > 1;
  const confidence = conflict ? 'LOW' : chosen.confirmed ? 'HIGH' : rows.length > 1 ? 'HIGH' : chosen.provider === 'ROTOWIRE' ? 'MEDIUM' : 'LOW';
  return { ...metadata(chosen, conflict ? 'CONFLICTING' : chosen.confirmed ? 'CONFIRMED' : `PROJECTED_${confidence}`, confidence),
    conflict: conflict ? 'STARTER_CONFLICT' : null, candidates: rows };
}
export function projectionGate({ away, home, previous = null }) {
  const reasons = [];
  const changes = [];
  for (const [side, current] of Object.entries({away,home})) {
    if (current.starter.status === 'CONFLICTING') reasons.push('STARTER_CONFLICT');
    if (!current.starter.name || current.starter.status === 'TBD') reasons.push('STARTER_MISSING');
    if (!validOrder(current.lineup.players) || current.lineup.unresolved) reasons.push('LINEUP_UNRESOLVED');
    if (current.news?.some(n => ['STARTER_UPDATE_REQUIRED','PROJECTION_INVALIDATED'].includes(n.recommendedAction))) reasons.push('NEWS_REVIEW_REQUIRED');
    const prior = previous?.[side];
    if (prior?.starter?.name && norm(prior.starter.name) !== norm(current.starter.name)) changes.push('PROJECTION_INVALIDATED_STARTER_CHANGE');
    if (prior?.lineup?.players && JSON.stringify(prior.lineup.players.map(norm)) !== JSON.stringify(current.lineup.players.map(norm))) changes.push('PROJECTION_INVALIDATED_LINEUP_CHANGE');
  }
  return { projection: changes.length ? 'INVALIDATED' : reasons.length ? 'PRELIMINARY' : [away,home].every(c => c.lineup.status.startsWith('CONFIRMED')) ? 'VALID' : 'PRELIMINARY',
    requiresCleanRerun: changes.length > 0, invalidations: [...new Set(changes)],
    eligible: reasons.length === 0 && changes.length === 0, status: reasons.length || changes.length ? 'NO_ACTIONABLE_RECOMMENDATION' : 'ELIGIBLE', reasons: [...new Set([...reasons,...changes])] };
}
export function srmReview(news, currentStarter) {
  return news.filter(n => n.recommendedAction === 'SRM_REVIEW_RECOMMENDED').map(n => ({ currentInput: currentStarter, proposedChange: n.action,
    evidence: {source:n.source,timestamp:n.timestamp,reason:n.reason}, affectedComponent: 'starter quality/duration only', expectedDirection: 'REQUIRES_APPROVED_SRM', expectedMagnitude: null,
    status: 'APPROVAL_AND_EXISTING_SRM_REQUIRED', applied: false, beforeAfterDelta: null }));
}
export function compactAudit(game) {
  const input=game.inputAudit;
  const out={gamePk:game.gamePk,game:`${game.away} @ ${game.home}`,starter:{},awayLineup:null,homeLineup:null,news:[],projection:game.modelStatus==='PROJECTION_INVALIDATED'?'INVALIDATED':game.bettingEligibility?.projection || 'PRELIMINARY',bettingEligibility:game.bettingEligibility || {eligible:false,reasons:['INPUT_AUDIT_MISSING']}};
  if (!input) return out;
  for (const side of ['away','home']) {
    const s=input[side].starter,l=input[side].lineup;
    out.starter[side]={name:s.name,status:s.status,source:s.source,confidence:s.confidence,timestamp:s.timestamp,retrievedAt:s.retrievedAt,inputStatus:s.inputStatus,fallback:s.fallback};
    out[`${side}Lineup`]={status:l.status,source:l.source,confidence:l.confidence,timestamp:l.timestamp,retrievedAt:l.retrievedAt,inputStatus:l.inputStatus,fallback:l.fallback,top4:l.players.slice(0,4),differencesVsPrevious:l.audit?.vsPrevious};
    out.news.push(...input[side].news.map(n=>({side,source:n.source,timestamp:n.timestamp,finding:n.reason,recommendedAction:n.recommendedAction || 'LINEUP_UPDATE_RECOMMENDED'})));
  }
  if(!out.news.length)out.news=[{recommendedAction:'NO_CHANGE',coverage:input.providers.find(p=>p.source==='MLB_NEWS')?.status || 'MISSING'}];
  return out;
}

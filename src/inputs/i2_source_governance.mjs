// Baseball input policy only. Never import model, market, calibration or staking code.
export const norm = value => String(value || '').normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase().replace(/[^a-z0-9]/g, '');
const GENERATIONAL_SUFFIXES = new Set(['jr','sr','ii','iii','iv']);
function identityParts(value) {
  const tokens=String(value || '').normalize('NFD').replace(/\p{Diacritic}/gu,'').toLowerCase().replace(/[^a-z0-9]+/g,' ').trim().split(/\s+/).filter(Boolean);
  const suffix=GENERATIONAL_SUFFIXES.has(tokens.at(-1)) ? tokens.pop() : null;
  return {base:tokens.join(''),suffix,exact:norm(value)};
}
export function canonicalMlbIdentityName(value) { return identityParts(value).base; }
export function sameMlbIdentityName(a,b) {
  const left=identityParts(a), right=identityParts(b);
  if (!left.base || !right.base) return false;
  if (left.exact===right.exact) return true;
  if (left.suffix && right.suffix && left.suffix!==right.suffix) return false;
  return left.base===right.base;
}
export function assertBaseballOnly(value) {
  if (!value || typeof value !== 'object') return;
  for (const [key, child] of Object.entries(value)) {
    if (/odds|sportsbook|bookmaker|implied.*prob|kelly|expectedvalue|fullgametotal|moneyline|spread|price|probabilityadjustment/i.test(key)) throw new Error(`NON_BASEBALL_FIELD:${key}`);
    assertBaseballOnly(child);
  }
}
export const validOrder = a => Array.isArray(a) && a.length === 9 && a.every(x => typeof x === 'string' && x.trim()) && new Set(a.map(norm)).size === 9;
export function lineupDelta(before = [], after = []) {
  const beforeIndex = player => before.findIndex(x => sameMlbIdentityName(x,player));
  const moves = after.flatMap((player, i) => {
    const from=beforeIndex(player);
    return from>=0 && from!==i ? [{ player, from: from + 1, to: i + 1 }] : [];
  });
  const slotSame=(i)=>sameMlbIdentityName(before[i],after[i]);
  return { added: after.filter(player => beforeIndex(player)<0), removed: before.filter(player => !after.some(x=>sameMlbIdentityName(x,player))), moves,
    top4Changes: after.slice(0, 4).flatMap((player, i) => !slotSame(i) ? [{position: i + 1, before: before[i] || null, after: player}] : []),
    exactSlots: after.filter((_, i) => slotSame(i)).length,
    // Audit metric only; never consumed by prediction weights.
    weightedSlotAccuracy: validOrder(before) && validOrder(after) ? after.reduce((n, _, i) => n + (slotSame(i) ? (i < 4 ? 2 : 1) : 0), 0) / 13 : null };
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
  // One active provisional source: RotoWire. RosterResource remains audit-only.
  // If RotoWire is unavailable, fall back to the previous completed MLB lineup rather
  // than substituting a second projected lineup into production.
  const chosen = confirmed[0] || rw || rows.find(c => c.provider === 'PREVIOUS_GAME');
  if (!chosen) return { status: 'MISSING', inputStatus: 'MISSING', confidence: 'LOW', source: null, timestamp: null, retrievedAt: null, fallback: false, players: [], unresolved: true };
  let players = [...chosen.players];
  const changes = [];
  let unresolved = false;
  for (const n of news.filter(n => n.verified === true && n.explicit === true && fresh(n, now) && Date.parse(n.timestamp) > Date.parse(chosen.timestamp || chosen.retrievedAt))) {
    if (['REST','SCRATCH','INACTIVE'].includes(n.action) && players.some(p => sameMlbIdentityName(p,n.player))) {
      if (n.replacement && Number.isInteger(n.position) && sameMlbIdentityName(players[n.position - 1],n.player)) {
        players[n.position - 1] = n.replacement;
        changes.push(n);
      } else unresolved = true; // No invented substitute or batting slot.
    }
    if (['PLATOON_UNCERTAINTY','ROSTER_UNRESOLVED','INJURY_UNCERTAINTY'].includes(n.action)) unresolved = true;
  }
  if (!validOrder(players)) unresolved = true;
  const status = changes.length ? 'PROVISIONAL_NEWS_OVERRIDE' : chosen.confirmed ? (chosen.provider === 'MLB' ? 'CONFIRMED_MLB' : 'CONFIRMED_EXTERNAL') : chosen.provider === 'ROTOWIRE' ? 'PROVISIONAL_ROTOWIRE' : 'FALLBACK_PREVIOUS_GAME';
  const previous = rows.find(c => c.provider === 'PREVIOUS_GAME');
  return { ...metadata(chosen, status, chosen.confirmed && !unresolved && !changes.length ? 'HIGH' : provisionalConfidence(players, rr?.players, unresolved)), players, unresolved, changes,
    confidenceSource: 'DERIVED', top4: players.slice(0,4),
    audit: { provisionalSourcePolicy:'ROTOWIRE_ONLY', rosterResourceRole:'AUDIT_ONLY', candidates: rows,
      vsPrevious: lineupDelta(previous?.players, players), rotowireVsRosterResource: rw && rr ? lineupDelta(rw.players, rr.players) : null,
      projectedVsConfirmed: chosen.confirmed ? rows.filter(c => !c.confirmed && c.provider !== 'PREVIOUS_GAME').map(c => ({ source: c.source, ...lineupDelta(c.players, players) })) : [] } };
}
export function selectStarter(candidates = [], now = Date.now()) {
  assertBaseballOnly(candidates);
  const rows = candidates.filter(c => c.name && fresh(c, now)).sort((a,b) => ranks[a.provider] - ranks[b.provider] || Date.parse(b.timestamp || b.retrievedAt) - Date.parse(a.timestamp || a.retrievedAt));
  if (!rows.length) return { name: null, status: 'TBD', inputStatus: 'MISSING', confidence: 'LOW', fallback: false, source: null, timestamp: null, retrievedAt: null };
  const chosen = rows[0];
  const conflict = rows.some((row,i)=>rows.slice(i+1).some(other=>!sameMlbIdentityName(row.name,other.name)));
  const confidence = conflict ? 'LOW' : chosen.confirmed ? 'HIGH' : rows.length > 1 ? 'HIGH' : chosen.provider === 'ROTOWIRE' ? 'MEDIUM' : 'LOW';
  return { ...metadata(chosen, conflict ? 'CONFLICTING' : chosen.confirmed ? 'CONFIRMED' : `PROJECTED_${confidence}`, confidence),
    conflict: conflict ? 'STARTER_CONFLICT' : null, candidates: rows };
}
function starterIdentitySame(prior,current) {
  if (prior?.resolvedMlbId != null && current?.resolvedMlbId != null) return String(prior.resolvedMlbId)===String(current.resolvedMlbId);
  return sameMlbIdentityName(prior?.name,current?.name);
}
function lineupIdentitySame(prior,current) {
  const a=prior?.resolvedMlbIds, b=current?.resolvedMlbIds;
  if (Array.isArray(a) && Array.isArray(b) && a.length===9 && b.length===9 && a.every(x=>x!=null) && b.every(x=>x!=null)) {
    return a.every((id,i)=>String(id)===String(b[i]));
  }
  const before=prior?.players || [], after=current?.players || [];
  return before.length===after.length && before.every((player,i)=>sameMlbIdentityName(player,after[i]));
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
    if (prior?.starter?.name && !starterIdentitySame(prior.starter,current.starter)) changes.push('PROJECTION_INVALIDATED_STARTER_CHANGE');
    if (prior?.lineup?.players && !lineupIdentitySame(prior.lineup,current.lineup)) changes.push('PROJECTION_INVALIDATED_LINEUP_CHANGE');
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

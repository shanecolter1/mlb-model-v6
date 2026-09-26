import { collectSources, resolveGameInputs, applyResolvedInputs } from './i2_baseball_sources.mjs';
import { projectionGate } from './i2_source_governance.mjs';
// Revalidate identities before recommendations, never recompute or modify probabilities.
export async function revalidateFrozen(payload) {
  const sources=await collectSources(payload.date);
  let schedule;
  try {
    const r=await fetch(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${payload.date}&hydrate=team,probablePitcher`,{signal:AbortSignal.timeout(15000)});
    if (!r.ok) throw new Error('schedule');
    schedule=(await r.json()).dates?.flatMap(d=>d.games || []) || [];
  } catch { schedule=[]; }
  for (const game of payload.games || []) {
    try {
      const scheduled=schedule.find(g=>String(g.gamePk)===String(game.gamePk));
      if (!game.inputAudit || !scheduled) throw new Error('INPUT_AUDIT_MISSING');
      const r=await fetch(`https://statsapi.mlb.com/api/v1.1/game/${game.gamePk}/feed/live`,{signal:AbortSignal.timeout(15000)});
      if (!r.ok) throw new Error('feed');
      const feed=await r.json();
      const current=await resolveGameInputs(scheduled,feed,sources,game.inputAudit);
      await applyResolvedInputs(feed,current);
      current.gate=projectionGate({...current,previous:game.inputAudit});
      if (!['FROZEN_RESEARCH_PROJECTION','PROVISIONAL_RESEARCH_PROJECTION'].includes(game.modelStatus)) throw new Error('PROJECTION_NOT_VALID');
      if (Date.parse(game.gameDate)<=Date.now()) throw new Error('GAME_STARTED');
      game.bettingEligibility=current.gate;
      game.recommendationInputCheck={checkedAt:new Date().toISOString(),...current};
      if (current.gate.requiresCleanRerun) game.modelStatus='PROJECTION_INVALIDATED';
    } catch {
      game.bettingEligibility={eligible:false,status:'NO_ACTIONABLE_RECOMMENDATION',reasons:['CURRENT_BASEBALL_INPUTS_NOT_VERIFIED']};
    }
  }
  return payload;
}

// Legacy command name retained; all future runs share the production source hierarchy.
import fs from 'node:fs';
import path from 'node:path';
import { collectSources, resolveGameInputs } from '../inputs/i2_baseball_sources.mjs';
const date=process.env.I2_DATE || new Date().toISOString().slice(0,10);
const output=process.env.I2_LINEUP_OVERRIDES || `data/runtime/i2/${date}_lineup_overrides.json`;
async function json(url) { const r=await fetch(url,{signal:AbortSignal.timeout(15000)});if(!r.ok)throw new Error(`HTTP_${r.status}`);return r.json(); }
const sources=await collectSources(date);
const schedule=await json(`https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=${date}&hydrate=team,probablePitcher`);
const games={};
for (const game of (schedule.dates || []).flatMap(d=>d.games || [])) {
  try {
    const feed=await json(`https://statsapi.mlb.com/api/v1.1/game/${game.gamePk}/feed/live`);
    const audit=await resolveGameInputs(game,feed,sources);
    games[game.gamePk]={away:audit.away.lineup.players,home:audit.home.lineup.players,inputAudit:audit};
  } catch { games[game.gamePk]={status:'MISSING',bettingEligibility:{eligible:false,reasons:['BASEBALL_INPUTS_UNAVAILABLE']}}; }
}
fs.mkdirSync(path.dirname(output),{recursive:true});
fs.writeFileSync(output,JSON.stringify({date,generatedAt:new Date().toISOString(),governance:{sourceStatus:'BASEBALL_ONLY_SOURCE_HIERARCHY',marketDataUsed:false},games},null,2));
console.log(JSON.stringify({output,games:Object.keys(games).length}));

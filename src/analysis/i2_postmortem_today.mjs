import fs from 'node:fs';
import path from 'node:path';

const DATE = process.env.I2_DATE || new Date().toISOString().slice(0,10);
const PREDICTIONS = process.env.I2_PREDICTIONS || `data/runtime/i2/${DATE}_frozen_predictions.json`;
const OUTPUT = process.env.I2_POSTMORTEM_OUTPUT || `data/runtime/i2/${DATE}_postmortem.json`;
const UPSTREAM = String(process.env.MLB_OTHER_MODEL_BASE_URL || '').replace(/\/$/,'');
const HIST_BASELINE_UNDER = Number(process.env.I2_HIST_BASELINE_UNDER || 0.564958);

if (!fs.existsSync(PREDICTIONS)) throw new Error(`Missing prediction artifact: ${PREDICTIONS}`);
const pred = JSON.parse(fs.readFileSync(PREDICTIONS,'utf8'));

const audit={upstreamAttempts:0,upstreamSuccesses:0,officialFallbacks:0};
async function getFeed(gamePk){
  if (UPSTREAM){
    try{
      audit.upstreamAttempts++;
      const u=new URL(`${UPSTREAM}/.netlify/functions/mlb`);
      u.searchParams.set('type','feed'); u.searchParams.set('gamePk',String(gamePk));
      const r=await fetch(u,{headers:{accept:'application/json','user-agent':'MLB-I2-Postmortem/0.2'}});
      if(!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      audit.upstreamSuccesses++;
      return {feed:await r.json(),source:'USER_UPSTREAM'};
    }catch(e){ /* governed fallback below */ }
  }
  audit.officialFallbacks++;
  const r=await fetch(`https://statsapi.mlb.com/api/v1.1/game/${gamePk}/feed/live`,{headers:{accept:'application/json','user-agent':'MLB-I2-Postmortem/0.2'}});
  if(!r.ok) throw new Error(`MLB feed ${r.status} ${r.statusText}`);
  return {feed:await r.json(),source:'MLB_STATS_API_FALLBACK'};
}

// MLB boxscore battingOrder uses 100/200/.../900 for the original starters;
// later substitutes in the same batting slot use sequence suffixes such as 101.
// Restricting to values divisible by 100 avoids falsely treating substitutions
// as pregame lineup changes in a postgame comparison.
function lineupNames(feed,side){
  const team=feed?.liveData?.boxscore?.teams?.[side];
  return Object.values(team?.players||{})
    .filter(p=>Number(p.battingOrder)>0 && Number(p.battingOrder)%100===0)
    .sort((a,b)=>Number(a.battingOrder)-Number(b.battingOrder))
    .slice(0,9)
    .map(p=>p.person?.fullName);
}
function actualStarter(feed,side){
  const team=feed?.liveData?.boxscore?.teams?.[side];
  const pitcherIds=team?.pitchers||[];
  const id=pitcherIds[0];
  return id ? feed?.gameData?.players?.[`ID${id}`]?.fullName || team?.players?.[`ID${id}`]?.person?.fullName || null : null;
}
function secondInning(feed){
  const inn=(feed?.liveData?.linescore?.innings||[]).find(x=>Number(x.num)===2);
  if(!inn) return null;
  const away=Number(inn.away?.runs ?? 0), home=Number(inn.home?.runs ?? 0);
  const complete=Boolean(inn.away) && Boolean(inn.home);
  return {away,home,total:away+home,complete};
}
function sameLineup(a,b){return Array.isArray(a)&&Array.isArray(b)&&a.length===9&&b.length===9&&a.every((x,i)=>x===b[i]);}
function clamp(p){return Math.min(1-1e-12,Math.max(1e-12,p));}

const rows=[];
for(const g of pred.games||[]){
  if(!['FROZEN_RESEARCH_PROJECTION','PROVISIONAL_RESEARCH_PROJECTION'].includes(g.modelStatus)) continue;
  try{
    const {feed,source}=await getFeed(g.gamePk);
    const i2=secondInning(feed);
    const status=feed?.gameData?.status?.detailedState || feed?.gameData?.status?.abstractGameState || null;
    const actualAwayLineup=lineupNames(feed,'away'), actualHomeLineup=lineupNames(feed,'home');
    const actualAwayStarter=actualStarter(feed,'away'), actualHomeStarter=actualStarter(feed,'home');
    const p=Number(g.under05);
    const y=i2?.complete ? (i2.total===0?1:0) : null;
    rows.push({
      gamePk:g.gamePk, matchup:`${g.away} @ ${g.home}`, status, source,
      modelStatus:g.modelStatus, predictionClass:g.predictionClass||null, lineupSource:g.lineupSource||null,
      modelUnderPct:g.under05Pct, modelOverPct:g.over05Pct, fairUnder:g.fairUnder, fairOver:g.fairOver,
      actualI2:i2, actualUnder:y,
      brier:y===null?null:(p-y)**2,
      logLoss:y===null?null:-(y*Math.log(clamp(p))+(1-y)*Math.log(clamp(1-p))),
      baselineBrier:y===null?null:(HIST_BASELINE_UNDER-y)**2,
      predictedAwayStarter:g.awayStarter||null, actualAwayStarter,
      predictedHomeStarter:g.homeStarter||null, actualHomeStarter,
      awayStarterMatch:actualAwayStarter?actualAwayStarter===g.awayStarter:null,
      homeStarterMatch:actualHomeStarter?actualHomeStarter===g.homeStarter:null,
      predictedAwayLineup:g.awayLineup||[], actualAwayLineup,
      predictedHomeLineup:g.homeLineup||[], actualHomeLineup,
      awayLineupExactMatch:actualAwayLineup.length===9?sameLineup(g.awayLineup,actualAwayLineup):null,
      homeLineupExactMatch:actualHomeLineup.length===9?sameLineup(g.homeLineup,actualHomeLineup):null,
    });
  }catch(e){rows.push({gamePk:g.gamePk,matchup:`${g.away} @ ${g.home}`,error:String(e)});}
}

const graded=rows.filter(r=>r.actualUnder!==null&&r.actualUnder!==undefined);
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
const actualUnders=graded.filter(r=>r.actualUnder===1).length;
const expectedUnders=graded.reduce((s,r)=>s+Number(r.modelUnderPct||0)/100,0);
const summary={
  date:DATE, generatedAt:new Date().toISOString(), historicalBaselineUnder:HIST_BASELINE_UNDER,
  gradedGames:graded.length, actualUnders, actualOvers:graded.length-actualUnders,
  observedUnderRate:graded.length?actualUnders/graded.length:null,
  modelExpectedUnders:expectedUnders,
  meanModelUnderProbability:graded.length?expectedUnders/graded.length:null,
  modelBrier:mean(graded.map(r=>r.brier)), baselineBrier:mean(graded.map(r=>r.baselineBrier)),
  modelLogLoss:mean(graded.map(r=>r.logLoss)),
  starterMismatches:graded.filter(r=>r.awayStarterMatch===false||r.homeStarterMatch===false).map(r=>({gamePk:r.gamePk,matchup:r.matchup,predictedAwayStarter:r.predictedAwayStarter,actualAwayStarter:r.actualAwayStarter,predictedHomeStarter:r.predictedHomeStarter,actualHomeStarter:r.actualHomeStarter})),
  lineupMismatches:graded.filter(r=>r.awayLineupExactMatch===false||r.homeLineupExactMatch===false).map(r=>({gamePk:r.gamePk,matchup:r.matchup,predictionClass:r.predictionClass,awayMatch:r.awayLineupExactMatch,homeMatch:r.homeLineupExactMatch})),
  sourcePriorityAudit:audit,
};
const out={summary,rows};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(out,null,2));
console.log(JSON.stringify(summary,null,2));
for(const r of graded) console.log(`${r.matchup}: model U ${r.modelUnderPct}% | actual I2 ${r.actualI2.away}-${r.actualI2.home} (${r.actualI2.total}) | ${r.actualUnder?'UNDER':'OVER'} | starters ${r.awayStarterMatch&&r.homeStarterMatch?'MATCH':'MISMATCH'} | lineups ${r.awayLineupExactMatch&&r.homeLineupExactMatch?'MATCH':'MISMATCH'}`);

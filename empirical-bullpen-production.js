(function(){
'use strict';
const API='/.netlify/functions/mlb';
const modelCache={hazards:null,hazardPromise:null,contexts:new Map(),contextPromises:new Map()};
const CHOICE={
  median:[5,45,2,.5,0,0,0],
  mean:[6.40100364,56.6425670,2.86318137,.440720393,.138584330,.0876722513,-.459469506,.0428537291],
  scale:[3.71864001,39.82127258,2.41009642,.41009357,.24728821,.28281766,3.23886388,.20252725],
  cats:[['in_inning'],['1-4','5-6','7-8','9+'],['L','R'],['-3..-1','0','1..3','4+','<=-4']],
  coef:[.88936774,-1.87386758,.39061159,.0278413,-.31171156,-1.50438002,-.08295662,.00925017,-.02423848,-.13424822,-.23221,-.05324347,.3954632,.17076444,-.19500293,-.03652642,-.03433515,-.01151738,.07552574,-.01738528],
  intercept:-1.02464811
};
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const sigmoid=x=>1/(1+Math.exp(-clamp(x,-35,35)));
const inningBand=i=>Number(i)<=4?'1-4':Number(i)<=6?'5-6':Number(i)<=8?'7-8':'9+';
const scoreBand=d=>Number(d)<=-4?'<=-4':Number(d)<=-1?'-3..-1':Number(d)===0?'0':Number(d)<=3?'1..3':'4+';
function dateValue(){const el=document.querySelector('input[type="date"]');return /^\d{4}-\d{2}-\d{2}$/.test(el?.value||'')?el.value:new Date(Date.now()-6*3600e3).toISOString().slice(0,10)}
function teamId(live,side){return Number(live?.boxscore?.teams?.[side]?.team?.id||0)}
function scoreDiff(live,side){const t=live?.linescore?.teams||{};return side==='home'?Number(t.home?.runs||0)-Number(t.away?.runs||0):Number(t.away?.runs||0)-Number(t.home?.runs||0)}
function usedIds(live,side){return new Set((live?.boxscore?.teams?.[side]?.pitchers||[]).map(Number))}
function currentPid(live,side){const box=live?.boxscore?.teams?.[side]||{};const ids=box.pitchers||[];return Number(ids[ids.length-1]||0)}
function normalizeChoice(c,live,side,inning){
  const vals=[Number(c.prior_apps_30d||0),Number(c.prior_bf_30d||0),c.prior_rest_days==null?CHOICE.median[2]:Number(c.prior_rest_days),Number(c.prior_late_inning_share_30d||0),Number(c.prior_save_like_share_30d||0),usedIds(live,side).has(Number(c.pitcher_id))?1:0,scoreDiff(live,side)];
  let z=CHOICE.intercept;
  vals.forEach((v,i)=>z+=((v-CHOICE.mean[i])/CHOICE.scale[i])*CHOICE.coef[i]);
  let k=vals.length;
  const cats=['in_inning',inningBand(inning),String(c.throws||'UNK').toUpperCase(),scoreBand(vals[6])];
  cats.forEach((v,j)=>{for(const cat of CHOICE.cats[j]){z+=(v===cat?1:0)*CHOICE.coef[k++];}});
  return z;
}
function contextKey(team,date){return `${team}|${date}`}
async function fetchContext(team,date){
  const key=contextKey(team,date);if(modelCache.contexts.has(key))return modelCache.contexts.get(key);if(modelCache.contextPromises.has(key))return modelCache.contextPromises.get(key);
  const p=fetch(`${API}?type=bullpenContext&teamId=${encodeURIComponent(team)}&date=${encodeURIComponent(date)}`,{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`bullpenContext ${r.status}`);return r.json()}).then(x=>{modelCache.contexts.set(key,x);modelCache.contextPromises.delete(key);return x}).catch(e=>{modelCache.contextPromises.delete(key);throw e});
  modelCache.contextPromises.set(key,p);return p;
}
function warmContext(live,side){const tid=teamId(live,side);if(tid)fetchContext(tid,dateValue()).catch(()=>{});}
function relieverWeights(live,side,inning,outgoingPid){
  const tid=teamId(live,side),ctx=modelCache.contexts.get(contextKey(tid,dateValue()));if(!ctx?.candidates?.length)return null;
  const rows=ctx.candidates.filter(c=>Number(c.pitcher_id)!==Number(outgoingPid));if(!rows.length)return null;
  const scores=rows.map(c=>normalizeChoice(c,live,side,inning)),m=Math.max(...scores),w=scores.map(x=>Math.exp(x-m)),s=w.reduce((a,b)=>a+b,0)||1;
  return rows.map((c,i)=>({pitcherId:Number(c.pitcher_id),weight:w[i]/s}));
}
function weightedRunDistribution(live,side,lineup,startIdx,outs,mask,currentCount,outgoingPid){
  warmContext(live,side);const weights=relieverWeights(live,side,Number(live?.linescore?.currentInning||1),outgoingPid);if(!weights||typeof window.runDistribution!=='function')return null;
  const out=Array(7).fill(0);for(const x of weights){const d=window.runDistribution(lineup,startIdx,x.pitcherId,outs,mask,currentCount);for(let i=0;i<out.length;i++)out[i]+=x.weight*Number(d?.[i]||0)}
  const t=out.reduce((a,b)=>a+b,0)||1;return out.map(x=>x/t);
}
function transformHazard(model,row){
  if(!model)return null;let x=[];
  for(const tr of model.transformers||[]){
    if(tr.name==='num'){
      const raw=tr.columns.map(i=>row[i]);const vals=raw.map((v,j)=>Number.isFinite(Number(v))?Number(v):Number(tr.statistics?.[j]||0));
      let base=vals.map((v,j)=>(v-Number(tr.mean?.[j]||0))/(Number(tr.scale?.[j]||1)||1));
      if(tr.add_indicator){for(const idx of tr.indicator_features||[])base.push(Number.isFinite(Number(raw[idx]))?0:1)}x.push(...base);
    } else if(tr.name==='cat'){
      tr.columns.forEach((idx,j)=>{const v=String(row[idx]??tr.statistics?.[j]??'');for(const cat of tr.categories?.[j]||[])x.push(v===String(cat)?1:0)});
    }
  }
  let z=Number(model.intercept||0);for(let i=0;i<Math.min(x.length,model.coef?.length||0);i++)z+=x[i]*Number(model.coef[i]||0);return sigmoid(z);
}
async function loadHazards(){if(modelCache.hazards)return modelCache.hazards;if(modelCache.hazardPromise)return modelCache.hazardPromise;modelCache.hazardPromise=fetch('/bullpen-production-models.json',{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`models ${r.status}`);return r.json()}).then(x=>modelCache.hazards=x.models||null).catch(()=>null);return modelCache.hazardPromise}
loadHazards();
function inInningRemovalProbability(live,side,pid){
  const m=modelCache.hazards?.in_inning_removal;if(!m)return null;const ls=live?.linescore||{},cp=live?.plays?.currentPlay||{},box=live?.boxscore?.teams?.[side]||{};const ids=(box.pitchers||[]).map(Number),starter=ids[0]===Number(pid)?1:0;
  const stats=box.players?.[`ID${pid}`]?.stats?.pitching||{};const bf=Number(stats.battersFaced||0);const half=String(ls.isTopInning===false?'bottom':'top');const outs=Number(ls.outs??cp.count?.outs??0);const mask=(ls.offense?.first?1:0)|(ls.offense?.second?2:0)|(ls.offense?.third?4:0);const batter=cp.matchup?.batter?.id;let pos='';
  try{const order=live?.boxscore?.teams?.[ls.isTopInning===false?'home':'away']?.battingOrder||[];const ix=order.indexOf(batter);pos=ix>=0?String(ix+1):''}catch(_){pos=''}
  return transformHazard(m,[Number(ls.currentInning||1),bf,scoreDiff(live,side),half,String(outs),String(mask),pos,String(starter)]);
}
function betweenInningRemovalProbability(live,side,pid){
  const m=modelCache.hazards?.between_inning_removal;if(!m)return null;const ls=live?.linescore||{},box=live?.boxscore?.teams?.[side]||{};const ids=(box.pitchers||[]).map(Number),starter=ids[0]===Number(pid)?1:0;const stats=box.players?.[`ID${pid}`]?.stats?.pitching||{};const bf=Number(stats.battersFaced||0),pitches=Number(stats.numberOfPitches||0);
  // The exporter preserves the fitted feature order. Missing features remain null and are imputed by the frozen preprocessing statistics.
  return transformHazard(m,[Number(ls.currentInning||1),bf,scoreDiff(live,side),pitches,String(starter)]);
}
function currentHalfMixture(args){
  const {live,pitchingSide,lineup,startIdx,pitcherId,outs,mask,currentCount}=args||{};if(typeof window.runDistribution!=='function')return null;warmContext(live,pitchingSide);const base=window.runDistribution(lineup,startIdx,pitcherId,outs,mask,currentCount);const p=inInningRemovalProbability(live,pitchingSide,pitcherId);if(!(p>0))return base;const repl=weightedRunDistribution(live,pitchingSide,lineup,startIdx,outs,mask,currentCount,pitcherId);if(!repl)return base;return base.map((v,i)=>(1-p)*Number(v||0)+p*Number(repl[i]||0));
}
function nextHalfProjection(original,live,pitchingSide,battingLineup,startIdx,pregameSeed){
  warmContext(live,pitchingSide);const pid=currentPid(live,pitchingSide);if(!pid)return original(live,pitchingSide,battingLineup,startIdx,pregameSeed);const remove=betweenInningRemovalProbability(live,pitchingSide,pid);if(remove==null)return original(live,pitchingSide,battingLineup,startIdx,pregameSeed);
  const continueDist=window.runDistribution(battingLineup,startIdx,pid,0,0,null);const bullpenDist=window.runDistribution(battingLineup,startIdx,null,0,0,null);const cont=clamp(1-remove,0,1);const out=continueDist.map((v,i)=>cont*Number(v||0)+(1-cont)*Number(bullpenDist[i]||0));return {dist:out,pitcherId:pid,continueProb:cont,mode:'empirical-between-inning'};
}
window.empiricalBullpenEngine={warmContext,currentHalfMixture,nextHalfProjection,inInningRemovalProbability,betweenInningRemovalProbability,weightedRunDistribution,version:'validated-2025-hazards+frozen-2026-reliever-choice'};
})();

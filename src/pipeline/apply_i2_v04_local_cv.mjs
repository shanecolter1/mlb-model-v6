#!/usr/bin/env node
import fs from 'fs';

const inPath = process.env.I2_INPUT;
const outPath = process.env.I2_OUTPUT;
if (!inPath || !outPath) throw new Error('I2_INPUT and I2_OUTPUT are required');

const FIXED_EDGES = [
  0.5238396498423443,
  0.5491157480364879,
  0.5505588397195031,
  0.5722073331366149,
  0.5765829070767106,
  0.5802455678499445,
  0.5868605910212334,
  0.5886883820373516,
  0.5947180190516053
];
const BIN_SHRINKAGE = [200,10,10,10,10,10,10,10,10,10];

function clamp(p){ return Math.min(1-1e-12, Math.max(1e-12, p)); }
function logit(p){ p=clamp(p); return Math.log(p/(1-p)); }
function logistic(x){ return 1/(1+Math.exp(-Math.max(-40,Math.min(40,x)))); }
function american(p){
  p=clamp(p);
  return p>=0.5 ? Math.round(-100*p/(1-p)) : Math.round(100*(1-p)/p);
}
function binIndex(anchorUnder){
  let b=0;
  while (b<FIXED_EDGES.length && anchorUnder>=FIXED_EDGES[b]) b++;
  return Math.max(0,Math.min(9,b));
}

const src=JSON.parse(fs.readFileSync(inPath,'utf8'));
const rows=(src.ranking||[]).map(r=>{
  const env=r.runEnvironment||{};
  const n=Number(env.totalPriorN);
  const rawOver=Number(env.totalPriorOverPct)/100;
  const broadOver=Number(env.broadOverPct)/100;
  const delta=Number(env.baseballLogitDelta||0);
  if (!Number.isFinite(n)||n<=0||!Number.isFinite(rawOver)||!Number.isFinite(broadOver)) {
    return {...r, v04Calibration:{status:'SKIPPED_MISSING_TOTAL_PRIOR'}};
  }
  // Anchor exactly mirrors the historical Local-CV design: p100 total-prior Under,
  // before the baseball-context logit delta is applied.
  const over100=(rawOver*n + 100*broadOver)/(n+100);
  const under100=1-over100;
  const b=binIndex(under100);
  const s=BIN_SHRINKAGE[b];
  const overLocalPrior=(rawOver*n + s*broadOver)/(n+s);
  const overFinal=logistic(logit(overLocalPrior)+delta);
  const underFinal=1-overFinal;
  return {
    ...r,
    under05Pct:Number((underFinal*100).toFixed(2)),
    over05Pct:Number((overFinal*100).toFixed(2)),
    fairUnder:american(underFinal),
    fairOver:american(overFinal),
    v04Calibration:{
      status:'APPLIED',
      model:'I2 v0.4 Local-CV',
      anchorP100UnderPct:Number((under100*100).toFixed(4)),
      localCvBin:b+1,
      localCvShrinkage:s,
      rawTotalPriorUnderPct:Number(((1-rawOver)*100).toFixed(4)),
      localCvTotalPriorUnderPct:Number(((1-overLocalPrior)*100).toFixed(4)),
      baseballLogitDelta:delta,
      fixedEdges:FIXED_EDGES,
      binShrinkage:BIN_SHRINKAGE
    }
  };
});
rows.sort((a,b)=>Number(b.under05Pct)-Number(a.under05Pct));
rows.forEach((r,i)=>r.rank=i+1);
const out={
  ...src,
  model:'MLB I2 Under/Over v0.4 Local-CV Production',
  parentModel:src.model,
  localCvProduction:true,
  localCvValidationRelease:'i2-v0.4-production-validation-v1',
  localCvRule:{fixedEdges:FIXED_EDGES,binShrinkage:BIN_SHRINKAGE,anchor:'p100 total-prior Under before baseball delta'},
  ranking:rows
};
fs.writeFileSync(outPath,JSON.stringify(out,null,2));
console.log(JSON.stringify({model:out.model,date:out.date,projectedGames:rows.length,ranking:rows.map(r=>({rank:r.rank,matchup:r.matchup,under05Pct:r.under05Pct,over05Pct:r.over05Pct,total:r.runEnvironment?.fullGameTotal??null,shrinkage:r.v04Calibration?.localCvShrinkage??null,bin:r.v04Calibration?.localCvBin??null}))},null,2));

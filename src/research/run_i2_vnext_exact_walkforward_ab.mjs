#!/usr/bin/env node
/**
 * Reusable exact full-I2 A/B evaluator for governed walk-forward models.
 *
 * Supports model-family refit comparisons and park-HR shrinkage without Monte
 * Carlo noise. Both arms use the same replay cohort, empirical base/out
 * transitions, batting orders, prior-season park profiles, and leakage-safe
 * point-in-time model selection.
 */
import fs from 'node:fs';
import path from 'node:path';
import { buildNeutralEventVector, applyEnvironmentalEventVector, validateEventVector } from '../event_probability_engine.js';
import { predictI2EventVector } from '../model/i2_vnext_event_model.js';
import { venueForReplayGame } from './i2_vnext_replay_venue.mjs';

function arg(name,fallback=null){const i=process.argv.indexOf(name);return i>=0?process.argv[i+1]:fallback;}
const INPUT=arg('--input');
const BASE_MANIFEST=arg('--baseline-manifest');
const CAND_MANIFEST=arg('--candidate-manifest',BASE_MANIFEST);
const ARSENAL=arg('--arsenal');
const PARKS=arg('--parks');
const PLAY=arg('--play-calibration');
const OUTPUT=arg('--output');
const BASE_HR_ALPHA=Number(arg('--baseline-hr-alpha','1'));
const CAND_HR_ALPHA=Number(arg('--candidate-hr-alpha','1'));
const I1_PLAYER_RATES=arg('--i1-player-rates','active');
if(!INPUT||!BASE_MANIFEST||!ARSENAL||!PARKS||!PLAY||!OUTPUT) throw new Error('Missing required argument');
if(![BASE_HR_ALPHA,CAND_HR_ALPHA].every(x=>Number.isFinite(x)&&x>=0&&x<=1)) throw new Error('HR alpha must be in [0,1]');
if(!['active','neutralized'].includes(I1_PLAYER_RATES)) throw new Error('Unknown I1 player-rate mode');

const replay=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const arsenal=JSON.parse(fs.readFileSync(ARSENAL,'utf8'));
const parkProfiles=JSON.parse(fs.readFileSync(PARKS,'utf8'));
const rawPlay=JSON.parse(fs.readFileSync(PLAY,'utf8'));
if(replay.market_inputs_used!==false) throw new Error('Replay inputs are not market-isolated');

function loadWalkforward(manifestPath){
  const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
  if(manifest.market_inputs_used!==false) throw new Error('Walk-forward manifest is not market-isolated');
  const root=path.dirname(manifestPath);
  const models=new Map();
  for(const entry of manifest.entries||[]){
    const p=path.join(root,entry.model_file);
    const model=JSON.parse(fs.readFileSync(p,'utf8'));
    if(model.market_inputs_used!==false) throw new Error('Model is not market-isolated');
    models.set(entry.model_file,model);
  }
  return {manifest,models};
}
const baseWF=loadWalkforward(BASE_MANIFEST);
const candWF=loadWalkforward(CAND_MANIFEST);

function compactDate(x){return String(x??'').replace(/[^0-9]/g,'').slice(0,8);}
function modelForGame(wf,game){
  const d=compactDate(game.date); let chosen=null;
  for(const entry of wf.manifest.entries||[]){
    if(compactDate(entry.effective_from)<=d && (!chosen||compactDate(entry.effective_from)>compactDate(chosen.effective_from))) chosen=entry;
  }
  if(!chosen) throw new Error(`No model for ${game.gid} ${game.date}`);
  const model=wf.models.get(chosen.model_file);
  if(!model) throw new Error(`Missing model ${chosen.model_file}`);
  const end=compactDate(model?.training?.end);
  if(end&&end>=d) throw new Error(`Point-in-time leakage ${model.training.end} >= ${game.date}`);
  return model;
}

function adaptPlayCalibration(payload){
  if(payload?.base_transitions) return payload;
  if(!payload?.states) throw new Error('Unsupported transition artifact');
  const eventMap={strikeout:'out',ball_in_play_out:'out',walk:'bb',hit_by_pitch:'bb',single:'single',double:'double',triple:'triple',home_run:'hr'};
  const base_transitions={};
  for(const [event,source] of Object.entries(eventMap)) for(let outs=0;outs<3;outs++) for(let mask=0;mask<8;mask++){
    base_transitions[`${event}|${outs}|${mask}`]=payload.states[`${source}|${outs}|${mask}`]||[];
  }
  return {version:`adapted-${payload.version||'validated'}`,base_transitions};
}
const play=adaptPlayCalibration(rawPlay);
const league=replay?.i1_state_model?.event_rates;
if(!league) throw new Error('Missing league I1 event rates');

function norm(x){return String(x??'').trim().toUpperCase();}
function stand(bats,throws){
  const b=norm(bats),t=norm(throws);
  if(b==='B') return t==='L'?'R':'L';
  return b==='L'?'L':'R';
}
function nextSlot(slot){return slot===9?1:slot+1;}
function stateIndex(outs,mask,slot){return ((outs*8+mask)*9)+(slot-1);}
function transitions(event,outs,mask){
  const x=play.base_transitions?.[`${event}|${outs}|${mask}`];
  if(!Array.isArray(x)||!x.length) throw new Error(`Missing transition ${event}|${outs}|${mask}`);
  return x;
}
function shrinkFactor(x,alpha){
  const n=Number(x);
  return Number.isFinite(n)?1+alpha*(n-1):x;
}
function parkWithHrAlpha(context,alpha){
  if(!context||alpha===1) return context;
  const out=JSON.parse(JSON.stringify(context));
  if(out.multipliers && out.multipliers.hr!==undefined) out.multipliers.hr=shrinkFactor(out.multipliers.hr,alpha);
  for(const side of ['L','R']) if(out.handedness?.[side]?.hr!==undefined) out.handedness[side].hr=shrinkFactor(out.handedness[side].hr,alpha);
  return out;
}
function makeLineup(rows){return rows.map((x,i)=>({
  id:Number(x.mlbam),bats:x.bats,
  eventRates:I1_PLAYER_RATES==='neutralized'?league:(x.i1_event_rates_asof||(()=>{throw new Error(`Missing hitter rates ${x.mlbam||i}`)})())
}));}
function makePitcher(x){return {
  id:Number(x.mlbam),throws:x.throws,
  eventRatesAllowed:I1_PLAYER_RATES==='neutralized'?league:(x.i1_event_rates_asof||(()=>{throw new Error(`Missing pitcher rates ${x.mlbam}`)})())
};}
function i1Provider(environmentalContext){
  return ({batter,pitcher})=>{
    const neutral=buildNeutralEventVector({batter:batter.eventRates,pitcher:pitcher.eventRatesAllowed,league,weights:{batter:0.5,pitcher:0.5}});
    return applyEnvironmentalEventVector({neutralVector:neutral,environmentalContext,batterSide:stand(batter.bats,pitcher.throws)}).probabilities;
  };
}
function i2Provider(model,environmentalContext){
  return ({batter,pitcher})=>{
    const batterSide=stand(batter.bats,pitcher.throws);
    const neutral=predictI2EventVector({batterId:batter.id,pitcherId:pitcher.id,batterSide,pitcherThrows:pitcher.throws,model,arsenalProfile:arsenal});
    return applyEnvironmentalEventVector({neutralVector:neutral,environmentalContext,batterSide}).probabilities;
  };
}

function exactNextSlotDistribution({lineup,pitcher,eventVectorForPA}){
  const N=3*8*9;
  const edges=Array.from({length:N},()=>[]);
  const absorb=Array.from({length:N},()=>Array(10).fill(0));
  const vectors=Array(10);
  for(let slot=1;slot<=9;slot++){vectors[slot]=eventVectorForPA({batter:lineup[slot-1],pitcher});validateEventVector(vectors[slot]);}
  for(let outs=0;outs<3;outs++) for(let mask=0;mask<8;mask++) for(let slot=1;slot<=9;slot++){
    const i=stateIndex(outs,mask,slot),ns=nextSlot(slot),v=vectors[slot];
    for(const [event,pe] of Object.entries(v)) if(pe>0) for(const tr of transitions(event,outs,mask)){
      const q=pe*Number(tr.p||0); if(!(q>0)) continue;
      const no=Math.min(3,outs+Number(tr.outs_added||0)),nm=Number(tr.post_mask||0);
      if(no>=3) absorb[i][ns]+=q; else edges[i].push([stateIndex(no,nm,ns),q]);
    }
  }
  let f=Array.from({length:N},()=>Array(10).fill(0));
  let converged=false;
  for(let iter=0;iter<500;iter++){
    const nf=Array.from({length:N},()=>Array(10).fill(0)); let diff=0;
    for(let i=0;i<N;i++) for(let s=1;s<=9;s++){
      let x=absorb[i][s];
      for(const [j,w] of edges[i]) x+=w*f[j][s];
      nf[i][s]=x; diff=Math.max(diff,Math.abs(x-f[i][s]));
    }
    f=nf;if(diff<1e-13){converged=true;break;}
  }
  if(!converged) throw new Error('I1 slot propagation failed to converge');
  const out=Array(10).fill(0);
  for(let s=1;s<=9;s++) out[s]=f[stateIndex(0,0,1)][s];
  const total=out.reduce((a,b)=>a+b,0);
  if(Math.abs(total-1)>1e-8) throw new Error(`I1 slot mass ${total}`);
  return out;
}
function exactScorelessByStartSlot({lineup,pitcher,eventVectorForPA}){
  const N=3*8*9,edges=Array.from({length:N},()=>[]),absorb=new Float64Array(N),vectors=Array(10);
  for(let slot=1;slot<=9;slot++){vectors[slot]=eventVectorForPA({batter:lineup[slot-1],pitcher});validateEventVector(vectors[slot]);}
  for(let outs=0;outs<3;outs++) for(let mask=0;mask<8;mask++) for(let slot=1;slot<=9;slot++){
    const i=stateIndex(outs,mask,slot),ns=nextSlot(slot),v=vectors[slot];
    for(const [event,pe] of Object.entries(v)) if(pe>0) for(const tr of transitions(event,outs,mask)){
      if(Number(tr.runs||0)>0) continue;
      const q=pe*Number(tr.p||0);if(!(q>0)) continue;
      const no=Math.min(3,outs+Number(tr.outs_added||0)),nm=Number(tr.post_mask||0);
      if(no>=3) absorb[i]+=q; else edges[i].push([stateIndex(no,nm,ns),q]);
    }
  }
  let f=new Float64Array(N),converged=false;
  for(let iter=0;iter<500;iter++){
    const nf=new Float64Array(N);let diff=0;
    for(let i=0;i<N;i++){let x=absorb[i];for(const [j,w] of edges[i]) x+=w*f[j];nf[i]=x;diff=Math.max(diff,Math.abs(x-f[i]));}
    f=nf;if(diff<1e-13){converged=true;break;}
  }
  if(!converged) throw new Error('I2 scoreless propagation failed to converge');
  const out=Array(10).fill(0);for(let s=1;s<=9;s++) out[s]=f[stateIndex(0,0,s)];return out;
}
function weighted(slotDist,p0){let x=0;for(let s=1;s<=9;s++) x+=slotDist[s]*p0[s];return x;}
function scoreArm(game,model,park){
  const awayLineup=makeLineup(game.away_lineup),homeLineup=makeLineup(game.home_lineup);
  const awayPitcher=makePitcher(game.away_starter),homePitcher=makePitcher(game.home_starter);
  const i1=i1Provider(park),i2=i2Provider(model,park);
  const awaySlots=exactNextSlotDistribution({lineup:awayLineup,pitcher:homePitcher,eventVectorForPA:i1});
  const homeSlots=exactNextSlotDistribution({lineup:homeLineup,pitcher:awayPitcher,eventVectorForPA:i1});
  const awayP0=exactScorelessByStartSlot({lineup:awayLineup,pitcher:homePitcher,eventVectorForPA:i2});
  const homeP0=exactScorelessByStartSlot({lineup:homeLineup,pitcher:awayPitcher,eventVectorForPA:i2});
  return weighted(awaySlots,awayP0)*weighted(homeSlots,homeP0);
}
function loss(y,p){const q=Math.max(1e-12,Math.min(1-1e-12,p));return -(y*Math.log(q)+(1-y)*Math.log1p(-q));}

const predictions=[];
let baseLL=0,candLL=0,baseB=0,candB=0;
for(const game of replay.games||[]){
  const venue=venueForReplayGame(game,parkProfiles,replay.season);
  const basePark=parkWithHrAlpha(venue.profile,BASE_HR_ALPHA);
  const candPark=parkWithHrAlpha(venue.profile,CAND_HR_ALPHA);
  const bp=scoreArm(game,modelForGame(baseWF,game),basePark);
  const cp=scoreArm(game,modelForGame(candWF,game),candPark);
  const y=Number(game.observed.under05);
  baseLL+=loss(y,bp);candLL+=loss(y,cp);baseB+=(bp-y)**2;candB+=(cp-y)**2;
  predictions.push({gid:game.gid,date:game.date,observed_under05:y,baseline_under05:bp,candidate_under05:cp});
}
const n=predictions.length;
const payload={
  version:'i2-vnext-exact-walkforward-ab-v1',generated_at:new Date().toISOString(),
  season:replay.season,n,market_inputs_used:false,method:'exact deterministic state propagation; no Monte Carlo',
  baseline_manifest:baseWF.manifest.version,candidate_manifest:candWF.manifest.version,
  baseline_hr_alpha:BASE_HR_ALPHA,candidate_hr_alpha:CAND_HR_ALPHA,i1_player_rates:I1_PLAYER_RATES,
  baseline_logloss:baseLL/n,candidate_logloss:candLL/n,candidate_minus_baseline_logloss:(candLL-baseLL)/n,
  baseline_brier:baseB/n,candidate_brier:candB/n,candidate_minus_baseline_brier:(candB-baseB)/n,
  predictions
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});fs.writeFileSync(OUTPUT,JSON.stringify(payload,null,2)+'\n');
console.log(JSON.stringify({...payload,predictions:undefined},null,2));

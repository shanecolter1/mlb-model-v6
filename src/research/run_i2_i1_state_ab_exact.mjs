#!/usr/bin/env node
/**
 * Exact A/B for I1 state generation.
 *
 * Compares:
 * A) league-average I1 event vector
 * B) point-in-time player-specific I1 event vectors
 *
 * Everything after the I2 starting slot is held fixed. The same frozen direct-I2
 * model, arsenal feature, Savant park profile, and empirical base/out transition
 * table are used in both arms. Exact state propagation avoids Monte Carlo noise.
 */
import fs from 'node:fs';
import path from 'node:path';
import { buildNeutralEventVector, applyEnvironmentalEventVector, validateEventVector } from '../event_probability_engine.js';
import { predictI2EventVector } from '../model/i2_vnext_event_model.js';
import { venueForReplayGame } from './i2_vnext_replay_venue.mjs';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input','data/derived/i2_vnext/replay_2025_inputs_i1_ab.json');
const MODEL=arg('--model','data/derived/i2_vnext/i2_vnext_event_model.json');
const ARSENAL=arg('--arsenal','data/derived/i2_vnext/arsenal_profile_2024.json');
const PARKS=arg('--parks','data/derived/i2_vnext/park/savant_venue_profiles_2024_3yr.json');
const PLAY=arg('--play-calibration','data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json');
const OUTPUT=arg('--output','data/derived/i2_vnext/i1_state_ab_exact_predictions.json');
const DIRECT_PA_MODEL=arg('--direct-i1-pa-model');

const replay=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const model=JSON.parse(fs.readFileSync(MODEL,'utf8'));
const arsenal=JSON.parse(fs.readFileSync(ARSENAL,'utf8'));
const rawPlay=JSON.parse(fs.readFileSync(PLAY,'utf8'));
const parkProfiles=fs.existsSync(PARKS) ? JSON.parse(fs.readFileSync(PARKS,'utf8')) : [];

function adaptPlayCalibration(payload){
  if (payload?.base_transitions) return payload;
  if (!payload?.states) throw new Error('Unsupported transition artifact');
  const eventMap={
    strikeout:'out', ball_in_play_out:'out', walk:'bb', hit_by_pitch:'bb',
    single:'single', double:'double', triple:'triple', home_run:'hr',
  };
  const base_transitions={};
  for (const [event,source] of Object.entries(eventMap)) {
    for (let outs=0; outs<3; outs+=1) {
      for (let mask=0; mask<8; mask+=1) {
        base_transitions[`${event}|${outs}|${mask}`]=payload.states[`${source}|${outs}|${mask}`] || [];
      }
    }
  }
  return {version:`adapted-${payload.version || 'validated'}`,base_transitions};
}
const play=adaptPlayCalibration(rawPlay);
const league=replay?.i1_state_model?.event_rates;
if (!league) throw new Error('Missing league I1 event rates');
if (replay.market_inputs_used!==false || model.market_inputs_used!==false) {
  throw new Error('Full-I2 inputs or model are not market-isolated');
}
const directModel=DIRECT_PA_MODEL ? JSON.parse(fs.readFileSync(DIRECT_PA_MODEL,'utf8')) : null;
const EVENTS=['single','double','triple','home_run','walk','hit_by_pitch','strikeout','ball_in_play_out'];
if (directModel) {
  if (directModel.market_inputs_used!==false || directModel.observed_2024_outcomes_used_for_fit!==false ||
      directModel.status!=='RESEARCH_ONLY_NOT_PROMOTED' ||
      JSON.stringify(directModel.training_years)!==JSON.stringify([2022,2023]) ||
      !directModel.training_years.every(y=>y<replay.season) ||
      JSON.stringify(directModel.event_order)!==JSON.stringify(EVENTS)) {
    throw new Error('Direct I1 PA model is not governed for this replay');
  }
}

function norm(x){ return String(x??'').trim().toUpperCase(); }
function stand(bats, throws){
  const b=norm(bats), t=norm(throws);
  if (b==='B') return t==='L' ? 'R' : 'L';
  return b==='L' ? 'L' : 'R';
}
function nextSlot(slot){ return slot===9 ? 1 : slot+1; }
function stateKey(outs,mask,slot){ return `${outs}|${mask}|${slot}`; }
function parseState(key){ return key.split('|').map(Number); }

function transitions(event, outs, mask){
  const x=play.base_transitions?.[`${event}|${outs}|${mask}`];
  if (!Array.isArray(x) || !x.length) throw new Error(`Missing transition ${event}|${outs}|${mask}`);
  return x;
}

function exactNextSlotDistribution({lineup,pitcher,eventVectorForPA}){
  let active=new Map([[stateKey(0,0,1),1]]);
  const absorbed=Array(10).fill(0);
  for (let pa=0; pa<40 && active.size; pa+=1) {
    const next=new Map();
    for (const [key,mass] of active) {
      const [outs,mask,slot]=parseState(key);
      const batter=lineup[slot-1];
      const vector=eventVectorForPA({batter,pitcher});
      validateEventVector(vector);
      const ns=nextSlot(slot);
      for (const [event,pe] of Object.entries(vector)) {
        if (!(pe>0)) continue;
        for (const tr of transitions(event,outs,mask)) {
          const q=mass*pe*Number(tr.p||0);
          if (!(q>0)) continue;
          const no=Math.min(3,outs+Number(tr.outs_added||0));
          const nm=Number(tr.post_mask||0);
          if (no>=3) absorbed[ns]+=q;
          else {
            const nk=stateKey(no,nm,ns);
            next.set(nk,(next.get(nk)||0)+q);
          }
        }
      }
    }
    active=next;
  }
  const residual=[...active.values()].reduce((a,b)=>a+b,0);
  if (residual>1e-7) throw new Error(`I1 exact propagation residual too large: ${residual}`);
  const total=absorbed.reduce((a,b)=>a+b,0);
  if (!(total>0.999999 && total<1.000001)) throw new Error(`I1 absorption mass ${total}`);
  return absorbed.map(x=>x/total);
}

function exactScorelessProbability({lineup,startSlot,pitcher,eventVectorForPA}){
  let active=new Map([[stateKey(0,0,startSlot),1]]);
  let absorbed0=0;
  for (let pa=0; pa<40 && active.size; pa+=1) {
    const next=new Map();
    for (const [key,mass] of active) {
      const [outs,mask,slot]=parseState(key);
      const batter=lineup[slot-1];
      const vector=eventVectorForPA({batter,pitcher});
      validateEventVector(vector);
      const ns=nextSlot(slot);
      for (const [event,pe] of Object.entries(vector)) {
        if (!(pe>0)) continue;
        for (const tr of transitions(event,outs,mask)) {
          if (Number(tr.runs||0)>0) continue;
          const q=mass*pe*Number(tr.p||0);
          if (!(q>0)) continue;
          const no=Math.min(3,outs+Number(tr.outs_added||0));
          const nm=Number(tr.post_mask||0);
          if (no>=3) absorbed0+=q;
          else {
            const nk=stateKey(no,nm,ns);
            next.set(nk,(next.get(nk)||0)+q);
          }
        }
      }
    }
    active=next;
  }
  const residual=[...active.values()].reduce((a,b)=>a+b,0);
  if (residual>1e-7) throw new Error(`I2 scoreless propagation residual too large: ${residual}`);
  return absorbed0;
}

function makeLineup(rows){ return rows.map(x=>({id:Number(x.mlbam),bats:x.bats,retro:x.retro,i1:x.i1_event_rates_asof})); }
function makePitcher(x){ return {id:Number(x.mlbam),throws:x.throws,retro:x.retro,i1:x.i1_event_rates_asof}; }

function i1Vector(mode,{batter,pitcher},side){
  const br=mode==='player_asof' ? (batter.i1 || league) : league;
  const pr=mode==='player_asof' ? (pitcher.i1 || league) : league;
  if (mode==='direct') {
    if (!batter.i1 || !pitcher.i1) throw new Error('Direct I1 model requires player-as-of rates');
    const p=directModel.parameters;
    const logit=x=>Math.log(x/(1-x));
    const logits=EVENTS.map(event=>Math.log(league[event]) +
      p.batter_weight*(logit(batter.i1[event])-logit(league[event])) +
      p.pitcher_weight*(logit(pitcher.i1[event])-logit(league[event])) +
      (p.event_intercepts[event] ?? 0) +
      (side==='bottom' ? (p.home_event_terms[event] ?? 0) : 0));
    const max=Math.max(...logits);
    const weights=logits.map(v=>Math.exp(v-max));
    const total=weights.reduce((a,b)=>a+b,0);
    return Object.fromEntries(EVENTS.map((e,i)=>[e,weights[i]/total]));
  }
  return buildNeutralEventVector({batter:br,pitcher:pr,league,weights:{batter:0.5,pitcher:0.5}});
}
function i2Provider(environmentalContext){
  return ({batter,pitcher})=>{
    const batterSide=stand(batter.bats,pitcher.throws);
    const neutral=predictI2EventVector({
      batterId:batter.id,pitcherId:pitcher.id,batterSide,pitcherThrows:pitcher.throws,
      model,arsenalProfile:arsenal,
    });
    return applyEnvironmentalEventVector({
      neutralVector:neutral,environmentalContext,batterSide,
    }).probabilities;
  };
}
function stateIndex(outs,mask,slot){ return ((outs*8+mask)*9)+(slot-1); }

function exactScorelessByStartSlot({lineup,pitcher,eventVectorForPA}){
  const N=3*8*9;
  const edges=Array.from({length:N},()=>[]);
  const absorb=new Float64Array(N);
  const vectors=Array(10);
  for(let slot=1;slot<=9;slot+=1){
    const v=eventVectorForPA({batter:lineup[slot-1],pitcher});
    validateEventVector(v);
    vectors[slot]=v;
  }
  for(let outs=0;outs<3;outs+=1){
    for(let mask=0;mask<8;mask+=1){
      for(let slot=1;slot<=9;slot+=1){
        const i=stateIndex(outs,mask,slot);
        const ns=nextSlot(slot);
        const vector=vectors[slot];
        for(const [event,pe] of Object.entries(vector)){
          if(!(pe>0)) continue;
          for(const tr of transitions(event,outs,mask)){
            if(Number(tr.runs||0)>0) continue;
            const q=pe*Number(tr.p||0);
            if(!(q>0)) continue;
            const no=Math.min(3,outs+Number(tr.outs_added||0));
            const nm=Number(tr.post_mask||0);
            if(no>=3) absorb[i]+=q;
            else edges[i].push([stateIndex(no,nm,ns),q]);
          }
        }
      }
    }
  }
  let f=new Float64Array(N);
  let converged=false;
  for(let iter=0;iter<500;iter+=1){
    const nf=new Float64Array(N);
    let diff=0;
    for(let i=0;i<N;i+=1){
      let x=absorb[i];
      for(const [j,w] of edges[i]) x+=w*f[j];
      nf[i]=x;
      diff=Math.max(diff,Math.abs(x-f[i]));
    }
    f=nf;
    if(diff<1e-13){ converged=true; break; }
  }
  if(!converged) throw new Error('I2 scoreless fixed-point iteration did not converge');
  const out=Array(10).fill(0);
  for(let slot=1;slot<=9;slot+=1) out[slot]=f[stateIndex(0,0,slot)];
  return out;
}

function weightedP0(slotDist,p0BySlot){
  let x=0;
  for (let slot=1;slot<=9;slot+=1) x+=Number(slotDist[slot]||0)*Number(p0BySlot[slot]||0);
  return x;
}
function meanAbsDiff(a,b){
  let s=0;
  for(let i=1;i<=9;i+=1) s+=Math.abs(Number(a[i]||0)-Number(b[i]||0));
  return s/2;
}

const predictions=[];
let parkMatched=0;
for (const game of replay.games) {
  const env=venueForReplayGame(game,parkProfiles).profile;
  if (env) parkMatched+=1;
  const awayLineup=makeLineup(game.away_lineup);
  const homeLineup=makeLineup(game.home_lineup);
  const awayPitcher=makePitcher(game.away_starter);
  const homePitcher=makePitcher(game.home_starter);

  // Away offense faces home pitcher; home offense faces away pitcher.
  const awayLeague=exactNextSlotDistribution({
    lineup:awayLineup,pitcher:homePitcher,eventVectorForPA:x=>i1Vector('league',x),
  });
  const homeLeague=exactNextSlotDistribution({
    lineup:homeLineup,pitcher:awayPitcher,eventVectorForPA:x=>i1Vector('league',x),
  });
  const awayPlayer=exactNextSlotDistribution({
    lineup:awayLineup,pitcher:homePitcher,eventVectorForPA:x=>i1Vector('player_asof',x),
  });
  const homePlayer=exactNextSlotDistribution({
    lineup:homeLineup,pitcher:awayPitcher,eventVectorForPA:x=>i1Vector('player_asof',x),
  });
  const awayDirect=directModel ? exactNextSlotDistribution({
    lineup:awayLineup,pitcher:homePitcher,eventVectorForPA:x=>i1Vector('direct',x,'top'),
  }) : null;
  const homeDirect=directModel ? exactNextSlotDistribution({
    lineup:homeLineup,pitcher:awayPitcher,eventVectorForPA:x=>i1Vector('direct',x,'bottom'),
  }) : null;

  const provider=i2Provider(env);
  const awayP0=exactScorelessByStartSlot({lineup:awayLineup,pitcher:homePitcher,eventVectorForPA:provider});
  const homeP0=exactScorelessByStartSlot({lineup:homeLineup,pitcher:awayPitcher,eventVectorForPA:provider});

  const leagueTop0=weightedP0(awayLeague,awayP0);
  const leagueBot0=weightedP0(homeLeague,homeP0);
  const playerTop0=weightedP0(awayPlayer,awayP0);
  const playerBot0=weightedP0(homePlayer,homeP0);
  const row={
    gid:game.gid,date:game.date,observed_under05:Number(game.observed.under05),
    league_under05:leagueTop0*leagueBot0,
    player_asof_under05:playerTop0*playerBot0,
    away_start_slot_tv:meanAbsDiff(awayLeague,awayPlayer),
    home_start_slot_tv:meanAbsDiff(homeLeague,homePlayer),
  };
  if (directModel) {
    row.direct_under05=weightedP0(awayDirect,awayP0)*weightedP0(homeDirect,homeP0);
    row.direct_top0=weightedP0(awayDirect,awayP0);
    row.direct_bottom0=weightedP0(homeDirect,homeP0);
    row.player_top0=playerTop0;
    row.player_bottom0=playerBot0;
    row.away_direct_slot_tv=meanAbsDiff(awayPlayer,awayDirect);
    row.home_direct_slot_tv=meanAbsDiff(homePlayer,homeDirect);
  }
  predictions.push(row);
}

const payload={
  version:'i2-i1-state-ab-exact-v1',
  generated_at:new Date().toISOString(),
  season:2025,
  games:predictions.length,
  market_inputs_used:false,
  downstream_i2_model:model.version,
  downstream_i2_training:model.training,
  method:'exact finite-state propagation through empirical PA transitions; no Monte Carlo',
  i1_arms:{
    league:'prior-season league-average event vector',
    player_asof:'season-to-date player event rates strictly before game date, 100 PA hitter prior and 180 BF pitcher prior',
  },
  i1_environment:'neutral in both arms to isolate player-rate incremental value',
  direct_i1_pa_model:directModel?.version || null,
  park_rule_i2:'prior-season Savant event-vector park profile applied exactly once',
  park_match_rate:predictions.length ? parkMatched/predictions.length : null,
  predictions,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(payload));
console.log(JSON.stringify({
  games:payload.games,method:payload.method,park_match_rate:payload.park_match_rate,
  mean_start_slot_tv:predictions.reduce((s,r)=>s+(r.away_start_slot_tv+r.home_start_slot_tv)/2,0)/predictions.length,
},null,2));

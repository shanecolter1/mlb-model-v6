#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { buildNeutralEventVector, applyEnvironmentalEventVector } from '../event_probability_engine.js';
import { venueForReplayGame } from './i2_vnext_replay_venue.mjs';

function arg(name, fallback=null) {
  const i=process.argv.indexOf(name);
  return i>=0 ? process.argv[i+1] : fallback;
}
const INPUT=arg('--input','data/derived/i2_vnext/i1_state_ab_2024_inputs.json');
const PLAY=arg('--play-calibration','data/derived/model_calibration/seasonal/production_pa_transition_table_shrunk.json');
const OUTPUT=arg('--output','data/derived/i2_vnext/i1_state_ab_2024_rows.json');
const OBSERVED_SLOTS=arg('--observed-slots');
const PARKS=arg('--parks');
const DIRECT_PA_MODEL=arg('--direct-i1-pa-model');
const MAX_PA=Number(arg('--max-pa','40'));

const input=JSON.parse(fs.readFileSync(INPUT,'utf8'));
const raw=JSON.parse(fs.readFileSync(PLAY,'utf8'));
const league=input.league_event_rates ?? input.i1_state_model?.event_rates;
if (!league) throw new Error('Input missing pregame league event rates');
if (!raw?.states) throw new Error('Expected seasonal transition artifact with states');
if (input.market_inputs_used !== false) throw new Error('Input is not market-isolated');
const parkProfiles=PARKS ? JSON.parse(fs.readFileSync(PARKS,'utf8')) : [];
if (PARKS && (!Array.isArray(parkProfiles) || !parkProfiles.length || input.season!==2025)) {
  throw new Error('Park-aware I1 audit requires 2025 replay inputs and prior-season park profiles');
}
const directModel=DIRECT_PA_MODEL ? JSON.parse(fs.readFileSync(DIRECT_PA_MODEL,'utf8')) : null;
if (directModel) {
  if (directModel.market_inputs_used!==false || directModel.observed_2024_outcomes_used_for_fit!==false ||
      directModel.status!=='RESEARCH_ONLY_NOT_PROMOTED' ||
      !directModel.training_years?.every(y=>y<input.season)) {
    throw new Error('Direct PA research model is not governed for this season');
  }
}

// The compact Retrosheet file is used strictly after computing each game's
// pregame slot distribution. It supplies targets, never model features.
const observedByGame=new Map();
if (OBSERVED_SLOTS) {
  const [header,...lines]=fs.readFileSync(OBSERVED_SLOTS,'utf8').trim().split(/\r?\n/);
  const fields=header.split(',');
  for (const key of ['gid','half','i2_start_slot']) {
    if (!fields.includes(key)) throw new Error(`Observed slot CSV missing ${key}`);
  }
  for (const line of lines) {
    const values=line.split(',');
    const row=Object.fromEntries(fields.map((key,index)=>[key,values[index]]));
    const side=row.half;
    if (!['top','bottom'].includes(side)) continue;
    const key=`${row.gid}|${side}`;
    if (observedByGame.has(key)) throw new Error(`Duplicate observed slot: ${key}`);
    observedByGame.set(key,Number(row.i2_start_slot));
  }
}

const EVENT_TO_TRANSITION={
  strikeout:'out',
  ball_in_play_out:'out',
  walk:'bb',
  hit_by_pitch:'bb',
  single:'single',
  double:'double',
  triple:'triple',
  home_run:'hr',
};
const EVENTS=Object.keys(EVENT_TO_TRANSITION);
if (directModel && JSON.stringify([...directModel.event_order].sort())!==JSON.stringify([...EVENTS].sort())) {
  throw new Error('Direct PA event order mismatch');
}
const sourceForEvent=(event,outs,mask)=>
  raw.states[`${event}|${outs}|${mask}`] ? event : EVENT_TO_TRANSITION[event];

// For starting-slot prediction, run totals are irrelevant. Collapse transition
// rows that lead to the same outs/base state once, globally.
const transitionCache=new Map();
function collapsedTransitions(source,outs,mask){
  const key=`${source}|${outs}|${mask}`;
  if (transitionCache.has(key)) return transitionCache.get(key);
  const rows=raw.states[key];
  if (!Array.isArray(rows) || !rows.length) throw new Error(`Missing transition state ${key}`);
  const grouped=new Map();
  for(const t of rows){
    const newOuts=Math.min(3,outs+Math.max(0,Number(t.outs_added||0)));
    const postMask=newOuts>=3?0:Number(t.post_mask||0);
    const k=`${newOuts}|${postMask}`;
    grouped.set(k,(grouped.get(k)||0)+Number(t.p||0));
  }
  const out=[...grouped.entries()].map(([k,p])=>{
    const [o,m]=k.split('|').map(Number);
    return {outs:o,mask:m,p};
  });
  transitionCache.set(key,out);
  return out;
}
for(const source of new Set([...Object.values(EVENT_TO_TRANSITION),
  ...EVENTS.filter(event=>raw.states[`${event}|0|0`])])){
  for(let outs=0;outs<3;outs++) for(let mask=0;mask<8;mask++) collapsedTransitions(source,outs,mask);
}

function rates(item, mode){
  return mode==='player_asof' ? item.i1_event_rates_asof : league;
}
function directEventVector(batterRates,pitcherRates,side){
  const p=directModel.parameters;
  const logit=x=>Math.log(x/(1-x));
  const logits=EVENTS.map(event=>
    Math.log(league[event]) +
    p.batter_weight*(logit(batterRates[event])-logit(league[event])) +
    p.pitcher_weight*(logit(pitcherRates[event])-logit(league[event])) +
    (p.event_intercepts[event] ?? 0) +
    (side==='bottom' ? (p.home_event_terms[event] ?? 0) : 0)
  );
  const maximum=Math.max(...logits);
  const weights=logits.map(value=>Math.exp(value-maximum));
  const total=weights.reduce((sum,value)=>sum+value,0);
  return Object.fromEntries(EVENTS.map((event,index)=>[event,weights[index]/total]));
}
function precomputeVectors(lineup,pitcher,mode,environmentalContext,side){
  return lineup.map(batter=>{
    const neutralVector=directModel && mode==='player_asof'
      ? directEventVector(rates(batter,mode),rates(pitcher,mode),side)
      : buildNeutralEventVector({
      batter:rates(batter,mode),
      pitcher:rates(pitcher,mode),
      league,
      weights:{batter:0.5,pitcher:0.5},
    });
    if (!environmentalContext) return neutralVector;
    const bats=String(batter.bats??'').toUpperCase();
    const throws=String(pitcher.throws??'').toUpperCase();
    const batterSide=bats==='B' || bats==='S' ? (throws==='L' ? 'R' : 'L') : bats;
    return applyEnvironmentalEventVector({
      neutralVector,environmentalContext,batterSide,
    }).probabilities;
  });
}
function idx(outs,mask){ return outs*8+mask; }

let leagueDistributionCache=null;
function exactDistribution(game,side,mode){
  if(mode==='league' && !PARKS && leagueDistributionCache) return leagueDistributionCache;

  const isTop=side==='top';
  const lineup=isTop?game.away_lineup:game.home_lineup;
  const pitcher=isTop?game.home_starter:game.away_starter;
  const venue=PARKS ? venueForReplayGame(game,parkProfiles) : {profile:null};
  const vectors=precomputeVectors(lineup,pitcher,mode,venue.profile,side);

  let active=new Float64Array(24);
  active[idx(0,0)]=1;
  const absorbed=new Float64Array(10);

  for(let pa=0;pa<MAX_PA;pa++){
    const next=new Float64Array(24);
    const slot=(pa%9)+1;
    const nextSlot=slot===9?1:slot+1;
    const vector=vectors[slot-1];
    let activeMass=0;

    for(let outs=0;outs<3;outs++){
      for(let mask=0;mask<8;mask++){
        const stateProb=active[idx(outs,mask)];
        if(!(stateProb>0)) continue;
        activeMass+=stateProb;
        for(const event of EVENTS){
          const eventProb=Number(vector[event]||0);
          if(!(eventProb>0)) continue;
          const source=sourceForEvent(event,outs,mask);
          for(const t of collapsedTransitions(source,outs,mask)){
            const p=stateProb*eventProb*t.p;
            if(!(p>0)) continue;
            if(t.outs>=3) absorbed[nextSlot]+=p;
            else next[idx(t.outs,t.mask)]+=p;
          }
        }
      }
    }
    active=next;
    if(activeMass<1e-14) break;
  }

  const tail=active.reduce((a,b)=>a+b,0);
  if(tail>1e-8) throw new Error(`Exact I1 state tail too large after ${MAX_PA} PA: ${tail}`);
  const total=absorbed.reduce((a,b)=>a+b,0);
  if(!(total>0.999999 && total<=1.000001)) throw new Error(`I1 slot distribution mass=${total}`);
  const result=Object.fromEntries(Array.from({length:9},(_,j)=>[String(j+1),absorbed[j+1]/total]));
  if(mode==='league' && !PARKS) leagueDistributionCache=result;
  return result;
}

function loss(dist, observed){
  const p=Math.max(1e-15,Number(dist[String(observed)]||0));
  let brier=0;
  for(let s=1;s<=9;s++){
    const q=Number(dist[String(s)]||0);
    const y=s===observed?1:0;
    brier+=(q-y)**2;
  }
  const best=Object.entries(dist).sort((a,b)=>b[1]-a[1])[0][0];
  return {logloss:-Math.log(p),brier,observed_probability:p,top1:Number(best)===observed};
}

const rows=[];
for(const game of input.games){
  for(const side of ['top','bottom']){
    const observed=OBSERVED_SLOTS
      ? observedByGame.get(`${game.gid}|${side}`)
      : Number(game.observed?.[side==='top'?'top2_start_slot':'bottom2_start_slot']);
    if(!(Number.isInteger(observed) && observed>=1 && observed<=9)) {
      if (OBSERVED_SLOTS) throw new Error(`Missing observed slot for ${game.gid}|${side}`);
      continue;
    }
    const leagueDist=exactDistribution(game,side,'league');
    const playerDist=exactDistribution(game,side,'player_asof');
    rows.push({
      gid:game.gid,date:game.date,side,observed_slot:observed,
      league:{...loss(leagueDist,observed),distribution:leagueDist},
      player_asof:{...loss(playerDist,observed),distribution:playerDist},
    });
  }
}
const payload={
  version:'i1-state-ab-evaluation-v3-exact-optimized',
  generated_at:new Date().toISOString(),
  season:input.season,
  market_inputs_used:false,
  observed_i2_start_slot_used_as_predictor:false,
  observed_slot_source:OBSERVED_SLOTS ? 'joined Retrosheet compact CSV target only' : 'input target only',
  i1_environment:PARKS ? 'prior_season_park' : 'neutral',
  i1_pa_model:directModel?.version || 'existing_50_50_formula',
  evaluation_method:'exact dynamic propagation through validated empirical event/base-out transition table; run-only transition differences collapsed',
  max_pa:MAX_PA,
  n_halves:rows.length,
  rows,
};
fs.mkdirSync(path.dirname(OUTPUT),{recursive:true});
fs.writeFileSync(OUTPUT,JSON.stringify(payload));
console.log(JSON.stringify({n_halves:rows.length,method:payload.evaluation_method},null,2));
